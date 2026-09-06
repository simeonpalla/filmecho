"""In-memory cache for completed Greenlight Memo pipeline runs.

Why this exists, in two parts:

1. A *released* film's underlying facts (box office, cast reception, what
   critics said) don't change from one query to the next — but this
   pipeline re-runs live Parallel search + Gemini synthesis on every
   call, with no seeding between calls, so the exact same title produced
   noticeably different confidence scores and even different box-office
   figures across repeat runs during testing.

2. Less obviously, but just as real: even an *upcoming* title's memo
   swung 78% -> 90% confidence one minute apart during testing, and
   diffing the two runs showed why — Parallel's live search returned
   MEANINGFULLY DIFFERENT excerpts each call (one run reported "Chris
   Evans not returning" as a rumor, the very next run stated "Chris
   Evans will return" as a news fact). That's input-level variance, not
   model sampling — no amount of lowering the LLM's temperature fixes it,
   because the two calls are reasoning over genuinely different source
   material. The real world did not change in that one minute; the
   search index's response did.

Policy, matching how a real archive would treat this:
- release_status == "released": cache indefinitely once computed. Serve
  the cached memo on every subsequent request for the same title unless
  the caller explicitly asks for force_refresh (e.g. a "Refresh this
  memo" button in the UI) — that's the escape hatch for the re-release /
  sudden-relevance case, not an automatic timer, since guessing at
  "sudden renewed interest" from inside this pipeline would be its own
  source of flakiness.
- release_status in ("upcoming", "unclear"): cached too, but with a
  short TTL (see UPCOMING_CACHE_TTL_SECONDS in pipeline.py) rather than
  never. Pre-release buzz does genuinely move over days, so this can't
  be frozen forever like a released title — but it does NOT genuinely
  move minute-to-minute, so a short window trades away zero real
  freshness while eliminating the exact flakiness above. force_refresh
  still bypasses this immediately when someone wants guaranteed-current
  data right now.

This is a plain in-memory dict, not a database. That's a deliberate
scope decision for the hackathon deadline, not an oversight — see the
`persist` hook below for how to swap in real persistence (Firestore,
Redis, a mounted volume) later without touching call sites. In-memory
means the cache resets on every Cloud Run cold start/redeploy, which is
an acceptable trade for "consistent within a demo session" but NOT a
substitute for real persistence if this ships past the hackathon.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _cache_key(canonical_title: str, release_year: Optional[int]) -> str:
    """Normalize (title, year) into a stable cache key.

    Lowercased and punctuation-stripped so "R.R.R", "RRR", and "rrr " all
    land on the same entry — the whole point is that re-typing the same
    film shouldn't miss the cache over formatting noise. Year is part of
    the key specifically to keep genuine remakes/same-title films apart
    (e.g. two different films both called "Doomsday").
    """
    slug = re.sub(r"[^a-z0-9]+", "", (canonical_title or "").lower())
    return f"{slug}:{release_year or 'unknown'}"


@dataclass
class _CacheEntry:
    payload: dict
    cached_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def age_seconds(self) -> float:
        cached_dt = datetime.fromisoformat(self.cached_at)
        return (datetime.now(timezone.utc) - cached_dt).total_seconds()


class MemoCache:
    """Thread-safe in-memory store for completed pipeline payloads."""

    def __init__(self) -> None:
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, canonical_title: str, release_year: Optional[int], ttl_seconds: Optional[float] = None) -> Optional[dict]:
        """Return a cached payload plus its cache metadata, or None on a
        miss OR an expired entry.

        ttl_seconds=None means "never expires" (the released-title
        policy). A numeric ttl_seconds treats an entry older than that
        as a miss — it is NOT deleted here, just not served, so a
        concurrent request that's already mid-flight recomputing it
        doesn't race against this one deleting the same key.
        """
        key = _cache_key(canonical_title, release_year)
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return None
        if ttl_seconds is not None and entry.age_seconds() > ttl_seconds:
            return None
        return {**entry.payload, "served_from_cache": True, "cached_at": entry.cached_at}

    def set(self, canonical_title: str, release_year: Optional[int], payload: dict) -> None:
        key = _cache_key(canonical_title, release_year)
        with self._lock:
            self._entries[key] = _CacheEntry(payload=payload)

    def invalidate(self, canonical_title: str, release_year: Optional[int]) -> None:
        """Drop a single entry — used before a forced refresh so a failed
        recompute doesn't leave a stale-but-still-served entry behind."""
        key = _cache_key(canonical_title, release_year)
        with self._lock:
            self._entries.pop(key, None)

    def stats(self) -> dict:
        """Cheap introspection for a debug endpoint or the activity log."""
        with self._lock:
            return {"entries": len(self._entries), "keys": sorted(self._entries.keys())}


# Module-level singleton: one process-wide cache, imported by both the
# pipeline (to read/write it) and app.py (to expose /api/cache/stats and
# a manual-invalidate path) without passing an instance through every
# call site.
memo_cache = MemoCache()
