"""Tests for orchestration/memo_cache.py.

No API keys or network needed — this is pure in-memory logic. What's
tested:
- A miss returns None, a set-then-get round-trips the payload.
- The cache key normalizes title formatting/case so re-typing the same
  film doesn't accidentally miss.
- Two different years for the same title are kept as separate entries
  (same-titled films released in different years must not collide).
- invalidate() actually removes an entry rather than just marking it.
- Each cache instance is independent — tests don't leak state into each
  other via the module-level singleton.
"""

from orchestration.memo_cache import MemoCache


class TestBasicRoundTrip:
    def test_miss_returns_none(self):
        cache = MemoCache()
        assert cache.get("RRR", 2022) is None

    def test_set_then_get_round_trips_payload(self):
        cache = MemoCache()
        payload = {"entity": {"title": "RRR"}, "result": {"verdict": "greenlight"}}
        cache.set("RRR", 2022, payload)

        cached = cache.get("RRR", 2022)

        assert cached is not None
        assert cached["entity"]["title"] == "RRR"
        assert cached["result"]["verdict"] == "greenlight"

    def test_get_adds_cache_metadata_without_mutating_stored_payload(self):
        cache = MemoCache()
        payload = {"entity": {"title": "RRR"}}
        cache.set("RRR", 2022, payload)

        cached = cache.get("RRR", 2022)

        assert cached["served_from_cache"] is True
        assert "cached_at" in cached
        # The original dict passed to set() must not have been mutated —
        # otherwise a caller holding a reference to it would see cache
        # bookkeeping leak into what they thought was their own object.
        assert "served_from_cache" not in payload


class TestKeyNormalization:
    def test_title_case_and_punctuation_are_normalized(self):
        cache = MemoCache()
        cache.set("R.R.R", 2022, {"entity": {"title": "R.R.R"}})

        assert cache.get("rrr", 2022) is not None
        assert cache.get("RRR", 2022) is not None
        assert cache.get("  R R R  ", 2022) is not None

    def test_different_years_are_separate_entries(self):
        """Same title, different films — e.g. two movies both called
        'Doomsday' — must not collide into one cache entry."""
        cache = MemoCache()
        cache.set("Doomsday", 2008, {"entity": {"title": "Doomsday", "release_year": 2008}})
        cache.set("Doomsday", 2026, {"entity": {"title": "Doomsday", "release_year": 2026}})

        assert cache.get("Doomsday", 2008)["entity"]["release_year"] == 2008
        assert cache.get("Doomsday", 2026)["entity"]["release_year"] == 2026

    def test_missing_year_still_caches_distinctly_from_a_known_year(self):
        cache = MemoCache()
        cache.set("Toxic", None, {"entity": {"title": "Toxic"}})
        cache.set("Toxic", 2025, {"entity": {"title": "Toxic", "release_year": 2025}})

        assert cache.get("Toxic", None)["entity"].get("release_year") is None
        assert cache.get("Toxic", 2025)["entity"]["release_year"] == 2025


class TestInvalidate:
    def test_invalidate_removes_the_entry(self):
        cache = MemoCache()
        cache.set("RRR", 2022, {"entity": {"title": "RRR"}})
        assert cache.get("RRR", 2022) is not None

        cache.invalidate("RRR", 2022)

        assert cache.get("RRR", 2022) is None

    def test_invalidate_on_missing_key_does_not_raise(self):
        cache = MemoCache()
        cache.invalidate("Never Cached", 1999)  # should be a no-op, not an error


class TestStats:
    def test_stats_reflects_entry_count(self):
        cache = MemoCache()
        assert cache.stats()["entries"] == 0

        cache.set("RRR", 2022, {})
        cache.set("Toxic", 2025, {})

        stats = cache.stats()
        assert stats["entries"] == 2
        assert len(stats["keys"]) == 2
