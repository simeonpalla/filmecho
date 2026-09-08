"""Poster Agent — plain data fetch via TMDB, not an LLM agent.

Fetches a real movie poster image for a resolved title via The Movie
Database's (TMDB) free /search/movie endpoint. This exists because the
only image this pipeline had before was the YouTube trailer's video
thumbnail (see youtube_data_agent.py) — a landscape 16:9 video frame,
not a poster, and force-cropping it into a portrait "poster" shape in
the frontend produced a distorted, zoomed-in crop rather than an actual
movie poster.

This is a plain REST GET with no LLM reasoning involved — the same
category as the YouTube Data API calls elsewhere in this pipeline (not
an agent/tool call), which is why it's outside the hackathon's
"AI/agent tooling only" restriction the same way YouTube Data API
already is. Worth re-checking against the current rules text before
relying on that reading, since rules can be interpreted differently or
amended — this file makes no attempt to enforce compliance itself.

Requires a free TMDB API key (themoviedb.org) in TMDB_API_KEY. Degrades
to no poster (returns None, never raises) on a missing key, network
failure, or zero search results — the frontend already has a clean
fallback (a plain title card) for when this comes back empty, so a
missing poster is a normal, expected outcome, not an error state.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import requests

_TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
_TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def _fetch_poster_sync(title: str, release_year: Optional[int]) -> Optional[str]:
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key or not title:
        return None
    try:
        params = {"api_key": api_key, "query": title}
        if release_year:
            params["year"] = release_year
        response = requests.get(_TMDB_SEARCH_URL, params=params, timeout=6)
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            return None
        poster_path = results[0].get("poster_path")
        return f"{_TMDB_IMAGE_BASE}{poster_path}" if poster_path else None
    except Exception as exc:  # noqa: BLE001
        print(f"[poster_agent] TMDB lookup failed: {exc}")
        return None


async def fetch_poster(title: str, release_year: Optional[int]) -> Optional[str]:
    """Async wrapper matching the rest of this pipeline's pattern of
    running blocking I/O off the event loop via asyncio.to_thread (see
    every *_agent.py's Parallel client calls for the same pattern).

    Args:
        title: Canonical film title (post entity-resolution, not the
            raw user-typed string).
        release_year: Release year, if known — narrows the TMDB search
            to the right film when multiple share a title.

    Returns:
        A full poster image URL, or None if unavailable for any reason.
        Never raises.
    """
    return await asyncio.to_thread(_fetch_poster_sync, title, release_year)
