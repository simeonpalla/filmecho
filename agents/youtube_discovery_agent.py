"""YouTube Discovery Agent — search.list, finds trailer video IDs.

First half of the sequential YouTube pair: this must run before
youtube_data_agent, which needs the video IDs this agent returns. It is a
plain data-fetch step (no LLM reasoning needed to run a search.list call),
so it's a regular function rather than an ADK LlmAgent; the orchestration
pipeline calls it directly and feeds its output into the Data Agent.
"""

from __future__ import annotations

import os

from googleapiclient.discovery import build

_YOUTUBE = build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])


def discover_trailer_videos(title: str, release_year: str = "", max_results: int = 5) -> list[dict]:
    """Find candidate trailer video IDs for a title via YouTube search.list.

    Args:
        title: Canonical film title.
        release_year: Release year as a string, used to disambiguate
            remakes/re-releases/anniversary re-uploads. Pass "" if unknown.
        max_results: Number of candidate videos to return. YouTube caps
            search.list at 50 per call; a single trailer search rarely
            needs more than 5-10, so this pipeline doesn't paginate here.

    Returns:
        List of dicts: {video_id, title, channel_title, published_at},
        ordered by YouTube's relevance ranking. Empty list if nothing is
        found or the API call fails — callers should treat that as a
        signal to skip the YouTube branch, not as a fatal error.
    """
    query = f"{title} {release_year} official trailer".strip()
    try:
        response = _YOUTUBE.search().list(
            q=query,
            part="id,snippet",
            type="video",
            maxResults=max_results,
            order="relevance",
        ).execute()
    except Exception as exc:  # noqa: BLE001 - degrade, don't crash the pipeline
        print(f"[youtube_discovery_agent] search.list failed: {exc}")
        return []

    videos = []
    for item in response.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        if not video_id:
            continue
        snippet = item.get("snippet", {})
        videos.append(
            {
                "video_id": video_id,
                "title": snippet.get("title"),
                "channel_title": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
            }
        )
    return videos
