"""YouTube Data Agent — videos.list (stats) + commentThreads.list (comments).

Second half of the sequential YouTube pair. Takes the video IDs found by
youtube_discovery_agent and pulls view/like/comment counts plus a sample of
top-level comments for each. This is the more implementation-heavy of the
two YouTube agents (two endpoints, a dependency on Discovery, pagination on
comments), so budget real time for it, per the roadmap's build order.
"""

from __future__ import annotations

import os

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

_YOUTUBE = build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])


def get_video_stats(video_ids: list[str]) -> dict[str, dict]:
    """Fetch view/like/comment counts for up to 50 video IDs in one call.

    Args:
        video_ids: List of YouTube video IDs (only the first 50 are used;
            videos.list doesn't accept more per call).

    Returns:
        Dict keyed by video_id, each value {title, view_count, like_count,
        comment_count, published_at}. Missing/deleted videos are simply
        absent from the result rather than raising.
    """
    if not video_ids:
        return {}
    try:
        response = _YOUTUBE.videos().list(
            part="statistics,snippet",
            id=",".join(video_ids[:50]),
        ).execute()
    except Exception as exc:  # noqa: BLE001
        print(f"[youtube_data_agent] videos.list failed: {exc}")
        return {}

    stats = {}
    for item in response.get("items", []):
        statistics = item.get("statistics", {})
        snippet = item.get("snippet", {})
        thumbnails = snippet.get("thumbnails", {})
        thumb = thumbnails.get("high") or thumbnails.get("medium") or thumbnails.get("default") or {}
        stats[item["id"]] = {
            "title": snippet.get("title"),
            "view_count": int(statistics.get("viewCount", 0)),
            "like_count": int(statistics.get("likeCount", 0)),
            "comment_count": int(statistics.get("commentCount", 0)),
            "thumbnail_url": thumb.get("url", ""),
            "published_at": snippet.get("publishedAt", ""),
        }
    return stats


def get_top_comments(video_id: str, max_results: int = 50) -> list[dict]:
    """Fetch top-level comments for a single video, paginating as needed.

    Args:
        video_id: The video to pull comments for.
        max_results: Total comments to collect across pages (YouTube caps
            each page at 100).

    Returns:
        List of {text, like_count, author}. Returns an empty list, not an
        error, when a video has comments disabled — that's an expected
        state (YouTube reports it as an HTTP 403), not a bug to surface.
    """
    comments: list[dict] = []
    page_token = None
    try:
        while len(comments) < max_results:
            response = _YOUTUBE.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                order="relevance",
                pageToken=page_token,
                textFormat="plainText",
            ).execute()
            for item in response.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                comments.append(
                    {
                        "text": top.get("textDisplay", ""),
                        "like_count": top.get("likeCount", 0),
                        "author": top.get("authorDisplayName", ""),
                    }
                )
            page_token = response.get("nextPageToken")
            if not page_token:
                break
    except HttpError as exc:
        if exc.resp.status == 403:
            print(f"[youtube_data_agent] Comments disabled for {video_id}, skipping.")
        else:
            print(f"[youtube_data_agent] commentThreads.list failed for {video_id}: {exc}")
    return comments


def collect_youtube_data(video_ids: list[str], comments_per_video: int = 30) -> dict[str, dict]:
    """Full YouTube Data Agent step: stats + comments for discovered videos.

    Args:
        video_ids: Output of youtube_discovery_agent.discover_trailer_videos.
        comments_per_video: How many top-level comments to pull per video.

    Returns:
        Dict keyed by video_id, each entry extending get_video_stats' output
        with a "comments" list.
    """
    stats = get_video_stats(video_ids)
    for video_id in stats:
        stats[video_id]["comments"] = get_top_comments(video_id, comments_per_video)
    return stats
