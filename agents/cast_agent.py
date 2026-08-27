"""Cast Agent — per-actor performance reception, via Parallel Search.

This was a real gap: nothing in the pipeline previously searched for how
individual performances landed, only trailer/box-office sentiment as a
whole. Works in both release-status modes: for an upcoming title this
picks up early buzz/anticipation around specific cast members; for a
released title it picks up actual performance reviews.
"""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from parallel import Parallel

from agents.entity_context import GEMINI_MODEL
from agents.schemas import CastResult

_client = Parallel(api_key=os.environ["PARALLEL_API_KEY"])


def get_cast_reception(title: str, release_year: str, cast: str, session_id: str, release_status: str) -> dict:
    """Search the web for reception of specific cast members' performances.

    Args:
        title: Canonical film title.
        release_year: Release year as a string, or "" if unknown.
        cast: Comma-separated list of top-billed cast members (from entity
            resolution), so the search can target named performances
            rather than a generic "the cast" query.
        session_id: Parallel session id shared across this pipeline run.
        release_status: "released", "upcoming", or "unclear". Changes
            whether this asks about actual performance reviews or
            early anticipation for specific actors.

    Returns:
        dict with key "results": a list of {url, title, excerpts}.
    """
    if release_status == "released":
        objective = (
            f"How were the individual actors' performances in {title}"
            + (f" ({release_year})" if release_year else "")
            + f" received by critics and audiences? Cast includes: {cast or 'unknown'}. "
            "Focus on specific praise or criticism of named performances, "
            "any standout or breakout performance, and any performances "
            "singled out as weak or miscast."
        )
    else:
        objective = (
            f"What is being said about the cast of {title}"
            + (f" ({release_year})" if release_year else "")
            + f"? Cast includes: {cast or 'unknown'}. Focus on anticipation or "
            "buzz around specific actors, casting reactions, and any early "
            "praise or skepticism about individual performers."
        )

    search = _client.search(
        objective=objective,
        search_queries=[f"{title} cast performance reviews", f"{title} acting reactions"],
        session_id=session_id or None,
        mode="fast",
    )
    return {
        "results": [
            {"url": r.url, "title": r.title, "excerpts": r.excerpts}
            for r in search.results
        ]
    }


cast_tool = FunctionTool(func=get_cast_reception)

cast_agent = Agent(
    name="cast_agent",
    model=GEMINI_MODEL,
    description="Searches per-actor performance reception via Parallel Search.",
    instruction=(
        "You have a tool, get_cast_reception, that searches the web for "
        "reception of specific cast members. You will be given title, "
        "release_year, cast, session_id, and release_status as key=value "
        "pairs; parse them and call the tool exactly once. Then populate "
        "performances with one entry per actor the excerpts actually "
        "discuss (actor name, a specific note on their reception, source "
        "URL), standout_performance naming whoever got the strongest praise "
        "(empty string if none stood out), and overall_cast_reception as "
        "one sentence. Do not invent a reception for an actor the excerpts "
        "don't mention, leave performances covering only the actors you "
        "actually found material on."
    ),
    tools=[cast_tool],
    output_schema=CastResult,
)
