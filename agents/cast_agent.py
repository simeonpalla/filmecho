"""Cast Agent — per-actor performance reception, via Parallel Search.

This was a real gap: nothing in the pipeline previously searched for how
individual performances landed, only trailer/box-office sentiment as a
whole. Works in both release-status modes: for an upcoming title this
picks up early buzz/anticipation around specific cast members; for a
released title it picks up actual performance reviews.
"""

from __future__ import annotations

import asyncio

import os

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from parallel import Parallel

from agents.entity_context import GEMINI_MODEL
from agents.schemas import CastResult

_client = Parallel(api_key=os.environ["PARALLEL_API_KEY"])


async def get_cast_reception(
    title: str, release_year: str, cast: str, session_id: str, release_status: str, region_hint: str = ""
) -> dict:
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
        region_hint: Optional locale/timezone string from the requesting
            browser, used to prioritize region-relevant cast reception
            (e.g. a regional star's local fanbase reaction). Empty string
            if unavailable.

    Returns:
        dict with key "results": a list of {url, title, excerpts}.
    """
    region_clause = (
        f" Prioritize reception from the {region_hint} audience/press if "
        "distinguishable."
        if region_hint else ""
    )
    if release_status == "released":
        objective = (
            f"How were the individual actors' performances in {title}"
            + (f" ({release_year})" if release_year else "")
            + f" received by critics and audiences? Cast includes: {cast or 'unknown'}. "
            "Focus on specific praise or criticism of named performances, "
            "any standout or breakout performance, and any performances "
            "singled out as weak or miscast. SEPARATELY, also look for "
            "behind-the-scenes or personal news about these specific cast "
            "members: injuries during filming, remuneration or salary "
            "disputes, on-set incidents, personal-life updates (travel, "
            "relationships, controversies) connected to this production or "
            "its promotion, not generic celebrity gossip unrelated to it."
            + region_clause
        )
    else:
        objective = (
            f"What is being said about the cast of {title}"
            + (f" ({release_year})" if release_year else "")
            + f"? Cast includes: {cast or 'unknown'}. Focus on anticipation or "
            "buzz around specific actors, casting reactions, and any early "
            "praise or skepticism about individual performers. SEPARATELY, "
            "also look for behind-the-scenes or personal news about these "
            "specific cast members: injuries during filming, remuneration "
            "or salary disputes, on-set incidents, personal-life updates "
            "connected to this production or its promotion." + region_clause
        )

    search = await asyncio.to_thread(
        _client.search,
        objective=objective,
        search_queries=[
            f"{title} cast performance reviews",
            f"{title} acting reactions",
            f"{title} cast injury controversy salary",
        ],
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
        "release_year, cast, session_id, release_status, and region_hint as "
        "key=value pairs; parse them and call the tool exactly once "
        "(region_hint may be empty, pass it through as given). Then act as "
        "an entertainment journalist who covers performance criticism and "
        "industry-insider reporting specifically — not a general news "
        "reporter. Populate performances with one entry per actor the "
        "excerpts actually discuss (actor name, a specific note on their "
        "reception — cite what critics/fans actually said about the "
        "performance itself, not casting announcements), "
        "standout_performance naming whoever got the strongest praise "
        "(empty string if none stood out), overall_cast_reception as one "
        "sentence, and personal_updates with any behind-the-scenes or "
        "personal news — injuries, salary/remuneration disputes, on-set "
        "incidents, personal-life updates connected to this production. "
        "Your job is specifically PEOPLE, not the production itself — a "
        "fact like 'filming took place in Kochi' or 'the film is a remake' "
        "belongs to the news agent, not here, even if it's true and "
        "interesting. Keep performances (professional reception) and "
        "personal_updates (behind-the-scenes news) strictly separate, "
        "don't mix them. Do not invent a reception or personal update for "
        "an actor the excerpts don't mention."
    ),
    tools=[cast_tool],
    output_schema=CastResult,
)
