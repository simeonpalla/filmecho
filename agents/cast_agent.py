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
from google.genai import types as genai_types
from parallel import Parallel

from agents.entity_context import GEMINI_MODEL, GROUNDING_TEMPERATURE
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
    # A known cast list sharpens the search; an unknown one should never
    # block it — for large ensemble/crossover films (e.g. a multi-hero
    # team-up), entity resolution deliberately leaves cast empty rather
    # than guess at an arbitrary "top 5" from a dozen-plus co-leads. The
    # objective below must read as a normal research task either way,
    # not as a task that's missing a required input.
    known_cast_clause = f" Known cast so far: {cast}." if cast else ""
    if release_status == "released":
        objective = (
            f"How were the individual actors' performances in {title}"
            + (f" ({release_year})" if release_year else "")
            + " received by critics and audiences? Identify the specific "
            "actors discussed in your search results yourself if a cast "
            "list isn't given below." + known_cast_clause + " "
            "Focus on specific praise or criticism of named performances, "
            "any standout or breakout performance, and any performances "
            "singled out as weak or miscast. SEPARATELY, also look for "
            "behind-the-scenes or personal news about cast members: "
            "injuries during filming, remuneration or salary "
            "disputes, on-set incidents, personal-life updates (travel, "
            "relationships, controversies) connected to this production or "
            "its promotion, not generic celebrity gossip unrelated to it."
            + region_clause
        )
    else:
        objective = (
            f"What is being said about the cast of {title}"
            + (f" ({release_year})" if release_year else "")
            + "? Identify the specific actors discussed in your search "
            "results yourself if a cast list isn't given below."
            + known_cast_clause + " Focus on anticipation or "
            "buzz around specific actors, casting reactions, and any early "
            "praise or skepticism about individual performers. SEPARATELY, "
            "also look for behind-the-scenes or personal news about cast "
            "members: injuries during filming, remuneration "
            "or salary disputes, on-set incidents, personal-life updates "
            "connected to this production or its promotion." + region_clause
        )

    search = await asyncio.to_thread(
        _client.search,
        objective=objective,
        search_queries=[
            f"{title} cast list",
            f"{title} cast performance reviews",
            f"{title} acting reactions",
            f"{title} biggest casting reveal fan reaction",
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
        "(cast and region_hint may both be empty — an empty cast is NORMAL "
        "for large ensemble films where entity resolution couldn't pick a "
        "definitive top-billed handful, it is NOT missing information you "
        "need to ask for; the search results themselves will name the "
        "actual actors, use those). Then act as "
        "an entertainment journalist who covers performance criticism and "
        "industry-insider reporting specifically — not a general news "
        "reporter. Populate performances with one entry per actor the "
        "excerpts discuss WITH SOME ATTACHED AUDIENCE OR CRITIC REACTION "
        "(actor name, a specific note on that reception). Draw a careful "
        "line here: a bare, reaction-free fact like 'X was cast as Y, "
        "confirmed by the studio' with nothing else belongs to the news "
        "agent, not here — but genuine excitement, anticipation, "
        "skepticism, surprise, or debate ABOUT a casting choice absolutely "
        "counts as reception and belongs here, especially for an upcoming "
        "title where there's no performance footage yet to review and "
        "casting reaction IS the reception signal available. A single "
        "actor's casting can be the single biggest story surrounding a "
        "film (a franchise veteran's surprise return, a high-profile "
        "reveal) — don't let that story's 'announcement' framing make you "
        "skip them: if the excerpts show ANY audience or press reaction to "
        "it, they get a performances entry, not just a mention elsewhere. "
        "standout_performance naming whoever got the strongest praise "
        "(empty string if none stood out), overall_cast_reception as one "
        "sentence, and personal_updates with any behind-the-scenes or "
        "personal news — injuries, salary/remuneration disputes, on-set "
        "incidents, personal-life updates connected to this production. "
        "CRITICAL — a personal_updates headline stating someone's "
        "involvement status (returning, not returning, joining, leaving) "
        "must reflect how settled that claim actually is: if your own "
        "search results disagree about the same person's involvement, or "
        "the source itself hedges ('reportedly', 'sources say', "
        "'rumored'), write the headline that way too ('Reportedly not "
        "returning, per [outlet]' rather than a flat 'Not returning') — "
        "don't launder a contested or hedged claim into a confident-"
        "sounding headline just because that reads cleaner. This applies "
        "especially to involvement/casting claims, which are exactly the "
        "kind of thing that leaks, changes, or gets reported prematurely. "
        "Your job is specifically PEOPLE, not the production itself — a "
        "fact like 'filming took place in Kochi' or 'the film is a remake' "
        "belongs to the news agent, not here, even if it's true and "
        "interesting. Keep performances (professional reception) and "
        "personal_updates (behind-the-scenes news) strictly separate, "
        "don't mix them. Do not invent a reception or personal update for "
        "an actor the excerpts don't mention. NEVER write a refusal or "
        "meta-comment like 'I need to know the cast members first' into "
        "any field — if the search genuinely returned nothing usable, "
        "leave performances empty and set overall_cast_reception to a "
        "plain factual statement of that ('No cast-specific reception "
        "surfaced in this run.'), not an explanation of what you'd need."
    ),
    tools=[cast_tool],
    output_schema=CastResult,
    generate_content_config=genai_types.GenerateContentConfig(temperature=GROUNDING_TEMPERATURE),
)
