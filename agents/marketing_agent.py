"""Marketing Agent — release-status-aware marketing analysis.

For an upcoming title, this tracks the campaign so far: what channels and
tactics are being used, and early signs of what's landing. For a released
title, it's a genuine retrospective: what marketing strategies were used,
what demonstrably worked or underperformed against the film's actual
outcome, and concrete lessons_learned, this is the piece that answers
"I missed this movie when it came out, what happened with it and what
should a studio take away from it."

lessons_learned is deliberately left empty for upcoming titles in the
instruction below: there's no outcome yet to learn a lesson from, and
guessing one would be exactly the kind of unearned confidence this
pipeline has tried to avoid elsewhere (see entity_context.py's confidence
field for the same principle applied to disambiguation).
"""

from __future__ import annotations

import asyncio

import os

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from parallel import Parallel

from agents.entity_context import GEMINI_MODEL
from agents.schemas import MarketingResult

_client = Parallel(api_key=os.environ["PARALLEL_API_KEY"])


async def get_marketing_analysis(title: str, release_year: str, session_id: str, release_status: str) -> dict:
    """Search the web for marketing strategy and, for released titles, outcome.

    Args:
        title: Canonical film title.
        release_year: Release year as a string, or "" if unknown.
        session_id: Parallel session id shared across this pipeline run.
        release_status: "released", "upcoming", or "unclear". Determines
            whether this asks about ongoing campaign tactics or a
            completed campaign's actual results.

    Returns:
        dict with key "results": a list of {url, title, excerpts}.
    """
    if release_status == "released":
        objective = (
            f"What marketing and promotional strategies did {title}"
            + (f" ({release_year})" if release_year else "")
            + " use (trailers, posters, social media campaigns, influencer "
            "partnerships, promotional screenings, tie-ins)? How effective "
            "was the campaign in hindsight, given how the film actually "
            "performed and was received? Look for retrospective analysis, "
            "marketing case studies, or industry commentary specifically "
            "evaluating what worked and what didn't. If no dedicated "
            "marketing analysis exists (common for older or smaller "
            "releases), identify what promotional materials and approach "
            "the film actually used, from ordinary coverage of its trailer, "
            "poster, or release campaign, rather than requiring a formal "
            "case study to exist."
        )
        search_queries = [f"{title} marketing campaign analysis", f"{title} marketing strategy review"]
    else:
        objective = (
            f"What marketing and promotional strategies is {title}"
            + (f" ({release_year})" if release_year else "")
            + " using so far (trailers, posters, social media, partnerships, "
            "promotional events)? Focus on what's been announced or observed, "
            "and any early commentary on whether the campaign is landing."
        )
        search_queries = [f"{title} marketing campaign", f"{title} promotion strategy"]

    search = await asyncio.to_thread(
        _client.search,
        objective=objective,
        search_queries=search_queries,
        session_id=session_id or None,
        mode="fast",
    )
    return {
        "results": [
            {"url": r.url, "title": r.title, "excerpts": r.excerpts}
            for r in search.results
        ]
    }


marketing_tool = FunctionTool(func=get_marketing_analysis)

marketing_agent = Agent(
    name="marketing_agent",
    model=GEMINI_MODEL,
    description="Analyzes marketing strategy — retrospectively for released titles, in-progress for upcoming ones.",
    instruction=(
        "You have a tool, get_marketing_analysis, that searches the web for "
        "marketing strategy and (for released titles) outcome. You will be "
        "given title, release_year, session_id, and release_status as "
        "key=value pairs; parse them and call the tool exactly once. Then "
        "act as a film marketing strategist reviewing this campaign. "
        "Populate: strategies_observed (named tactics/channels actually "
        "mentioned in the excerpts — be specific, 'social media' is weaker "
        "than 'a Twitter-native audio teaser'); what_worked and "
        "what_underperformed as attributed claims with source URLs. "
        "Your job is the CAMPAIGN ITSELF — tactics, channels, timing, "
        "creative choices — not the audience's opinion of the film (that's "
        "sentiment's job) and not bare production facts like cast or crew "
        "(that's news's job); if a fact doesn't describe a promotional "
        "tactic or its effect, it doesn't belong here even if it's "
        "otherwise true and interesting. "
        "CRITICAL — release_status changes what what_worked/"
        "what_underperformed mean: if release_status is 'released', they "
        "are verdicts judged against the actual outcome. If 'upcoming', "
        "there IS NO outcome yet — do not phrase these as verdicts on "
        "success or failure, only populate with genuine early signals the "
        "excerpts support, and leave either list empty if there's nothing "
        "signal-level to report. Never state or imply a campaign 'worked' "
        "or 'failed' for a title that hasn't released. lessons_learned "
        "follows the same rule: only for 'released' titles with excerpts "
        "supporting a concrete takeaway."
    ),
    tools=[marketing_tool],
    output_schema=MarketingResult,
)
