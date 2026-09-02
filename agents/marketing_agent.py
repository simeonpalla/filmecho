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


async def get_marketing_analysis(
    title: str, release_year: str, session_id: str, release_status: str, region_hint: str = ""
) -> dict:
    """Search the web for marketing strategy and, for released titles, outcome.

    Args:
        title: Canonical film title.
        release_year: Release year as a string, or "" if unknown.
        session_id: Parallel session id shared across this pipeline run.
        release_status: "released", "upcoming", or "unclear". Determines
            whether this asks about ongoing campaign tactics or a
            completed campaign's actual results.
        region_hint: Optional locale/timezone string from the requesting
            browser, used to bias toward region-specific campaign elements
            (regional distributors, local promotional partners) rather
            than a global/US-only default. Empty string if unavailable.

    Returns:
        dict with key "results": a list of {url, title, excerpts}.
    """
    region_clause = (
        f" Include any {region_hint}-specific marketing activity (regional "
        "distributor campaigns, local promotional partnerships, market-"
        "specific release strategy) alongside the global campaign."
        if region_hint else ""
    )
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
            "case study to exist." + region_clause
        )
        search_queries = [f"{title} marketing campaign analysis", f"{title} marketing strategy review"]
    else:
        objective = (
            f"What marketing and promotional strategies is {title}"
            + (f" ({release_year})" if release_year else "")
            + " using so far (trailers, posters, social media, partnerships, "
            "promotional events)? Focus on what's been announced or observed, "
            "and any early commentary on whether the campaign is landing."
            + region_clause
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
        "given title, release_year, session_id, release_status, and "
        "region_hint as key=value pairs; parse them and call the tool "
        "exactly once (region_hint may be empty, pass it through as "
        "given).\n\n"
        "CRITICAL GROUNDING RULE, read this before writing anything: "
        "every claim you write must be something a specific source excerpt "
        "actually said. NEVER write a specific number, percentage, or "
        "statistic (e.g. '34% higher conversion', '3.2 million shares', "
        "'847 fan theories') unless that exact figure appears verbatim in "
        "the excerpts you were given. If you want to convey that something "
        "performed well and the excerpts don't give you a number, say so "
        "qualitatively ('generated strong organic engagement, per "
        "[source]') — inventing a precise-sounding statistic to make a "
        "vague claim seem more authoritative is fabrication, and it is "
        "worse than an honest qualitative claim, not better. Marketing "
        "trade coverage and SEO content sites are also known to publish "
        "inflated or invented statistics themselves — if a source's claim "
        "reads as suspiciously precise with no clear methodology, either "
        "attribute it explicitly as that outlet's claim rather than fact, "
        "or leave it out.\n\n"
        "Then act as a film marketing strategist reviewing this campaign, "
        "but a disciplined one who'd rather say less and be right: "
        "populate strategies_observed (up to 6, each a named tactic/channel "
        "actually mentioned in the excerpts, with its source — 3 "
        "well-attributed tactics beats 6 vague ones), and what_worked / "
        "what_underperformed (up to 4 each). Your job is the CAMPAIGN "
        "ITSELF — tactics, channels, timing, creative choices — not the "
        "audience's opinion of the film (that's sentiment's job) and not "
        "bare production facts like cast or crew (that's news's job); if a "
        "fact doesn't describe a promotional tactic or its effect, it "
        "doesn't belong here.\n\n"
        "CRITICAL — release_status changes what what_worked/"
        "what_underperformed mean: if release_status is 'released', they "
        "are verdicts judged against the actual outcome. If 'upcoming', "
        "there IS NO outcome yet — do not phrase these as verdicts on "
        "success or failure, only populate with genuine early signals the "
        "excerpts support, and leave either list empty if there's nothing "
        "signal-level to report. Never state or imply a campaign 'worked' "
        "or 'failed' for a title that hasn't released. lessons_learned "
        "(up to 4) follows the same rule: only for 'released' titles with "
        "excerpts supporting a concrete takeaway."
    ),
    tools=[marketing_tool],
    output_schema=MarketingResult,
)
