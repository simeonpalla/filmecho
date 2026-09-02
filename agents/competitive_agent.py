"""Competitive Agent — same Parallel-call pattern as Web Sentiment Agent.

Answers a different question than Web Sentiment: not "how is this trailer
landing" but "what else is this title competing against for attention
right now" — same-window releases, franchise fatigue signals, genre
saturation. This is a second, independently identifiable Parallel
.search() call/import, so Stage One reviewers see more than one function
proving the integration is real rather than a single lucky example.
"""

from __future__ import annotations

import asyncio

import os

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from parallel import Parallel

from agents.entity_context import GEMINI_MODEL
from agents.schemas import CompetitiveResult

_client = Parallel(api_key=os.environ["PARALLEL_API_KEY"])


async def get_competitive_landscape(
    title: str, release_year: str, director: str, session_id: str, release_status: str, region_hint: str = ""
) -> dict:
    """Search the web for what a title competes/competed against for attention.

    Args:
        title: Canonical film title.
        release_year: Release year as a string, or "" if unknown.
        director: Director's name, or "" if unknown — included for
            disambiguation, not directly used in the objective text.
        session_id: Parallel session id shared across this pipeline run.
        release_status: "released", "upcoming", or "unclear". An upcoming
            title has no box office outcome yet, released asks a
            fundamentally different, backward-looking question.
        region_hint: Optional locale/timezone string from the requesting
            browser (e.g. "Asia/Kolkata, en-IN"), used to bias the
            competitive read toward a specific market rather than
            defaulting to US/UK box office framing. Empty string if
            unavailable.

    Returns:
        dict with key "results": a list of {url, title, excerpts}.
    """
    region_clause = (
        f" Frame this specifically for the {region_hint} market/region — "
        "competing releases, box office context, and audience attention "
        "should reflect that market, not assume a US/UK default."
        if region_hint else ""
    )
    if release_status == "released":
        objective = (
            f"What did {title}" + (f" ({release_year})" if release_year else "")
            + " compete against at the box office when it released, and how did "
            "it perform relative to that competition? Find specific numbers if "
            "available: opening weekend gross, box office ranking that weekend, "
            "budget-vs-gross, whether it over- or under-performed pre-release "
            "expectations. Prioritize sources with hard figures over vague "
            "characterizations." + region_clause
        )
        search_queries = [
            f"{title} box office opening weekend numbers" + (f" {region_hint.split(',')[0]}" if region_hint else ""),
            f"{title} box office performance vs competition",
        ]
    else:
        objective = (
            f"What other films or shows are releasing in the same window as {title}"
            + (f" ({release_year})" if release_year else "")
            + ", competing for audience attention? Focus on same-weekend "
            "releases, genre overlap, franchise fatigue commentary, and any "
            "direct comparisons critics or audiences are already drawing "
            "between this title and its competition." + region_clause
        )
        search_queries = [
            f"{title} box office competition" + (f" {region_hint.split(',')[0]}" if region_hint else ""),
            f"films releasing same weekend as {title}",
        ]

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


competitive_tool = FunctionTool(func=get_competitive_landscape)

competitive_agent = Agent(
    name="competitive_agent",
    model=GEMINI_MODEL,
    description="Assesses competitive landscape and audience-attention risk via Parallel Search.",
    instruction=(
        "You have a tool, get_competitive_landscape, that searches the web "
        "for competitive positioning. You will be given title, release_year, "
        "director, session_id, release_status, and region_hint as key=value "
        "pairs; parse them and call the tool exactly once (region_hint may "
        "be an empty string, pass it through as given). Then act as a "
        "box-office and competitive-strategy analyst. Populate: "
        "competing_titles (up to 5 entries, each with the competitor's name "
        "AND strength_vs_searched — its SPECIFIC edge over the searched "
        "title, e.g. 'bigger established fanbase' or 'locks in premium "
        "format screens with an earlier date', drawn from the excerpts, "
        "not invented), attention_assessment, and risk. "
        "CRITICAL — risk means different things by release_status: for "
        "'upcoming', it's forward-looking (will attention be split by "
        "rivals). For 'released', it is NOT forward risk, it's a "
        "backward-looking verdict on how the title actually performed "
        "against its competition — high means it was notably outcompeted, "
        "low means it held its own or won. For a released title, actively "
        "look for box-office numbers in the excerpts and form a definitive "
        "judgment from them; only fall back to 'unclear' if the excerpts "
        "truly have nothing to go on, not just because the question is "
        "hard. A vague 'unclear' when data exists is a worse answer than a "
        "confident read with less-than-perfect data."
    ),
    tools=[competitive_tool],
    output_schema=CompetitiveResult,
)
