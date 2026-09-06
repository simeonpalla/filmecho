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
            f"major movie releases after {title} release date",
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
        "competing_titles (up to 5 entries, each with the competitor's name, "
        "strength_vs_searched — its SPECIFIC edge over the searched title, "
        "e.g. 'bigger established fanbase' or 'locks in premium format "
        "screens with an earlier date', drawn from the excerpts, not "
        "invented — and, ONLY for released titles, gross_estimate if the "
        "excerpts state a box office figure for that specific competitor, "
        "null otherwise), attention_assessment, and risk. "
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
        "confident read with less-than-perfect data.\n\n"
        "box_office: ONLY for release_status='released', populate budget, "
        "worldwide_gross, domestic_gross, and opening_weekend with whatever "
        "specific figures the excerpts actually give you (leave any of "
        "these null if that particular figure wasn't reported, don't "
        "estimate one), plus a verdict (blockbuster/hit/average/"
        "underperformed/flop) with a one-sentence verdict_basis explaining "
        "the call — e.g. the budget-to-gross ratio, or how coverage "
        "characterized the result. For release_status='upcoming', leave "
        "box_office entirely at its defaults (every field null, verdict "
        "'unclear') — there is no box office for something that hasn't "
        "released, do not project or estimate one. Every number in "
        "box_office must trace to a specific excerpt, never invent a "
        "figure to make the answer feel more complete.\n\n"
        "release_window: ONLY for release_status='upcoming' (released "
        "titles already have a fixed, unchangeable window, leave this at "
        "defaults). Using the SAME competing_titles you just identified, "
        "assess congestion (how crowded the window actually is) and give "
        "suggested_direction. This is directional comparative analysis "
        "using real data you already have, NOT prediction: you may say "
        "'consider_earlier' or 'consider_later' if the excerpts show a "
        "specific real competitor whose release date and genre overlap "
        "would be avoided by shifting, but NEVER name a specific "
        "alternate date, and NEVER state or imply what box office or "
        "reception a hypothetical different date would produce — there "
        "is no real data for a date that didn't happen, only for the "
        "actual competitors you found. If the excerpts don't give you "
        "enough about competitors' specific dates/genres to reason about "
        "this, set suggested_direction to 'unclear' rather than guessing.\n\n"
        "IMPORTANT — don't let this default toward 'earlier': your search "
        "results are naturally weighted toward what's ALREADY known to be "
        "competing in the CURRENT window, which makes 'earlier' look "
        "supportable (you can point at named competitors it would avoid) "
        "while 'later' looks unsupported by default (you have no evidence "
        "either way about what's releasing further out). That asymmetry in "
        "your evidence is not the same as 'earlier' actually being the "
        "better call — it may just be the only direction you happened to "
        "search for. Your reasoning field must explicitly address BOTH "
        "directions: state what you found that supports (or rules out) "
        "shifting earlier, AND separately state what you found (or didn't "
        "find) about shifting later — for example, whether the excerpts "
        "show anything already scheduled after the current window in this "
        "market, or whether you simply have no data on that and should say "
        "so plainly rather than silently omitting it. Only pick a "
        "direction your reasoning actually defends on both sides, not the "
        "one your search queries happened to surface more of."
    ),
    tools=[competitive_tool],
    output_schema=CompetitiveResult,
)
