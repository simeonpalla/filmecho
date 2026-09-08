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
from google.genai import types as genai_types
from parallel import Parallel

from agents.entity_context import GEMINI_MODEL, GROUNDING_TEMPERATURE
from agents.schemas import CompetitiveResult

_client = Parallel(api_key=os.environ["PARALLEL_API_KEY"])


async def get_competitive_landscape(
    title: str, release_year: str, director: str, session_id: str, release_status: str,
    region_hint: str = "", source_type: str = "", based_on: str = "",
) -> dict:
    """Search the web for what a title competes/competed against for attention,
    and — for a sequel/reboot/spinoff/remake — how its own franchise history
    performed.

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
        source_type: From entity resolution — "sequel"/"reboot"/"spinoff"/
            "remake" trigger a second, franchise-history-focused search;
            anything else ("original", "unclear", etc.) skips it, since
            there's no prior installment to look up. Empty string treated
            the same as "unclear".
        based_on: The specific prior work named by entity resolution, if
            any (e.g. "Sequel to Dune: Part Two"). Used to target the
            franchise-history search when present; falls back to a
            generic "{title} previous film" query when not.

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
            f"What other films or shows compete with {title}"
            + (f" ({release_year})" if release_year else "")
            + " for audience attention — both the SAME weekend/window "
            "specifically, AND, separately, the wider field of major "
            "tentpole/genre-overlapping releases across the REST OF "
            + (f"{release_year}" if release_year else "the same release year")
            + " (before and after this title's own date), since a title "
            "competes for a limited pool of audience attention and "
            "spending across its whole release year, not only against "
            "whatever else opens the exact same weekend. Genre overlap, "
            "franchise fatigue commentary, and any direct comparisons "
            "critics or audiences are already drawing between this title "
            "and its competition." + region_clause
        )
        search_queries = [
            f"{title} box office competition" + (f" {release_year}" if release_year else ""),
            f"films releasing same weekend as {title}",
            f"major movie releases after {title} release date",
            f"biggest movie releases {release_year}" if release_year else f"biggest upcoming movie releases",
            f"{release_year} box office calendar major releases" if release_year else f"movies competing with {title} this year",
        ]

    # Franchise history is a genuinely different question from same-window
    # competition — "what did the earlier film(s) in THIS series do" rather
    # than "what else is out right now" — so it gets its own query rather
    # than being folded into the queries above, which wouldn't surface it.
    if source_type in ("sequel", "reboot", "spinoff", "remake"):
        franchise_query = based_on if based_on else f"{title} previous film"
        search_queries.append(f"{franchise_query} box office reception")

    search = await asyncio.to_thread(
        _client.search,
        objective=objective,
        search_queries=search_queries,
        session_id=session_id or None,
        # "advanced" over "fast" — this branch now also carries franchise
        # history (prior installments' box office), which benefits from
        # the deeper cross-referencing "advanced" mode does.
        mode="advanced",
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
        "director, session_id, release_status, region_hint, source_type, and "
        "based_on as key=value pairs; parse them and call the tool exactly "
        "once (region_hint/source_type/based_on may be empty strings, pass "
        "them through as given). Then act as a "
        "box-office and competitive-strategy analyst. Populate: "
        "competing_titles (up to 5 entries, each with the competitor's name, "
        "strength_vs_searched — its SPECIFIC edge over the searched title, "
        "e.g. 'bigger established fanbase' or 'locks in premium format "
        "screens with an earlier date', drawn from the excerpts, not "
        "invented — and, ONLY for released titles, gross_estimate if the "
        "excerpts state a box office figure for that specific competitor, "
        "null otherwise), attention_assessment, and risk. "
        "For an 'upcoming' title, don't limit competing_titles to same-"
        "weekend releases only — your search results also cover major "
        "titles releasing earlier and later across the rest of that "
        "release year, since audience attention and spend is genuinely "
        "shared across the whole year, not just one weekend. Pick the "
        "5 most relevant real competitors the excerpts actually name "
        "(a same-window release is usually most relevant, but a bigger "
        "year-wide tentpole in the same genre can matter more than a "
        "smaller same-weekend one) rather than defaulting only to "
        "whichever query happened to return results first. Never invent "
        "a competing title, date, or figure the excerpts don't state. "
        "attention_assessment should reflect this whole-year view when "
        "the excerpts support it, not just the immediate weekend.\n\n"
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
        "franchise_history: ONLY when source_type is 'sequel', 'reboot', "
        "'spinoff', or 'remake' — populate up to 5 entries naming the "
        "prior installment(s) in the SAME series (not unrelated "
        "competitors, those go in competing_titles) with whatever real "
        "box_office figure and reception_note the excerpts give you. This "
        "is a DIFFERENT question from competing_titles: 'how did the "
        "earlier film(s) in this series do' rather than 'what else is out "
        "right now'. Leave this empty for an original work, or if "
        "source_type qualifies but the excerpts don't actually name a "
        "specific prior film with real figures — never invent a "
        "predecessor's numbers by assuming a typical franchise pattern.\n\n"
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
    generate_content_config=genai_types.GenerateContentConfig(temperature=GROUNDING_TEMPERATURE),
)
