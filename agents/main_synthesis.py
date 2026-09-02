"""Main Synthesis Agent — combines sentiment + competitive + news + cast + marketing.

This is the piece that turns five separate structured agent outputs into
the single object the frontend reads: "studio_brief" (for filmmakers/
studio crew) and "fan_pulse" (for fans).

Release-status-aware: the prompt tells the model explicitly whether this
title is released or upcoming, and instructs it to only populate
lessons_learned/worth_watching for released titles, there's no honest
lesson or "worth watching" verdict to give for something that hasn't
come out yet, and guessing one would be exactly the false-confidence
problem this pipeline tries to avoid elsewhere.

Two other things this instruction explicitly enforces, added after real
generated output showed both failing:
- studio_brief.recommendation must be genuinely actionable advice, not a
  restatement of sentiment_summary in different words.
- studio_brief.notable_news must stay strictly production/casting facts —
  it was absorbing marketing-campaign facts that then also showed up,
  redundantly, in the Marketing tab.
"""

from __future__ import annotations

from typing import Optional

from google.adk.agents import Agent

from agents.entity_context import EntityContext, GEMINI_MODEL
from agents.schemas import CastResult, CompetitiveResult, FinalBrief, MarketingResult, NewsResult, SentimentSynthesisResult

_PROMPT_TEMPLATE = """Produce the final combined brief for a film title, from five
structured upstream agent outputs below. Base every field only on this
material — if a section says data was unavailable, reflect that gap
honestly (e.g. competitive_risk: "unclear", empty notable_news) rather
than inventing a substitute.

Title: {title} ({release_year})
Release date: {release_date}
Director: {director}
Release status: {release_status}
Title-match confidence: {confidence}{confidence_note}

IMPORTANT: studio_brief.lessons_learned and fan_pulse.worth_watching must
stay EMPTY unless release_status is "released" — there is no lesson or
"worth watching" verdict to give for a title that hasn't come out yet.

IMPORTANT: if title-match confidence is not "high", both headlines must
make that uncertainty visible to the reader (e.g. "Note: match confidence
is medium — {confidence_note_short}" prepended to the headline or folded
into sentiment_summary/excitement context), don't present the brief with
unwarranted certainty about which specific title this data describes.

=== SENTIMENT SYNTHESIS ===
{sentiment_json}

=== COMPETITIVE LANDSCAPE ===
{competitive_json}

=== PRODUCTION / CAST NEWS ===
{news_json}

=== CAST RECEPTION (per-actor) ===
{cast_json}

=== MARKETING ANALYSIS{marketing_label} ===
{marketing_json}
"""


def build_main_synthesis_prompt(
    entity: EntityContext,
    sentiment: Optional[SentimentSynthesisResult],
    competitive: Optional[CompetitiveResult],
    news: Optional[NewsResult],
    cast: Optional[CastResult],
    marketing: Optional[MarketingResult],
) -> str:
    """Build the full prompt for main_synthesis_agent.

    Args:
        entity: The resolved EntityContext for this pipeline run (drives
            the release_status framing).
        sentiment: sentiment_synthesis_agent's structured result, or None.
        competitive: competitive_agent's structured result, or None.
        news: news_cast_agent's structured result, or None.
        cast: cast_agent's structured result, or None.
        marketing: marketing_agent's structured result, or None.

    Returns:
        A single prompt string built from validated Pydantic objects.
    """
    d = entity.as_dict()
    marketing_label = " (retrospective)" if d["release_status"] == "released" else " (in-progress)"
    confidence = d.get("confidence", "unclear")
    confidence_note = f" — {d['disambiguation_note']}" if confidence != "high" and d.get("disambiguation_note") else ""
    confidence_note_short = d.get("disambiguation_note") or "the title may be ambiguous"
    return _PROMPT_TEMPLATE.format(
        title=d["title"],
        release_year=d["release_year"] or "unknown",
        release_date=d.get("release_date") or "unknown",
        director=d["director"] or "unknown",
        release_status=d["release_status"],
        confidence=confidence,
        confidence_note=confidence_note,
        confidence_note_short=confidence_note_short,
        sentiment_json=sentiment.model_dump_json(indent=2) if sentiment else "UNAVAILABLE for this run.",
        competitive_json=competitive.model_dump_json(indent=2) if competitive else "UNAVAILABLE for this run.",
        news_json=news.model_dump_json(indent=2) if news else "UNAVAILABLE for this run.",
        cast_json=cast.model_dump_json(indent=2) if cast else "UNAVAILABLE for this run.",
        marketing_label=marketing_label,
        marketing_json=marketing.model_dump_json(indent=2) if marketing else "UNAVAILABLE for this run.",
    )


main_synthesis_agent = Agent(
    name="main_synthesis_agent",
    model=GEMINI_MODEL,
    description="Combines sentiment, competitive, news, cast, and marketing agents into the final studio/fan brief.",
    instruction=(
        "Follow the instructions and structured data given to you exactly "
        "in the user message, including the release_status rules. "
        "GROUNDING: never write a specific number/percentage/statistic "
        "that isn't already present in the JSON you were given below — if "
        "an upstream section contains a number, you may repeat it "
        "attributed to that section, but do not invent new ones or "
        "round/adjust existing ones to sound more precise. notable_news "
        "items need a source url (from the PRODUCTION / CAST NEWS section) "
        "just like the upstream data does.\n\n"
        "You are writing for two genuinely different readers, in two "
        "genuinely different voices:\n\n"
        "studio_brief is for a studio marketing/production executive making "
        "a decision. Write like a strategy memo, not a summary. "
        "sentiment_summary describes the reaction (what happened). "
        "recommendation must be forward-looking, ACTIONABLE advice — a "
        "specific move a studio could make (adjust the release window, "
        "lean into a specific angle in advertising, address a specific "
        "criticism before wide release) — not a restatement of "
        "sentiment_summary with 'given the positive reaction...' bolted on "
        "front. If your recommendation's first clause just repeats the "
        "sentiment finding, rewrite it to lead with the action instead. "
        "notable_news must be strictly production/casting/release facts "
        "(dates, budget, crew, cast changes) — do not put marketing-"
        "campaign facts here, that's the Marketing tab's job exclusively, "
        "a fact should appear in exactly one place, not be echoed across "
        "sections.\n\n"
        "fan_pulse is for an actual fan browsing entertainment content, "
        "not a business reader. Write with genuine energy and specificity, "
        "not corporate paraphrase — pull directly from sentiment "
        "synthesis's recurring_themes (which already favors real audience "
        "language over generic description) rather than re-summarizing "
        "into blander language. excitement_reason is required and must "
        "name the SPECIFIC thing driving the excitement level — a scene, a "
        "reunion, a controversy, a specific line people are quoting — "
        "never just restate the level itself ('excitement_level: high, "
        "excitement_reason: fans are very excited' is not acceptable, that "
        "tells the reader nothing they didn't already know from the pill). "
        "fun_fact_or_news should be something genuinely interesting to "
        "read, not a filler fact.\n\n"
        "Populate sources_used with exactly which of the five upstream "
        "sections actually had data."
    ),
    output_schema=FinalBrief,
)
