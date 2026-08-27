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
Director: {director}
Release status: {release_status}

IMPORTANT: studio_brief.lessons_learned and fan_pulse.worth_watching must
stay EMPTY unless release_status is "released" — there is no lesson or
"worth watching" verdict to give for a title that hasn't come out yet.

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
    return _PROMPT_TEMPLATE.format(
        title=d["title"],
        release_year=d["release_year"] or "unknown",
        director=d["director"] or "unknown",
        release_status=d["release_status"],
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
        "in the user message, including the release_status rule about "
        "lessons_learned and worth_watching. Populate studio_brief and "
        "fan_pulse from that material only, and sources_used with exactly "
        "which of the five upstream sections actually had data."
    ),
    output_schema=FinalBrief,
)
