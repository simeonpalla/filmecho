"""Main Synthesis Agent — combines sentiment + competitive + news.

This is the piece that turns three separate structured agent outputs into
the single object the frontend reads: "studio_brief" (for filmmakers/
studio crew) and "fan_pulse" (for fans), per the two-audience-view
requirement.

Previous version of this file hand-parsed free text with regex and
drifted from its intended schema in testing (sources_used ended up
nested under fan_pulse instead of top-level). That's now structurally
prevented: output_schema=FinalBrief means ADK enforces the shape at the
framework level, not "hope the prompt words were persuasive enough."
"""

from __future__ import annotations

from typing import Optional

from google.adk.agents import Agent

from agents.entity_context import EntityContext, GEMINI_MODEL
from agents.schemas import CompetitiveResult, FinalBrief, NewsResult, SentimentSynthesisResult

_PROMPT_TEMPLATE = """Produce the final combined brief for a film title, from three
structured upstream agent outputs below. Base every field only on this
material — if a section says data was unavailable, reflect that gap
honestly (e.g. competitive_risk: "unclear", empty notable_news) rather
than inventing a substitute.

Title: {title} ({release_year})
Director: {director}

=== SENTIMENT SYNTHESIS ===
{sentiment_json}

=== COMPETITIVE LANDSCAPE ===
{competitive_json}

=== PRODUCTION / CAST NEWS ===
{news_json}
"""


def build_main_synthesis_prompt(
    entity: EntityContext,
    sentiment: Optional[SentimentSynthesisResult],
    competitive: Optional[CompetitiveResult],
    news: Optional[NewsResult],
) -> str:
    """Build the full prompt for main_synthesis_agent.

    Args:
        entity: The resolved EntityContext for this pipeline run.
        sentiment: sentiment_synthesis_agent's structured result, or None.
        competitive: competitive_agent's structured result, or None.
        news: news_cast_agent's structured result, or None.

    Returns:
        A single prompt string built from validated Pydantic objects
        (model_dump_json), not free text re-parsed from another agent's
        prose.
    """
    d = entity.as_dict()
    return _PROMPT_TEMPLATE.format(
        title=d["title"],
        release_year=d["release_year"] or "unknown",
        director=d["director"] or "unknown",
        sentiment_json=sentiment.model_dump_json(indent=2) if sentiment else "UNAVAILABLE for this run.",
        competitive_json=competitive.model_dump_json(indent=2) if competitive else "UNAVAILABLE for this run.",
        news_json=news.model_dump_json(indent=2) if news else "UNAVAILABLE for this run.",
    )


main_synthesis_agent = Agent(
    name="main_synthesis_agent",
    model=GEMINI_MODEL,
    description="Combines sentiment, competitive, and news agents into the final studio/fan brief.",
    instruction=(
        "Follow the instructions and structured data given to you exactly "
        "in the user message. Populate studio_brief and fan_pulse from that "
        "material only, and sources_used with exactly which of the three "
        "upstream sections actually had data."
    ),
    output_schema=FinalBrief,
)
