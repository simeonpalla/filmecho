"""Main Synthesis Agent — combines sentiment + competitive + news.

This is the piece that turns three separate agent outputs into the single
object the frontend actually reads: one JSON payload with a "studio_brief"
view (for filmmakers/studio crew) and a "fan_pulse" view (for fans), per
the two-audience-view requirement the hackathon rules call for.

Like sentiment_synthesis_agent, this agent makes no external calls of its
own, it only reasons over text the other three agents already produced.
It's asked to return structured JSON (same extract-and-parse pattern as
entity_context.py) rather than free text, since the frontend needs to
render two distinct views from one call, not eyeball a single paragraph.
"""

from __future__ import annotations

import json

from google.adk.agents import Agent

from agents.entity_context import EntityContext, GEMINI_MODEL

_JSON_PROMPT_TEMPLATE = """You are producing the final combined brief for a film title, from three
upstream agent outputs. Return ONLY a JSON object, no markdown fences, no
commentary, with exactly this shape:
{{
  "studio_brief": {{
    "headline": string,               // one-sentence takeaway for a studio exec
    "sentiment_summary": string,      // 2-3 sentences, business-toned
    "competitive_risk": string,       // "low" | "moderate" | "high" | "unclear"
    "notable_news": [string],         // up to 4 bullet facts, empty array if none
    "recommendation": string          // one sentence, actionable
  }},
  "fan_pulse": {{
    "headline": string,               // one-sentence takeaway for a fan audience
    "excitement_level": string,       // "high" | "mixed" | "low" | "unclear"
    "top_themes": [string],           // up to 4 short phrases fans are talking about
    "fun_fact_or_news": string        // one interesting, fan-relevant fact, or "" if none
  }},
  "sources_used": [string]            // e.g. ["web", "youtube", "competitive", "news"] — only ones with real data
}}

Base every field only on the material below. If a section says data was
unavailable, do not invent a substitute, reflect that gap honestly (e.g.
competitive_risk: "unclear", empty notable_news array).

Title: {title} ({release_year})
Director: {director}

=== SENTIMENT SYNTHESIS ===
{sentiment_text}

=== COMPETITIVE LANDSCAPE ===
{competitive_text}

=== PRODUCTION / CAST NEWS ===
{news_text}
"""


def build_main_synthesis_prompt(
    entity: EntityContext,
    sentiment_text: str | None,
    competitive_text: str | None,
    news_text: str | None,
) -> str:
    """Build the full prompt for main_synthesis_agent.

    Args:
        entity: The resolved EntityContext for this pipeline run.
        sentiment_text: sentiment_synthesis_agent's final text, or None.
        competitive_text: competitive_agent's final text, or None.
        news_text: news_cast_agent's final text, or None.

    Returns:
        A single prompt string ready to send to main_synthesis_agent.
    """
    d = entity.as_dict()
    return _JSON_PROMPT_TEMPLATE.format(
        title=d["title"],
        release_year=d["release_year"] or "unknown",
        director=d["director"] or "unknown",
        sentiment_text=sentiment_text or "UNAVAILABLE for this run.",
        competitive_text=competitive_text or "UNAVAILABLE for this run.",
        news_text=news_text or "UNAVAILABLE for this run.",
    )


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text.removeprefix("json").strip()
    return text


def parse_main_synthesis_result(raw_text: str) -> dict:
    """Parse main_synthesis_agent's output text into the final result dict.

    Args:
        raw_text: The agent's raw final response text.

    Returns:
        The parsed dict on success. On a parse failure, returns
        {"studio_brief": None, "fan_pulse": None, "raw": raw_text,
        "parse_error": str(exc)} so the pipeline can still return
        something inspectable instead of crashing the whole run over a
        formatting slip in one LLM call.
    """
    try:
        return json.loads(_strip_json_fences(raw_text))
    except Exception as exc:  # noqa: BLE001
        return {
            "studio_brief": None,
            "fan_pulse": None,
            "raw": raw_text,
            "parse_error": str(exc),
        }


main_synthesis_agent = Agent(
    name="main_synthesis_agent",
    model=GEMINI_MODEL,
    description="Combines sentiment, competitive, and news agents into the final studio/fan brief.",
    instruction=(
        "Follow the JSON schema and instructions given to you exactly in "
        "the user message. Return only the JSON object, nothing else."
    ),
)
