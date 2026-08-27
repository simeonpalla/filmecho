"""Shared structured-output schemas for the Filmecho pipeline.

Every agent that used to return free text and get regex-parsed downstream
(main_synthesis being the one that actually drifted in practice) now
declares an ADK `output_schema`. ADK enforces this at the framework level
(Gemini's controlled generation plus a validating tool call), so a field
being missing or renamed is no longer a silent runtime surprise, it's a
schema violation ADK itself catches. See:
https://google.github.io/adk-docs/agents/llm-agents/#structured-data

This also means agent-to-agent context is a validated object, not prose
one agent has to re-read and hope it parsed correctly, that's the
"grounded contexting" half of this upgrade. The other half is
EntityResolution's confidence/disambiguation_note fields below, which
force the entity-resolution step to say explicitly when a title is
ambiguous instead of silently picking one.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class EntityResolution(BaseModel):
    """Structured output for entity_context.resolve_entity's Gemini call."""

    canonical_title: str = Field(description="The official, correctly disambiguated title.")
    release_year: Optional[int] = Field(default=None, description="Release year, or null if unknown.")
    director: Optional[str] = Field(default=None, description="Director's name, or null if unknown.")
    cast: list[str] = Field(default_factory=list, description="Up to 5 top-billed cast members.")
    release_status: Literal["released", "upcoming", "unclear"] = Field(
        description=(
            "'released' if the title has already come out relative to today's "
            "date (given in the prompt) — a re-release counts as released. "
            "'upcoming' if it has a confirmed or expected future release. "
            "'unclear' if the excerpts don't establish this."
        )
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "'high' only if excerpts clearly and consistently identify one title. "
            "'medium' if plausible but corroboration is thin or partly conflicting. "
            "'low' if the title is genuinely ambiguous (shared by multiple works) "
            "or excerpts are sparse/contradictory."
        )
    )
    disambiguation_note: str = Field(
        default="",
        description=(
            "If the bare title is ambiguous (multiple real works share it), state "
            "which one was selected and the specific evidence that justified it. "
            "Empty string if the title was unambiguous."
        ),
    )


class SourceExcerpt(BaseModel):
    """One attributed claim, used across the fetch-agent schemas below."""

    url: str = Field(description="Source URL the claim came from.")
    claim: str = Field(description="The specific claim, in your own words, not a verbatim quote.")


class WebSentimentResult(BaseModel):
    overall_sentiment: Literal["positive", "mixed", "negative", "unclear"]
    praise_points: list[SourceExcerpt] = Field(default_factory=list)
    criticism_points: list[SourceExcerpt] = Field(default_factory=list)
    tone_consensus: str = Field(description="One sentence on pacing/visual/tone consensus, or '' if none.")


class CompetitiveResult(BaseModel):
    competing_titles: list[str] = Field(default_factory=list)
    attention_assessment: str = Field(description="1-2 sentences: is attention split, concentrated, or unaffected.")
    risk: Literal["low", "moderate", "high", "unclear"]


class NewsResult(BaseModel):
    facts: list[SourceExcerpt] = Field(default_factory=list, description="Reported facts only, attributed.")
    rumors: list[SourceExcerpt] = Field(default_factory=list, description="Explicitly labeled rumor/speculation.")


class SentimentSynthesisResult(BaseModel):
    overall_sentiment: Literal["positive", "mixed", "negative", "unclear"]
    justification: str
    per_source_breakdown: dict[str, str] = Field(default_factory=dict)
    recurring_themes: list[str] = Field(default_factory=list)
    agreement_note: str = Field(description="Do sources agree or conflict, one sentence.")
    sources_available: list[str] = Field(description="Which of web/youtube/reddit actually had data this run.")


class CastPerformanceNote(BaseModel):
    actor: str = Field(description="Actor's name, matched to a role where the excerpts make that clear.")
    note: str = Field(description="Specific reception of this actor's performance, in your own words.")
    source_url: str = Field(default="", description="Source URL, empty string if not attributable to one.")


class CastResult(BaseModel):
    performances: list[CastPerformanceNote] = Field(default_factory=list)
    standout_performance: str = Field(default="", description="Which actor got the most/strongest praise, if any.")
    overall_cast_reception: str = Field(default="", description="One sentence on the cast as a whole.")


class MarketingResult(BaseModel):
    """Release-status-aware: for upcoming titles this covers the campaign
    so far; for released titles it's a genuine retrospective with a
    lessons_learned field, since there's an actual outcome to evaluate
    the campaign against."""

    strategies_observed: list[str] = Field(
        default_factory=list, description="Named marketing tactics/channels actually used (trailers, posters, partnerships, social pushes, screenings, etc.)."
    )
    what_worked: list[SourceExcerpt] = Field(default_factory=list)
    what_underperformed: list[SourceExcerpt] = Field(default_factory=list)
    lessons_learned: list[str] = Field(
        default_factory=list,
        description="Only meaningful for released titles with a known outcome. Leave empty for upcoming titles, don't speculate.",
    )


class StudioBrief(BaseModel):
    headline: str
    sentiment_summary: str
    competitive_risk: Literal["low", "moderate", "high", "unclear"]
    notable_news: list[str] = Field(default_factory=list)
    recommendation: str
    lessons_learned: list[str] = Field(
        default_factory=list,
        description="Only for released titles: concrete takeaways for future marketing/release decisions. Leave empty for upcoming titles.",
    )


class FanPulse(BaseModel):
    headline: str
    excitement_level: Literal["high", "mixed", "low", "unclear"]
    top_themes: list[str] = Field(default_factory=list)
    fun_fact_or_news: str = ""
    worth_watching: str = Field(
        default="",
        description="Only for released titles: a brief, honest verdict on whether it's worth watching now. Leave empty for upcoming titles.",
    )


class FinalBrief(BaseModel):
    studio_brief: StudioBrief
    fan_pulse: FanPulse
    sources_used: list[str] = Field(default_factory=list)
