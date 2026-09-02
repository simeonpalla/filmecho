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


class TitleCandidate(BaseModel):
    """One possible match when a bare title is genuinely ambiguous."""

    title: str = Field(description="The candidate's full, distinguishing title.")
    year: Optional[int] = Field(default=None, description="Release year, or null if unknown.")
    note: str = Field(description="One short phrase distinguishing this candidate, e.g. 'the 2008 action film' or 'the 2026 Marvel tentpole where this is a subtitle'.")


class EntityResolution(BaseModel):
    """Structured output for entity_context.resolve_entity's Gemini call."""

    canonical_title: str = Field(description="The official, correctly disambiguated title.")
    release_year: Optional[int] = Field(default=None, description="Release year, or null if unknown.")
    release_date: Optional[str] = Field(
        default=None,
        description="Full release date as reported (e.g. 'September 11, 2026'), or null if only a year or nothing is known. Do not guess a date that wasn't in the excerpts.",
    )
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
    candidates: list[TitleCandidate] = Field(
        default_factory=list,
        description=(
            "Populate ONLY when confidence is not 'high' AND the excerpts genuinely "
            "support more than one distinct real work sharing this title — up to 3 "
            "candidates, most-likely-intended first. When populated, prefer listing "
            "the currently prominent/newsworthy candidate first if the excerpts "
            "suggest one is more likely what a person searching today means (e.g. a "
            "major upcoming tentpole vs. an obscure older film with the same name). "
            "Leave empty if the title was unambiguous."
        ),
    )


class SourceExcerpt(BaseModel):
    """One attributed claim, used across the fetch-agent schemas below."""

    url: str = Field(description="Source URL the claim came from.")
    claim: str = Field(description="The specific claim, in your own words, not a verbatim quote.")
    date: Optional[str] = Field(default=None, description="Date this was reported, if the excerpt states one (e.g. 'March 2025'). Null if unknown, don't guess.")


class WebSentimentResult(BaseModel):
    overall_sentiment: Literal["positive", "mixed", "negative", "unclear"]
    praise_points: list[SourceExcerpt] = Field(default_factory=list)
    criticism_points: list[SourceExcerpt] = Field(default_factory=list)
    tone_consensus: str = Field(description="One sentence on pacing/visual/tone consensus, or '' if none.")


class CompetitorInfo(BaseModel):
    """One competing title, with what edge it has over the searched title —
    not just a name, so a reader can see WHY it's competition, not just
    THAT it's competition."""

    title: str = Field(description="The competing title's name.")
    strength_vs_searched: str = Field(
        description="One short phrase on this competitor's specific edge over the searched title (e.g. 'bigger established fanbase', 'earlier release date locks in the premium format slots', 'higher pre-release tracking'). Must come from the excerpts, not be invented."
    )


class CompetitiveResult(BaseModel):
    competing_titles: list[CompetitorInfo] = Field(default_factory=list, max_length=5)
    attention_assessment: str = Field(description="1-2 sentences: is attention split, concentrated, or unaffected.")
    risk: Literal["low", "moderate", "high", "unclear"] = Field(
        description=(
            "For upcoming titles: forward-looking risk of audience attention being "
            "split by competing releases (high = seriously threatened by direct "
            "competition). For released titles: this is NOT a forward risk, it's a "
            "backward-looking read on competitive OUTCOME — high = notably "
            "outcompeted/underperformed against rivals, low = held its own or "
            "outperformed, moderate = mixed. Only use 'unclear' if the excerpts "
            "genuinely don't support even a rough read, not as a default when the "
            "question feels hard — for a released title with any box-office or "
            "reception data available, form a judgment from it."
        )
    )


class NewsItem(BaseModel):
    """One news item, categorized so the frontend can group related facts
    together instead of showing one long undifferentiated list."""

    url: str = Field(description="Source URL the claim came from.")
    claim: str = Field(description="The specific claim, in your own words, not a verbatim quote.")
    date: Optional[str] = Field(default=None, description="Date this was reported, if stated. Null if unknown.")
    category: Literal["casting", "production", "release", "box_office", "other"] = Field(
        description="'casting' = who's in it or cast changes. 'production' = filming, crew, budget, behind-the-scenes logistics. 'release' = dates, distribution, platform. 'box_office' = ticket sales, revenue figures. 'other' = doesn't fit the above."
    )


class NewsResult(BaseModel):
    facts: list[NewsItem] = Field(default_factory=list, max_length=8, description="Reported facts only, attributed and categorized.")
    rumors: list[NewsItem] = Field(default_factory=list, max_length=4, description="Explicitly labeled rumor/speculation, categorized.")


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


class CastPersonalNote(BaseModel):
    actor: str = Field(description="Actor's name.")
    headline: str = Field(description="Short description of the update, e.g. 'Reported wrist injury during reshoots.'")
    detail: str = Field(default="", description="Additional specifics if the excerpts give them.")
    source_url: str = Field(default="", description="Source URL, empty string if not attributable to one.")


class CastResult(BaseModel):
    performances: list[CastPerformanceNote] = Field(default_factory=list)
    standout_performance: str = Field(default="", description="Which actor got the most/strongest praise, if any.")
    overall_cast_reception: str = Field(default="", description="One sentence on the cast as a whole.")
    personal_updates: list[CastPersonalNote] = Field(
        default_factory=list,
        description="Behind-the-scenes or personal news about specific cast members — injuries during filming, remuneration/salary disputes, personal-life updates, controversies, on-set incidents. NOT performance reviews, that's the `performances` field. Leave empty if nothing like this surfaced, don't invent filler.",
    )


class MarketingResult(BaseModel):
    """Release-status-aware: for upcoming titles this covers the campaign
    so far; for released titles it's a genuine retrospective with a
    lessons_learned field, since there's an actual outcome to evaluate
    the campaign against.

    Every list here is capped and every item requires a source, on
    purpose: an earlier version of this schema let the agent produce an
    unbounded list of claims, and in testing it filled that space with
    precise-sounding statistics (e.g. "34% higher conversion") that were
    not actually in the source excerpts — invented numbers dressed up as
    data. Bounding length and requiring a URL per claim doesn't fully
    prevent fabrication on its own, but it removes the incentive to pad,
    and the agent instruction pairs with this by explicitly forbidding
    any number that doesn't trace to a specific excerpt.
    """

    strategies_observed: list[SourceExcerpt] = Field(
        default_factory=list,
        max_length=6,
        description="Up to 6 named marketing tactics/channels actually used, each attributed to a source URL. Quality over quantity — 3 well-attributed tactics beats 6 vague ones.",
    )
    what_worked: list[SourceExcerpt] = Field(
        default_factory=list,
        max_length=4,
        description="Up to 4 items. For released titles: tactics that demonstrably worked, judged against the actual outcome. For upcoming titles: do NOT use this for verdicts, there is no outcome yet — populate with genuinely positive early signals if the excerpts support one, otherwise leave empty. NEVER include a specific number/percentage/statistic unless that exact figure appears in the source excerpt — describe qualitatively instead of inventing a figure.",
    )
    what_underperformed: list[SourceExcerpt] = Field(
        default_factory=list,
        max_length=4,
        description="Up to 4 items. For released titles: tactics that demonstrably underperformed. For upcoming titles: do NOT use this for verdicts — populate with genuine early concerns if the excerpts support one, otherwise leave empty. NEVER include a specific number/percentage/statistic unless that exact figure appears in the source excerpt.",
    )
    lessons_learned: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Up to 4 items. Only meaningful for released titles with a known outcome. Leave empty for upcoming titles, don't speculate.",
    )


class StudioBrief(BaseModel):
    headline: str
    sentiment_summary: str
    competitive_risk: Literal["low", "moderate", "high", "unclear"]
    notable_news: list[SourceExcerpt] = Field(default_factory=list, max_length=4)
    recommendation: str = Field(
        default="",
        description="ONLY for upcoming titles: forward-looking, actionable business advice. For released/retrospective titles, leave this EMPTY — a forward recommendation for something that already happened and can't be changed (e.g. 'consider a re-release') is not useful advice, that's what lessons_learned is for instead.",
    )
    lessons_learned: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Up to 4 items. Only for released titles: concrete takeaways for future marketing/release decisions. Leave empty for upcoming titles.",
    )


class FanPulse(BaseModel):
    headline: str
    excitement_level: Literal["high", "mixed", "low", "unclear"]
    excitement_reason: str = Field(
        description="One or two sentences on WHY this excitement level was assigned — the specific thing driving it (a scene, a reunion, a controversy), not a restatement of the level itself."
    )
    standout_moment: str = Field(
        default="",
        description="One specific moment, scene, reveal, or beat fans keep mentioning by name — something concrete a browsing fan would want to see for themselves, not a generic descriptor like 'great action.' Empty string if nothing specific enough surfaced.",
    )
    hype_quote: str = Field(
        default="",
        description="A short, punchy paraphrase (NOT a verbatim quote) that captures the actual energy/vibe of how fans are talking about this — written like a pull-quote, not a summary sentence. Empty string if the available reactions don't support one.",
    )
    top_themes: list[str] = Field(default_factory=list, max_length=4)
    fun_fact_or_news: str = ""
    worth_watching: str = Field(
        default="",
        description="Only for released titles: a brief, honest verdict on whether it's worth watching now. Leave empty for upcoming titles.",
    )


class FinalBrief(BaseModel):
    studio_brief: StudioBrief
    fan_pulse: FanPulse
    sources_used: list[str] = Field(default_factory=list)
