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
    cast: list[str] = Field(
        default_factory=list,
        description=(
            "The most prominent named cast members, typically up to 5. For a "
            "genuine ensemble/crossover film with no clear top-5 (e.g. a "
            "multi-hero team-up), list up to 10 of the most prominent named "
            "actors instead of leaving this empty — a longer real list is "
            "more useful downstream than an empty one, but every name must "
            "still come from the excerpts, never guessed."
        ),
    )
    source_type: Literal[
        "original", "remake", "book_adaptation", "comic_adaptation",
        "true_story", "sequel", "reboot", "spinoff", "unclear",
    ] = Field(
        default="unclear",
        description="What this work is derived from, if anything. 'original' = not based on a prior work. Only set to something other than 'unclear'/'original' if the excerpts explicitly state it.",
    )
    based_on: Optional[str] = Field(
        default=None,
        description="One short phrase naming the specific source when source_type isn't 'original', e.g. 'Remake of the 2016 Malayalam film Oppam' or 'Based on Frank Herbert's novel Dune Messiah' or 'Based on Marvel Comics'. Null if source_type is 'original'/'unclear' or the excerpts don't name a specific source.",
    )
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
    gross_estimate: Optional[str] = Field(
        default=None,
        description="This competitor's own box office gross, as reported (e.g. '$2.1B worldwide'), ONLY for released-title comparisons where the excerpts state a figure for THIS competitor specifically. Null if unknown — never estimate or infer one.",
    )


class BoxOfficeInfo(BaseModel):
    """Only meaningful for released titles — there's no box office for
    something that hasn't come out. Every field defaults to null/'unclear'
    so an upcoming title's CompetitiveResult can carry this field without
    it ever containing invented figures."""

    budget: Optional[str] = Field(default=None, description="Production budget as reported (e.g. '$250 million'). Null if not stated.")
    worldwide_gross: Optional[str] = Field(default=None, description="Worldwide box office gross as reported. Null if not stated.")
    domestic_gross: Optional[str] = Field(default=None, description="Domestic/home-market gross as reported. Null if not stated.")
    opening_weekend: Optional[str] = Field(default=None, description="Opening weekend gross as reported. Null if not stated.")
    verdict: Literal["blockbuster", "hit", "average", "underperformed", "flop", "unclear"] = Field(
        default="unclear",
        description="Overall commercial verdict, judged from budget-vs-gross and how it's actually characterized in coverage. 'unclear' if the excerpts don't support a read — do not guess from genre/studio reputation alone.",
    )
    verdict_basis: str = Field(
        default="",
        description="One sentence explaining the verdict (e.g. the budget-to-gross ratio, or how commentary characterized it). Empty string if verdict is 'unclear'.",
    )


class ReleaseWindowAdvice(BaseModel):
    """Grounded, directional read on release timing — never a fabricated
    counterfactual. This does NOT project what would happen on a
    different date (there is no search result for a hypothetical), it
    only assesses how congested the ACTUAL known window is, based on the
    real competing_titles already found, and gives a directional
    suggestion (earlier/later/keep) grounded in which real competitors
    that would avoid or encounter — never a specific alternate date, and
    never an invented performance number for one."""

    congestion: Literal["low", "moderate", "high", "unclear"] = Field(
        description="How crowded the release window actually is, based on the real competing_titles found — not a guess."
    )
    reasoning: str = Field(
        description="Why, citing the SPECIFIC competing titles and whatever release-timing/genre-overlap info the excerpts actually gave. Never invented."
    )
    suggested_direction: Literal["keep_current_window", "consider_earlier", "consider_later", "unclear"] = Field(
        description=(
            "A directional suggestion only, grounded in the real competitors found. "
            "NEVER name a specific alternate date, and NEVER state or imply a "
            "projected box-office outcome for a hypothetical date — there is no "
            "real data for a counterfactual, only for what actually exists. "
            "'unclear' if the excerpts don't give enough to support even a "
            "direction."
        )
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
    box_office: BoxOfficeInfo = Field(
        default_factory=BoxOfficeInfo,
        description="ONLY populate meaningfully for released titles — for upcoming titles, leave every field at its default (null/'unclear'), there is no box office yet.",
    )
    release_window: ReleaseWindowAdvice = Field(
        default_factory=lambda: ReleaseWindowAdvice(congestion="unclear", reasoning="", suggested_direction="unclear"),
        description="ONLY meaningful for upcoming titles — for released titles the window is already fixed, leave at defaults.",
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


class ReputationTimelineEntry(BaseModel):
    """One dated milestone in a title's public life — not restricted to a
    fixed set of phases, so it can actually cover the whole arc
    (announcement, casting reveals, each trailer, premiere, opening
    weekend, long-tail reaction...), not just whichever YouTube videos
    happened to get discovered. Only populate one when a real date
    actually places it — never infer a milestone from guesswork."""

    milestone: str = Field(
        description=(
            "A short, specific label for what this is — e.g. 'Casting "
            "Announcement', 'First Trailer', 'Special Look', 'Premiere', "
            "'Opening Weekend', 'Long-Tail Reaction'. Not restricted to a "
            "fixed list; name it accurately for what actually happened."
        )
    )
    date: str = Field(
        description=(
            "The real date this happened, normalized to YYYY-MM-DD "
            "(date only — never include a time-of-day, even if the "
            "source gave one). Convert whatever format the source used "
            "into this format; never invent or estimate a date."
        )
    )
    sentiment: Literal["positive", "mixed", "negative", "unclear"]
    note: str = Field(description="What was actually being said/reported around this milestone, one sentence, grounded in dated sources.")


class SentimentSynthesisResult(BaseModel):
    overall_sentiment: Literal["positive", "mixed", "negative", "unclear"]
    justification: str
    per_source_breakdown: dict[str, str] = Field(default_factory=dict)
    recurring_themes: list[str] = Field(default_factory=list)
    agreement_note: str = Field(description="Do sources agree or conflict, one sentence.")
    sources_available: list[str] = Field(description="Which of web/youtube/reddit actually had data this run.")
    timeline: list[ReputationTimelineEntry] = Field(
        default_factory=list, max_length=8,
        description="Up to 8 dated milestones across the title's whole public life, drawn from YouTube publish dates AND dated news facts (announcement, casting reveals, each trailer, premiere, opening weekend, long-tail reaction, etc.) — as many distinct real-dated events as the sources actually support, sorted chronologically. Leave empty entirely if no dates are available to place anything — this is a bonus signal, not a required field.",
    )


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


class WarRoomVoice(BaseModel):
    """One persona's one-line take. These are NOT separate agent calls —
    main_synthesis already has all five upstream agents' JSON, this just
    forces it to distill each domain into a specific, attributable voice
    instead of blending everything into one generic paragraph."""

    role: Literal["director", "producer", "marketing_chief", "casting_executive", "distribution_executive", "analyst"]
    insight: str = Field(
        description="One sharp, specific sentence in this persona's voice, grounded in that persona's upstream data (director→sentiment/themes, producer→competitive, marketing_chief→marketing, casting_executive→cast, distribution_executive→release_window, analyst→overall confidence read). Never generic filler like 'the film looks promising.'"
    )


class GreenlightMemo(BaseModel):
    """The single output of main_synthesis — one verdict-first memo,
    not two audience-split briefs. Every field must trace back to the
    five upstream agents' actual data; nothing here is a place to
    editorialize beyond what the evidence supports."""

    headline: str
    verdict: Literal["greenlight", "greenlight_with_changes", "hold", "pass"] = Field(
        description="For upcoming titles: a genuine greenlight-style call. For released/retrospective titles, this reflects how the release performed in hindsight (greenlight=clearly worked, greenlight_with_changes=worked with real caveats, hold=mixed/underwhelming, pass=clearly underperformed) — reframe the label's meaning by release_status, same pattern as competitive_risk."
    )
    confidence: int = Field(
        ge=0, le=100,
        description="0-100. Lower when upstream sources were sparse/unavailable or when signals conflict; do not default to a round number like 50 or 75 out of habit, base it on how much real evidence actually supports the verdict.",
    )
    confidence_rationale: str = Field(
        default="",
        description=(
            "One sentence stating WHY the confidence number is what it is — "
            "name how many of the five upstream sources actually had data "
            "(cross-check against sources_used) and whether they agreed or "
            "conflicted. E.g. 'Based on 4 of 5 sources, which broadly "
            "agreed; competitive data was unavailable this run.' This is "
            "shown directly next to the confidence score, so it must "
            "actually explain the number, not restate the verdict."
        ),
    )
    why: list[str] = Field(max_length=3, min_length=1, description="Up to 3 short reasons for the verdict, each traceable to specific upstream data (sentiment, competitive, cast, marketing, or news).")
    biggest_opportunity: str = Field(description="The single strongest positive lever, specific and named — not 'strong audience interest' but what specifically is driving it.")
    biggest_risk: str = Field(description="The single biggest threat, specific and named.")
    recommended_action: str = Field(description="ONE specific, forward-looking action. For released titles, reframe as the single biggest actionable lesson rather than a future action, since there's no release left to act on.")
    what_we_would_change: list[str] = Field(
        default_factory=list, max_length=3,
        description="ONLY for upcoming titles — up to 3 concrete changes to make before release. Leave EMPTY for released titles, there's nothing left to change.",
    )
    lessons_learned: list[str] = Field(
        default_factory=list, max_length=4,
        description="ONLY for released titles — concrete takeaways for future decisions. Leave EMPTY for upcoming titles, there's no outcome yet to learn from.",
    )
    war_room: list[WarRoomVoice] = Field(
        min_length=6, max_length=6,
        description="Exactly 6 voices, one per role, in this order: director, producer, marketing_chief, casting_executive, distribution_executive, analyst.",
    )
    notable_news: list[SourceExcerpt] = Field(default_factory=list, max_length=4)
    sources_used: list[str] = Field(default_factory=list, description="Which of the five upstream sections actually had data this run.")
