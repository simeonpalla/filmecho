"""Main Synthesis — the Greenlight Memo.

This replaces the old two-audience (studio_brief/fan_pulse) split with a
single verdict-first memo. The change is deliberate: a dashboard of six
descriptive tabs is a research report, not a decision — a studio
executive wants "should we do this, why, what's the biggest risk" before
anything else, with evidence available on demand, not a wall of tabs to
read through first.

The "War Room" (six persona voices) is NOT six separate agent calls —
main_synthesis already receives all five upstream agents' full JSON, this
just forces it to distill each domain into one attributable, specific
line instead of blending everything into generic prose. Zero additional
API cost, meaningfully more "cinematic" presentation.

Fan Pulse is intentionally removed. This pipeline is now studio-decision
oriented; if a fan-facing product gets built later, it should be a
genuinely separate experience (different data needs — plot/theory
discussion, not sentiment — and a different interaction model), not
bolted back onto this schema.
"""

from __future__ import annotations

from typing import Optional

from google.adk.agents import Agent
from google.genai import types as genai_types

from agents.entity_context import EntityContext, GEMINI_MODEL, GROUNDING_TEMPERATURE
from agents.schemas import CastResult, CompetitiveResult, GreenlightMemo, MarketingResult, NewsResult, SentimentSynthesisResult

_PROMPT_TEMPLATE = """Produce the Greenlight Memo for a film title, from five structured
upstream agent outputs below. Base every field only on this material —
if a section says data was unavailable, reflect that gap honestly
(lower confidence, "unclear" verdict components) rather than inventing
a substitute.

Title: {title} ({release_year})
Release date: {release_date}
Director: {director}
Release status: {release_status}
Title-match confidence: {confidence}{confidence_note}
Region context: {region_hint}

IMPORTANT — verdict framing changes by release_status:
- 'upcoming': verdict is a genuine greenlight-style forward call.
  what_we_would_change may be populated (up to 3 items); lessons_learned
  MUST stay empty (there's no outcome yet to learn from).
- 'released': verdict reflects how the release performed IN HINDSIGHT.
  lessons_learned may be populated (up to 4 items); what_we_would_change
  MUST stay empty (nothing left to change about something that already
  happened).

IMPORTANT: if title-match confidence is not "high", reflect that
explicitly — fold it into the headline or the "why" list, and lower the
numeric confidence score accordingly. Don't present unwarranted
certainty about which specific title this data describes.

IMPORTANT — war_room voices: populate exactly 6, in this order, each
grounded in ITS OWN upstream section, one sharp sentence, not filler:
1. director — from SENTIMENT SYNTHESIS (audience/thematic reaction)
2. producer — from COMPETITIVE LANDSCAPE (competitive read)
3. marketing_chief — from MARKETING ANALYSIS (positioning/campaign read)
4. casting_executive — from CAST RECEPTION (per-actor read)
5. distribution_executive — from COMPETITIVE LANDSCAPE's release_window specifically (timing read; if release_window is unclear/not applicable, say so plainly rather than inventing a timing opinion)
6. analyst — your own overall confidence read, one sentence on what would most change your mind

IMPORTANT — war_room_transcript: the same 6 personas, but now write the
actual DISCUSSION those 5 real upstream sections would produce — a real
meeting, not 6 isolated lines. Target 12-18 turns (use the full range
when you have 4-5 real sources; a thinner run with fewer sources can run
shorter). Structure it like an actual back-and-forth, not a round-robin
of single statements:
- Each of the 5 domain personas opens with their own read (same
  role→source mapping as war_room above).
- Then have AT LEAST TWO separate exchanges where a second persona
  directly reacts to what an earlier one just said (set responding_to
  to that earlier role) — including at least one full push-and-reply
  pair: one turn disagreeing with an earlier point, AND a later turn
  from that original speaker (or a third persona) answering the
  pushback, not just a single one-sided objection left hanging.
- Set disagreement=true ONLY where two personas' real sources actually
  conflict this run — check CAST RECEPTION against PRODUCTION/CAST NEWS
  about the same person; check a positive SENTIMENT SYNTHESIS read
  against a crowded competitive release_window; check MARKETING's own
  read against what COMPETITIVE LANDSCAPE shows; check the release
  timing read against the audience-momentum read. Actively look for
  this kind of tension before defaulting to agreement — five
  independent sources researched separately very often DO pull in
  different directions even on a single metric (e.g. one section
  reads a number as a strength, another treats a related number as a
  risk) — surface that rather than smoothing it into consensus. If,
  after actually checking, nothing in the real data conflicts this run,
  don't invent tension — write the room genuinely converging instead,
  and it's fine for disagreement to be false throughout.
- The analyst speaks LAST, synthesizing the real discussion above
  (naming what the room did and didn't agree on) into the same
  verdict/confidence you're producing for the rest of this memo.
If fewer than 3 of the 5 upstream sections had real data, leave this
list empty (mirrors the insufficient_data threshold below) rather than
staging a discussion the room doesn't have material for.

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
    region_hint: str = "",
) -> str:
    """Build the full prompt for main_synthesis_agent.

    Args:
        entity: The resolved EntityContext for this pipeline run.
        sentiment: sentiment_synthesis_agent's structured result, or None.
        competitive: competitive_agent's structured result, or None.
        news: news_cast_agent's structured result, or None.
        cast: cast_agent's structured result, or None.
        marketing: marketing_agent's structured result, or None.
        region_hint: Optional locale/timezone string from the requesting
            browser, so the memo's framing reflects that market. Empty
            string if unavailable.

    Returns:
        A single prompt string built from validated Pydantic objects.
    """
    d = entity.as_dict()
    marketing_label = " (retrospective)" if d["release_status"] == "released" else " (in-progress)"
    confidence = d.get("confidence", "unclear")
    confidence_note = f" — {d['disambiguation_note']}" if confidence != "high" and d.get("disambiguation_note") else ""
    return _PROMPT_TEMPLATE.format(
        title=d["title"],
        release_year=d["release_year"] or "unknown",
        release_date=d.get("release_date") or "unknown",
        director=d["director"] or "unknown",
        release_status=d["release_status"],
        confidence=confidence,
        confidence_note=confidence_note,
        region_hint=region_hint or "not provided, no regional bias applied",
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
    description="Combines sentiment, competitive, news, cast, and marketing agents into one verdict-first Greenlight Memo.",
    instruction=(
        "Follow the instructions and structured data given to you exactly "
        "in the user message, including the release_status framing rules "
        "and the war_room ordering rules. "
        "GROUNDING: never write a specific number/percentage/statistic "
        "that isn't already present in the JSON you were given — you may "
        "repeat a number attributed to its source section, never invent "
        "or adjust one. Never state or imply a projected outcome for a "
        "hypothetical (e.g. a different release date) — only real, "
        "already-known data.\n\n"
        "CROSS-CHECK CAST RECEPTION against PRODUCTION/CAST NEWS "
        "specifically: both sections can independently look reasonable "
        "and still directly contradict each other about the same "
        "person's involvement (e.g. cast reception says an actor isn't "
        "returning, news facts say they are) — each agent only sees its "
        "own search results, so this kind of disagreement between them "
        "is a real, expected failure mode, not a hypothetical one. Before "
        "writing anything that states a specific person's involvement as "
        "settled, check whether the other section says something "
        "different about that same person. If they disagree, don't pick "
        "whichever framing sounds more confident — either omit that "
        "specific claim from your output entirely, or state plainly that "
        "reports conflict on it, and lower confidence accordingly (this "
        "is exactly the kind of conflicting signal confidence_rationale "
        "should mention — and, in war_room_transcript, exactly the kind "
        "of real tension a disagreement=true turn should be grounded "
        "in).\n\n"
        "Write like a strategy memo an executive would actually read: "
        "biggest_opportunity and biggest_risk must each be ONE specific, "
        "named thing, not a vague category. recommended_action must be a "
        "specific action, not a restatement of the verdict with 'given "
        "this...' bolted on front — if your action's first clause just "
        "repeats why, rewrite it to lead with the action itself. "
        "notable_news is strictly production/casting/release facts, not "
        "marketing-campaign facts (that's already captured upstream in "
        "MARKETING ANALYSIS, don't duplicate it here).\n\n"
        "confidence must actually vary based on how much real evidence "
        "you were given — five available, agreeing sources should score "
        "meaningfully higher than two available, conflicting ones. "
        "Populate sources_used with exactly which of the five upstream "
        "sections actually had data, and populate confidence_rationale "
        "with one sentence that a reader could use to verify the number — "
        "name how many of the five sources you had and whether they "
        "agreed. This is displayed directly next to the confidence score "
        "in the product, so a vague sentence here defeats the point of "
        "having the field at all.\n\n"
        "If fewer than 3 of the 5 upstream sections actually have data, "
        "set verdict to 'insufficient_data' instead of a normal "
        "greenlight/hold/pass call — a low confidence NUMBER next to a "
        "normal-looking verdict badge still reads as a real "
        "recommendation to someone skimming the memo. Note: this is "
        "checked and enforced in code after you respond, so it will be "
        "corrected either way — but get it right here so the rest of the "
        "memo (headline, why, recommended_action) is written consistently "
        "with there not being enough evidence for a real call, rather "
        "than writing a confident memo that then gets its verdict badge "
        "swapped out from under it."
    ),
    output_schema=GreenlightMemo,
    generate_content_config=genai_types.GenerateContentConfig(temperature=GROUNDING_TEMPERATURE),
)
