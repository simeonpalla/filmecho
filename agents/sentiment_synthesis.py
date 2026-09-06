"""Sentiment Synthesis Agent — combines the sentiment feeds into one read.

Takes the now-structured WebSentimentResult (a Pydantic object, not free
text) and the raw YouTube comments, and produces a schema-enforced
SentimentSynthesisResult. No external calls of its own; it only reasons
over what web_sentiment_agent and youtube_data_agent already fetched.

Built to degrade per-source: each input can be missing, and the schema's
sources_available field forces the model to say plainly which sources it
actually had, rather than blending in a source that wasn't there.
"""

from __future__ import annotations

from typing import Optional

from google.adk.agents import Agent
from google.genai import types as genai_types

from agents.entity_context import GEMINI_MODEL, GROUNDING_TEMPERATURE
from agents.schemas import NewsResult, SentimentSynthesisResult, WebSentimentResult

MAX_COMMENTS_PER_VIDEO = 15


def _flatten_youtube_comments(youtube_data: Optional[dict], max_per_video: int = MAX_COMMENTS_PER_VIDEO) -> str:
    """Turn the YouTube Data Agent's raw output into prompt-ready text.

    Includes each video's real publish date (published_at, from
    YouTube's own API, not inferred) so the synthesis agent has an actual
    date to anchor timeline phases against — not a guess.
    """
    if not youtube_data:
        return "NO YOUTUBE DATA AVAILABLE for this run."

    lines = []
    for video_id, info in youtube_data.items():
        published = info.get("published_at") or "date unknown"
        lines.append(
            f"Video: {info.get('title', video_id)} (published: {published}, "
            f"views={info.get('view_count', 0)}, likes={info.get('like_count', 0)}, "
            f"comments={info.get('comment_count', 0)})"
        )
        for c in info.get("comments", [])[:max_per_video]:
            text = (c.get("text") or "").replace("\n", " ").strip()
            if text:
                lines.append(f"  - ({c.get('like_count', 0)} likes) {text[:280]}")
    return "\n".join(lines) if lines else "NO YOUTUBE COMMENTS AVAILABLE for this run."


def _flatten_dated_news(news: Optional[NewsResult]) -> str:
    """Surface news_cast_agent's dated facts (production start, casting
    reveal, release-date confirmation, etc.) as timeline-building
    material. This is the source of milestones a YouTube-only timeline
    could never cover — an announcement or casting reveal usually
    doesn't have its own dedicated video, but the news agent already
    captured its date via NewsItem.date."""
    if not news:
        return "NO PRODUCTION/CAST NEWS DATA AVAILABLE for this run."
    facts = (news.facts if hasattr(news, "facts") else news.get("facts")) or []
    dated = [f for f in facts if (f.date if hasattr(f, "date") else f.get("date"))]
    if not dated:
        return "No dated facts among the production/cast news for this run."
    lines = []
    for f in dated:
        date = f.date if hasattr(f, "date") else f.get("date")
        claim = f.claim if hasattr(f, "claim") else f.get("claim")
        category = f.category if hasattr(f, "category") else f.get("category")
        lines.append(f"- ({date}, {category}) {claim}")
    return "\n".join(lines)


def build_sentiment_prompt(
    web_sentiment: Optional[WebSentimentResult],
    youtube_data: Optional[dict],
    release_date: Optional[str] = None,
    reddit_text: Optional[str] = None,
    news: Optional[NewsResult] = None,
) -> str:
    """Build the full prompt for sentiment_synthesis_agent.

    Args:
        web_sentiment: web_sentiment_agent's structured result, or None if
            that branch failed or didn't parse.
        youtube_data: youtube_data_agent.collect_youtube_data's raw output,
            or None/{} if that branch failed or found no videos.
        release_date: The resolved entity's release_date (if known), given
            for context only — timeline milestones are no longer
            classified against it into fixed phases, see the agent's
            instruction for why.
        reddit_text: Reserved for reddit_sentiment_agent's output once
            built; None until then.
        news: news_cast_agent's structured result, or None. Its dated
            facts (NewsItem.date) are the source of timeline milestones
            that aren't tied to a YouTube video — announcement, casting
            reveals, production wrap, etc.

    Returns:
        A single prompt string, with the web sentiment section serialized
        from the validated Pydantic object (model_dump_json), not raw
        prose, so this agent is reading structured data, not re-parsing
        another model's free text.
    """
    web_block = (
        web_sentiment.model_dump_json(indent=2)
        if web_sentiment
        else "NO WEB SENTIMENT DATA AVAILABLE for this run."
    )
    youtube_block = _flatten_youtube_comments(youtube_data)
    reddit_block = reddit_text or "NO REDDIT DATA AVAILABLE (not yet built for this pipeline)."
    news_dates_block = _flatten_dated_news(news)
    release_date_block = release_date or "unknown"

    return (
        f"Reference release date (for context only): {release_date_block}\n\n"
        "=== WEB SENTIMENT (structured, from Parallel + Gemini) ===\n"
        f"{web_block}\n\n"
        "=== YOUTUBE COMMENTS (raw, unanalyzed, with real publish dates) ===\n"
        f"{youtube_block}\n\n"
        "=== DATED PRODUCTION/CAST NEWS (for timeline milestones only) ===\n"
        f"{news_dates_block}\n\n"
        "=== REDDIT SENTIMENT ===\n"
        f"{reddit_block}"
    )


sentiment_synthesis_agent = Agent(
    name="sentiment_synthesis_agent",
    model=GEMINI_MODEL,
    description="Combines web, YouTube, and Reddit sentiment feeds into one read.",
    instruction=(
        "You will receive up to three labeled sections: WEB SENTIMENT, "
        "YOUTUBE COMMENTS, and REDDIT SENTIMENT. Any section may say data "
        "is unavailable — if so, exclude that source from sources_available "
        "and from your synthesis instead of guessing what it might have "
        "said.\n\n"
        "Act as an audience-insights analyst, not a summarizer. Using only "
        "the sources that ARE available, populate: overall_sentiment; "
        "justification (one sentence); per_source_breakdown (a short entry "
        "per available source); recurring_themes (2-4 items — make these "
        "SPECIFIC and vivid, grounded in the actual language people used, "
        "not generic marketing phrases; 'fans are calling the trailer "
        "chant \"chills-inducing\"' is a real theme, 'positive reception of "
        "the trailer' is not, that's just restating overall_sentiment); "
        "agreement_note (do sources agree or conflict); and "
        "sources_available (exactly which sources had real data this run). "
        "When YOUTUBE COMMENTS is available, prefer pulling recurring_themes "
        "from what commenters actually said (paraphrased, not quoted "
        "verbatim) over generic paraphrase of WEB SENTIMENT — real audience "
        "voice reads as more genuine than critic-summary language. Do not "
        "fabricate sentiment for a source with no data, and do not average "
        "YouTube like counts into a fake numeric score, describe the "
        "comments qualitatively instead.\n\n"
        "timeline (OPTIONAL, up to 8 entries): build this from TWO date "
        "sources — each YouTube video's published_at, AND the dated "
        "entries in DATED PRODUCTION/CAST NEWS above. Don't force these "
        "into a fixed 5-phase scheme; instead, write a short, specific "
        "milestone label for what actually happened at each date — "
        "'Casting Announcement', 'Production Wrap', 'First Trailer', "
        "'Special Look', 'Premiere', 'Opening Weekend', 'Long-Tail "
        "Reaction', or whatever accurately describes it. Cover the "
        "title's whole arc where the dates support it — an announcement "
        "and a casting reveal are just as valid a milestone as a trailer "
        "video, don't only report on YouTube-anchored moments if the news "
        "section gives you earlier ones too. Every entry's date MUST be "
        "normalized to YYYY-MM-DD (date only, strip any time-of-day, e.g. "
        "a YouTube '2026-07-20T13:00:20Z' becomes '2026-07-20') — never "
        "invent or estimate one. Write a one-sentence note per milestone "
        "based on what was actually said/reported around that date. Sort "
        "entries chronologically by date. If no dates are available at "
        "all across both sources, leave timeline as an empty list — do "
        "not guess a milestone without a real date behind it."
    ),
    output_schema=SentimentSynthesisResult,
    generate_content_config=genai_types.GenerateContentConfig(temperature=GROUNDING_TEMPERATURE),
)
