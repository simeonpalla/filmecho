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

from agents.entity_context import GEMINI_MODEL
from agents.schemas import SentimentSynthesisResult, WebSentimentResult

MAX_COMMENTS_PER_VIDEO = 15


def _flatten_youtube_comments(youtube_data: Optional[dict], max_per_video: int = MAX_COMMENTS_PER_VIDEO) -> str:
    """Turn the YouTube Data Agent's raw output into prompt-ready text."""
    if not youtube_data:
        return "NO YOUTUBE DATA AVAILABLE for this run."

    lines = []
    for video_id, info in youtube_data.items():
        lines.append(
            f"Video: {info.get('title', video_id)} "
            f"(views={info.get('view_count', 0)}, likes={info.get('like_count', 0)}, "
            f"comments={info.get('comment_count', 0)})"
        )
        for c in info.get("comments", [])[:max_per_video]:
            text = (c.get("text") or "").replace("\n", " ").strip()
            if text:
                lines.append(f"  - ({c.get('like_count', 0)} likes) {text[:280]}")
    return "\n".join(lines) if lines else "NO YOUTUBE COMMENTS AVAILABLE for this run."


def build_sentiment_prompt(
    web_sentiment: Optional[WebSentimentResult],
    youtube_data: Optional[dict],
    reddit_text: Optional[str] = None,
) -> str:
    """Build the full prompt for sentiment_synthesis_agent.

    Args:
        web_sentiment: web_sentiment_agent's structured result, or None if
            that branch failed or didn't parse.
        youtube_data: youtube_data_agent.collect_youtube_data's raw output,
            or None/{} if that branch failed or found no videos.
        reddit_text: Reserved for reddit_sentiment_agent's output once
            built; None until then.

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

    return (
        "=== WEB SENTIMENT (structured, from Parallel + Gemini) ===\n"
        f"{web_block}\n\n"
        "=== YOUTUBE COMMENTS (raw, unanalyzed) ===\n"
        f"{youtube_block}\n\n"
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
        "comments qualitatively instead."
    ),
    output_schema=SentimentSynthesisResult,
)
