"""Sentiment Synthesis Agent — combines the sentiment feeds into one read.

Unlike web_sentiment_agent, competitive_agent, and news_cast_agent, this
agent makes no external calls of its own. It only reasons over data the
other agents already fetched: Web Sentiment Agent's synthesized text, and
YouTube Data Agent's raw (unanalyzed) comments. A Reddit feed slots in the
same way once reddit_sentiment_agent.py exists — this agent is written to
degrade gracefully when a feed is missing, per the roadmap's explicit
guidance to design it that way from the start, since YouTube or Reddit
being unavailable during a live demo is a real risk, not a hypothetical.

Because it has no tool, it's driven entirely by the prompt text the
orchestration layer builds with build_sentiment_prompt() below — this
module owns both the agent and its own prompt construction so the two
stay in sync.
"""

from __future__ import annotations

from google.adk.agents import Agent

from agents.entity_context import GEMINI_MODEL

MAX_COMMENTS_PER_VIDEO = 15


def _flatten_youtube_comments(youtube_data: dict | None, max_per_video: int = MAX_COMMENTS_PER_VIDEO) -> str:
    """Turn the YouTube Data Agent's raw output into prompt-ready text.

    Args:
        youtube_data: Output of youtube_data_agent.collect_youtube_data,
            or None/{} if that branch failed or found nothing.
        max_per_video: Cap on comments included per video, to keep the
            prompt a reasonable size when a video has thousands.

    Returns:
        A plain-text block, or an explicit "no data" marker string if
        youtube_data is empty — the agent's instruction is written to
        handle that marker rather than silently inventing YouTube signal.
    """
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
    web_sentiment_text: str | None,
    youtube_data: dict | None,
    reddit_text: str | None = None,
) -> str:
    """Build the full prompt for sentiment_synthesis_agent.

    Args:
        web_sentiment_text: web_sentiment_agent's final synthesized text,
            or None if that branch failed.
        youtube_data: youtube_data_agent.collect_youtube_data's raw output,
            or None/{} if that branch failed or found no videos.
        reddit_text: Reserved for reddit_sentiment_agent's output once
            built; None until then, and the instruction already accounts
            for its absence.

    Returns:
        A single prompt string ready to send to sentiment_synthesis_agent.
    """
    web_block = web_sentiment_text or "NO WEB SENTIMENT DATA AVAILABLE for this run."
    youtube_block = _flatten_youtube_comments(youtube_data)
    reddit_block = reddit_text or "NO REDDIT DATA AVAILABLE (not yet built for this pipeline)."

    return (
        "=== WEB SENTIMENT (Parallel + Gemini synthesis) ===\n"
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
        "is unavailable — if so, exclude that source from your synthesis "
        "instead of guessing what it might have said, and note which "
        "sources were actually available at the end of your answer.\n\n"
        "Using only the sources that ARE available, produce:\n"
        "1. An overall sentiment label (positive, mixed, or negative) "
        "with one sentence justifying it.\n"
        "2. A per-source breakdown: what each available source suggests, "
        "in 1-2 sentences each.\n"
        "3. 2-4 recurring themes mentioned across sources (e.g. praise for "
        "visuals, complaints about pacing), each attributed to which "
        "source(s) raised it.\n"
        "4. A one-line note on whether sources agree or conflict with "
        "each other.\n\n"
        "Do not fabricate sentiment for a source that had no data, and do "
        "not average YouTube like counts into a fake numeric sentiment "
        "score, describe the comments qualitatively instead."
    ),
)
