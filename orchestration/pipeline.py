"""Orchestration pipeline.

Sequencing:
    Stage 1 — resolve entity context from the bare title (blocking,
              everything else depends on it).
    Stage 2 — run four independent fetch/synthesis agents concurrently,
              since none of them depend on each other's output:
              Web Sentiment, YouTube (Discovery -> Data, internally
              sequential), Competitive, News & Cast.
    Stage 3 — Sentiment Synthesis Agent combines Web Sentiment + YouTube
              (+ Reddit, once built) into one sentiment read.
    Stage 4 — Main Synthesis Agent combines Sentiment Synthesis +
              Competitive + News & Cast into the final {studio_brief,
              fan_pulse} object the frontend reads.

Each stage degrades gracefully: a failed or empty branch is passed through
as None/{} rather than raising, and every synthesis agent's instructions
already account for missing sources. This matters more than it might
seem, since a single flaky call during a live demo should shrink the
brief's confidence, not crash the whole pipeline.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from dotenv import load_dotenv

# Must run before the agents/* imports below, since those modules read
# PARALLEL_API_KEY / YOUTUBE_API_KEY (and, in AI-Studio mode,
# GEMINI_API_KEY) from os.environ at import time. If this pipeline is ever
# imported (not run directly) by another entry point, that entry point
# needs its own load_dotenv() call too, this one only fires when
# pipeline.py itself is the first thing Python imports.
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.competitive_agent import competitive_agent
from agents.entity_context import EntityContext, resolve_entity
from agents.main_synthesis import build_main_synthesis_prompt, main_synthesis_agent, parse_main_synthesis_result
from agents.news_cast_agent import news_cast_agent
from agents.sentiment_synthesis import build_sentiment_prompt, sentiment_synthesis_agent
from agents.web_sentiment_agent import web_sentiment_agent
from agents.youtube_data_agent import collect_youtube_data
from agents.youtube_discovery_agent import discover_trailer_videos

APP_NAME = "filmecho"
_session_service = InMemorySessionService()


async def _run_adk_agent(agent, prompt: str, user_id: str, session_id: str) -> str | None:
    """Run a single turn against an ADK agent and return its final text.

    Returns None (rather than raising) on any failure, so a broken agent
    call degrades one field of the final brief instead of crashing the
    whole pipeline run.
    """
    try:
        await _session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        runner = Runner(agent=agent, app_name=APP_NAME, session_service=_session_service)
        content = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])

        final_text = None
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text
        return final_text
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] Agent '{agent.name}' failed: {exc}")
        return None


def _entity_prompt(entity: EntityContext) -> str:
    d = entity.as_dict()
    return (
        f"title={d['title']!r} release_year={d['release_year'] or ''!r} "
        f"director={d['director'] or ''!r} session_id={d['session_id']!r}"
    )


async def _run_web_sentiment_branch(entity: EntityContext, user_id: str) -> str | None:
    return await _run_adk_agent(
        web_sentiment_agent, _entity_prompt(entity), user_id, session_id=f"ws_{uuid.uuid4().hex[:8]}"
    )


async def _run_competitive_branch(entity: EntityContext, user_id: str) -> str | None:
    return await _run_adk_agent(
        competitive_agent, _entity_prompt(entity), user_id, session_id=f"comp_{uuid.uuid4().hex[:8]}"
    )


async def _run_news_branch(entity: EntityContext, user_id: str) -> str | None:
    return await _run_adk_agent(
        news_cast_agent, _entity_prompt(entity), user_id, session_id=f"news_{uuid.uuid4().hex[:8]}"
    )


async def _run_youtube_branch(entity: EntityContext) -> dict:
    d = entity.as_dict()
    videos = await asyncio.to_thread(
        discover_trailer_videos, d["title"], str(d["release_year"] or "")
    )
    video_ids = [v["video_id"] for v in videos]
    if not video_ids:
        print("[pipeline] YouTube discovery found no videos; degrading gracefully.")
        return {}
    try:
        return await asyncio.to_thread(collect_youtube_data, video_ids)
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] YouTube data branch failed: {exc}")
        return {}


async def run_pipeline(title: str, user_id: str = "filmecho_user") -> dict:
    """Run the full pipeline for a single title, through main synthesis.

    Args:
        title: Bare title as typed by the user, e.g. "Dune Part Three".
        user_id: ADK session user id; any stable string works for a demo.

    Returns:
        dict with keys: "entity", "web_sentiment", "youtube",
        "competitive", "news", "sentiment_synthesis", and "result" (the
        parsed {studio_brief, fan_pulse, sources_used} object, or a
        {"parse_error": ...} dict if the final JSON didn't parse).
    """
    entity = resolve_entity(title)

    # Stage 2: four independent fetch/synthesis branches, concurrently.
    web_sentiment_result, youtube_result, competitive_result, news_result = await asyncio.gather(
        _run_web_sentiment_branch(entity, user_id),
        _run_youtube_branch(entity),
        _run_competitive_branch(entity, user_id),
        _run_news_branch(entity, user_id),
    )

    # Stage 3: combine sentiment feeds.
    sentiment_prompt = build_sentiment_prompt(web_sentiment_result, youtube_result)
    sentiment_synthesis_result = await _run_adk_agent(
        sentiment_synthesis_agent, sentiment_prompt, user_id, session_id=f"sent_{uuid.uuid4().hex[:8]}"
    )

    # Stage 4: combine everything into the final studio/fan brief.
    main_prompt = build_main_synthesis_prompt(
        entity, sentiment_synthesis_result, competitive_result, news_result
    )
    main_synthesis_text = await _run_adk_agent(
        main_synthesis_agent, main_prompt, user_id, session_id=f"main_{uuid.uuid4().hex[:8]}"
    )
    result = (
        parse_main_synthesis_result(main_synthesis_text)
        if main_synthesis_text
        else {"studio_brief": None, "fan_pulse": None, "parse_error": "main_synthesis_agent returned no text"}
    )

    return {
        "entity": entity.as_dict(),
        "web_sentiment": web_sentiment_result,
        "youtube": youtube_result,
        "competitive": competitive_result,
        "news": news_result,
        "sentiment_synthesis": sentiment_synthesis_result,
        "result": result,
    }


if __name__ == "__main__":
    output = asyncio.run(run_pipeline("Dune Part Three"))
    print(json.dumps(output, indent=2, default=str))
