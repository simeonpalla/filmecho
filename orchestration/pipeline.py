"""Orchestration pipeline.

Sequencing:
    Stage 1 — resolve entity context from the bare title (blocking,
              everything else depends on it). Now confidence-scored: see
              agents/entity_context.py for the disambiguation retry.
    Stage 2 — run four independent fetch/synthesis agents concurrently:
              Web Sentiment, YouTube (Discovery -> Data, internally
              sequential), Competitive, News & Cast. The three LLM agents
              now return validated Pydantic objects (output_schema), not
              free text.
    Stage 3 — Sentiment Synthesis Agent combines Web Sentiment + YouTube
              (+ Reddit, once built) into one structured sentiment read.
    Stage 4 — Main Synthesis Agent combines Sentiment Synthesis +
              Competitive + News & Cast into the final FinalBrief object
              the frontend reads.

Each stage degrades gracefully: a failed, empty, or schema-invalid branch
becomes None, and every synthesis agent's instructions already account
for missing sources. A single flaky call during a live demo should
shrink the brief's confidence, not crash the whole pipeline.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# Must run before the agents/* imports below, since those modules read
# PARALLEL_API_KEY / YOUTUBE_API_KEY (and, in AI-Studio mode,
# GEMINI_API_KEY) from os.environ at import time.
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.competitive_agent import competitive_agent
from agents.entity_context import EntityContext, resolve_entity
from agents.main_synthesis import build_main_synthesis_prompt, main_synthesis_agent
from agents.news_cast_agent import news_cast_agent
from agents.schemas import CompetitiveResult, FinalBrief, NewsResult, SentimentSynthesisResult, WebSentimentResult
from agents.sentiment_synthesis import build_sentiment_prompt, sentiment_synthesis_agent
from agents.web_sentiment_agent import web_sentiment_agent
from agents.youtube_data_agent import collect_youtube_data
from agents.youtube_discovery_agent import discover_trailer_videos

APP_NAME = "filmecho"
_session_service = InMemorySessionService()

_ModelT = TypeVar("_ModelT", bound=BaseModel)


async def _run_adk_agent(
    agent, prompt: str, user_id: str, session_id: str, output_model: Type[_ModelT]
) -> Optional[_ModelT]:
    """Run a single turn against an ADK agent and parse its structured output.

    Every agent this pipeline calls now declares output_schema, so ADK
    guarantees the final response text is valid JSON matching that model,
    the try/except here is defensive (network failures, empty responses),
    not a workaround for schema drift, that class of bug is now prevented
    at the ADK layer rather than patched after the fact.

    Returns None on any failure, so a broken agent call degrades one field
    of the final brief instead of crashing the whole pipeline run.
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

        if not final_text:
            print(f"[pipeline] Agent '{agent.name}' returned no content.")
            return None
        return output_model.model_validate_json(final_text)
    except ValidationError as exc:
        print(f"[pipeline] Agent '{agent.name}' output failed schema validation: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] Agent '{agent.name}' failed: {exc}")
        return None


def _entity_prompt(entity: EntityContext) -> str:
    d = entity.as_dict()
    return (
        f"title={d['title']!r} release_year={d['release_year'] or ''!r} "
        f"director={d['director'] or ''!r} session_id={d['session_id']!r}"
    )


async def _run_web_sentiment_branch(entity: EntityContext, user_id: str) -> Optional[WebSentimentResult]:
    return await _run_adk_agent(
        web_sentiment_agent, _entity_prompt(entity), user_id,
        session_id=f"ws_{uuid.uuid4().hex[:8]}", output_model=WebSentimentResult,
    )


async def _run_competitive_branch(entity: EntityContext, user_id: str) -> Optional[CompetitiveResult]:
    return await _run_adk_agent(
        competitive_agent, _entity_prompt(entity), user_id,
        session_id=f"comp_{uuid.uuid4().hex[:8]}", output_model=CompetitiveResult,
    )


async def _run_news_branch(entity: EntityContext, user_id: str) -> Optional[NewsResult]:
    return await _run_adk_agent(
        news_cast_agent, _entity_prompt(entity), user_id,
        session_id=f"news_{uuid.uuid4().hex[:8]}", output_model=NewsResult,
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
    """Run the full pipeline for a title, through main synthesis.

    Args:
        title: Bare title as typed by the user, e.g. "Dune Part Three".
        user_id: ADK session user id; any stable string works for a demo.

    Returns:
        dict with keys: "entity" (includes confidence/disambiguation_note),
        "web_sentiment", "youtube", "competitive", "news",
        "sentiment_synthesis" (all as plain dicts, dumped from their
        validated Pydantic models, or None if that branch failed), and
        "result" (the parsed FinalBrief as a dict, with entity_confidence
        and disambiguation_note added deterministically in Python rather
        than trusted from the LLM's own relay of those fields).
    """
    entity = resolve_entity(title)
    if entity.confidence != "high":
        print(
            f"[pipeline] Entity resolution confidence={entity.confidence!r} for "
            f"{title!r}: {entity.disambiguation_note or '(no note)'}"
        )

    # Stage 2: four independent fetch/synthesis branches, concurrently.
    web_sentiment, youtube_data, competitive, news = await asyncio.gather(
        _run_web_sentiment_branch(entity, user_id),
        _run_youtube_branch(entity),
        _run_competitive_branch(entity, user_id),
        _run_news_branch(entity, user_id),
    )

    # Stage 3: combine sentiment feeds.
    sentiment_prompt = build_sentiment_prompt(web_sentiment, youtube_data)
    sentiment_synthesis = await _run_adk_agent(
        sentiment_synthesis_agent, sentiment_prompt, user_id,
        session_id=f"sent_{uuid.uuid4().hex[:8]}", output_model=SentimentSynthesisResult,
    )

    # Stage 4: combine everything into the final studio/fan brief.
    main_prompt = build_main_synthesis_prompt(entity, sentiment_synthesis, competitive, news)
    final_brief = await _run_adk_agent(
        main_synthesis_agent, main_prompt, user_id,
        session_id=f"main_{uuid.uuid4().hex[:8]}", output_model=FinalBrief,
    )

    result: dict = final_brief.model_dump() if final_brief else {
        "studio_brief": None, "fan_pulse": None, "sources_used": [],
    }
    # Entity confidence is Python-known ground truth, not something we ask
    # the LLM to faithfully copy through two prompt hops, so it's added
    # here rather than round-tripped through main_synthesis_agent's schema.
    result["entity_confidence"] = entity.confidence
    result["disambiguation_note"] = entity.disambiguation_note

    return {
        "entity": entity.as_dict(),
        "web_sentiment": web_sentiment.model_dump() if web_sentiment else None,
        "youtube": youtube_data,
        "competitive": competitive.model_dump() if competitive else None,
        "news": news.model_dump() if news else None,
        "sentiment_synthesis": sentiment_synthesis.model_dump() if sentiment_synthesis else None,
        "result": result,
    }


if __name__ == "__main__":
    output = asyncio.run(run_pipeline("Toxic"))
    print(json.dumps(output, indent=2, default=str))
