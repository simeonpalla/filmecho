"""Orchestration pipeline.

Sequencing:
    Stage 1 — resolve entity context from the bare title (blocking,
              everything else depends on it). Confidence-scored and
              release-status-aware; see agents/entity_context.py.
    Stage 2 — run six independent fetch/synthesis agents concurrently:
              Web Sentiment, YouTube (Discovery -> Data, internally
              sequential), Competitive, News & Cast, Cast reception,
              Marketing. Web Sentiment, Competitive, and Marketing all
              change their actual search question based on whether the
              title has released yet, see each agent's tool function.
    Stage 3 — Sentiment Synthesis Agent combines Web Sentiment + YouTube
              (+ Reddit, once built) into one structured sentiment read.
    Stage 4 — Main Synthesis Agent combines Sentiment Synthesis +
              Competitive + News + Cast + Marketing into the final
              GreenlightMemo (verdict-first, single studio view — no
              separate fan-facing brief; see agents/main_synthesis.py's
              docstring for why that split was removed).

stream_pipeline() is the source of truth: an async generator that yields
a progress event as each stage genuinely completes (Stage 2's branches
use asyncio.as_completed, so events fire in true completion order).
run_pipeline() is a thin wrapper for callers that just want the final dict.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncIterator, Optional, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# Must run before the agents/* imports below, since those modules read
# PARALLEL_API_KEY / YOUTUBE_API_KEY (and, in AI-Studio mode,
# GEMINI_API_KEY) from os.environ at import time.
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.cast_agent import cast_agent
from agents.competitive_agent import competitive_agent
from agents.entity_context import EntityContext, resolve_entity
from agents.main_synthesis import build_main_synthesis_prompt, main_synthesis_agent
from agents.marketing_agent import marketing_agent
from agents.news_cast_agent import news_cast_agent
from agents.schemas import (
    CastResult, CompetitiveResult, GreenlightMemo, MarketingResult,
    NewsResult, SentimentSynthesisResult, WebSentimentResult,
)
from agents.sentiment_synthesis import build_sentiment_prompt, sentiment_synthesis_agent
from agents.web_sentiment_agent import web_sentiment_agent
from agents.youtube_data_agent import collect_youtube_data
from agents.youtube_discovery_agent import discover_trailer_videos
from orchestration.memo_cache import memo_cache

APP_NAME = "filmecho"
_session_service = InMemorySessionService()

_ModelT = TypeVar("_ModelT", bound=BaseModel)

_FETCH_STAGE_LABELS = {
    "web_sentiment": "Web sentiment analysis complete",
    "youtube": "YouTube discovery and comments complete",
    "competitive": "Competitive landscape complete",
    "news": "Production/cast news complete",
    "cast": "Cast performance reception complete",
    "marketing": "Marketing analysis complete",
}


async def _run_adk_agent(
    agent, prompt: str, user_id: str, session_id: str, output_model: Type[_ModelT]
) -> tuple[Optional[_ModelT], list[str]]:
    """Run a single turn against an ADK agent, parse its structured output,
    and capture a human-readable log of the actual tool calls it made.

    Every agent this pipeline calls declares output_schema, so ADK
    guarantees the final response text is valid JSON matching that model.
    Returns (None, log) on any failure, so a broken agent call degrades one
    field of the final brief instead of crashing the whole pipeline run —
    the log is still returned even on failure, so the activity feed shows
    what was attempted before the failure.

    The activity log is built from ADK's own Event.get_function_calls() /
    get_function_responses() (confirmed against ADK's source, not guessed),
    not from anything this codebase invents — it's a direct, honest trace
    of what actually happened, not a simulated "looks busy" animation.
    """
    log: list[str] = []
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
            for call in event.get_function_calls():
                args_summary = ", ".join(f"{k}={v!r}" for k, v in list(call.args.items())[:2])
                log.append(f"calling {call.name}({args_summary}, ...)")
            for resp in event.get_function_responses():
                log.append(f"{resp.name} returned")
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text

        if not final_text:
            print(f"[pipeline] Agent '{agent.name}' returned no content.")
            log.append("no content returned")
            return None, log
        log.append("structured output parsed")
        return output_model.model_validate_json(final_text), log
    except ValidationError as exc:
        print(f"[pipeline] Agent '{agent.name}' output failed schema validation: {exc}")
        log.append("output failed schema validation")
        return None, log
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] Agent '{agent.name}' failed: {exc}")
        log.append(f"failed: {exc}")
        return None, log


def _entity_prompt(entity: EntityContext, region_hint: str = "") -> str:
    d = entity.as_dict()
    return (
        f"title={d['title']!r} release_year={d['release_year'] or ''!r} "
        f"director={d['director'] or ''!r} session_id={d['session_id']!r} "
        f"release_status={d['release_status']!r} cast={', '.join(d['cast'])!r} "
        f"region_hint={region_hint!r}"
    )


async def _run_web_sentiment_branch(entity: EntityContext, user_id: str, region_hint: str) -> tuple[Optional[WebSentimentResult], list[str]]:
    return await _run_adk_agent(
        web_sentiment_agent, _entity_prompt(entity, region_hint), user_id,
        session_id=f"ws_{uuid.uuid4().hex[:8]}", output_model=WebSentimentResult,
    )


async def _run_competitive_branch(entity: EntityContext, user_id: str, region_hint: str) -> tuple[Optional[CompetitiveResult], list[str]]:
    return await _run_adk_agent(
        competitive_agent, _entity_prompt(entity, region_hint), user_id,
        session_id=f"comp_{uuid.uuid4().hex[:8]}", output_model=CompetitiveResult,
    )


async def _run_news_branch(entity: EntityContext, user_id: str, region_hint: str) -> tuple[Optional[NewsResult], list[str]]:
    return await _run_adk_agent(
        news_cast_agent, _entity_prompt(entity, region_hint), user_id,
        session_id=f"news_{uuid.uuid4().hex[:8]}", output_model=NewsResult,
    )


async def _run_cast_branch(entity: EntityContext, user_id: str, region_hint: str) -> tuple[Optional[CastResult], list[str]]:
    return await _run_adk_agent(
        cast_agent, _entity_prompt(entity, region_hint), user_id,
        session_id=f"cast_{uuid.uuid4().hex[:8]}", output_model=CastResult,
    )


async def _run_marketing_branch(entity: EntityContext, user_id: str, region_hint: str) -> tuple[Optional[MarketingResult], list[str]]:
    return await _run_adk_agent(
        marketing_agent, _entity_prompt(entity, region_hint), user_id,
        session_id=f"mktg_{uuid.uuid4().hex[:8]}", output_model=MarketingResult,
    )


async def _run_youtube_branch(entity: EntityContext) -> tuple[dict, list[str]]:
    """Same (data, log) shape as the five ADK-agent branches, for a
    uniform activity feed even though this branch is plain function calls,
    not an LLM agent — the log here is just as honest, it's the real
    YouTube API calls made, not simulated activity."""
    log: list[str] = []
    d = entity.as_dict()
    log.append(f"calling youtube.search.list(q={d['title']!r})")
    videos = await asyncio.to_thread(
        discover_trailer_videos, d["title"], str(d["release_year"] or "")
    )
    video_ids = [v["video_id"] for v in videos]
    if not video_ids:
        print("[pipeline] YouTube discovery found no videos; degrading gracefully.")
        log.append("no trailer videos found")
        return {}, log
    log.append(f"found {len(video_ids)} candidate videos")
    log.append("calling youtube.videos.list + commentThreads.list")
    try:
        data = await asyncio.to_thread(collect_youtube_data, video_ids)
        log.append("YouTube data collected")
        return data, log
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] YouTube data branch failed: {exc}")
        log.append(f"failed: {exc}")
        return {}, log


async def _named(name: str, coro) -> tuple[str, object]:
    """Tag a branch coroutine's result with its name for as_completed().

    Every branch coroutine now returns (parsed_result_or_None, activity_log)
    — even on an unexpected raise here, this returns that same shape
    (None, [log line]) rather than a bare None, so every downstream
    unpacking site can rely on the tuple shape unconditionally.
    """
    try:
        result = await coro
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] branch '{name}' raised unexpectedly: {exc}")
        result = (None, [f"failed: {exc}"])
    return name, result


async def stream_pipeline(
    title: str, user_id: str = "filmecho_user", region_hint: str = "", force_refresh: bool = False,
) -> AsyncIterator[dict]:
    """Run the full pipeline, yielding progress events as each stage completes.

    Args:
        title: Bare title as typed by the user, e.g. "Toxic".
        user_id: ADK session user id; any stable string works for a demo.
        region_hint: Optional locale/timezone string from the requesting
            browser (e.g. "Asia/Kolkata, en-IN"), threaded into every
            fetch agent and main synthesis so the brief is framed for
            that market rather than defaulting to US/UK. Empty string if
            unavailable — every agent instruction treats that as "no
            regional bias," not an error.
        force_refresh: If True, skip a cache hit and recompute even for a
            released title, then overwrite the cache with the fresh
            result. This is the deliberate escape hatch for the one case
            a released title's memo SHOULD change: a re-release,
            anniversary rewatch spike, or other genuinely new event.
            Ignored for upcoming/unclear titles, which are never cached
            in the first place.

    Yields:
        dicts with an "event" key:
        - {"event": "stage", "stage": <name>, "message": <str>} — a stage
          genuinely finished. Order for the six Stage 2 branches reflects
          real completion order, not a fixed guess.
        - {"event": "activity", "stage": <name>, "log": [<str>, ...]} —
          the real tool calls that stage made (built from ADK's own
          Event.get_function_calls()/get_function_responses(), not
          simulated), emitted right after that stage's "stage" event.
        - {"event": "error", "message": <str>} — unrecoverable failure;
          no further events follow.
        - {"event": "result", "data": <dict>} — always the last event on
          success, same shape run_pipeline() returns. For a cache hit,
          `data["served_from_cache"]` is True and `data["cached_at"]`
          gives when the underlying agents actually ran.
    """
    yield {"event": "stage", "stage": "entity_resolution",
           "message": "Resolving title and checking release status..."}
    try:
        entity = await asyncio.to_thread(resolve_entity, title)
    except Exception as exc:  # noqa: BLE001
        yield {"event": "error", "message": f"Entity resolution failed: {exc}"}
        return

    status_note = f", release_status={entity.release_status}"
    if entity.confidence != "high":
        note = entity.disambiguation_note or "no note given"
        yield {"event": "stage", "stage": "entity_resolution",
               "message": f"Resolved with {entity.confidence} confidence{status_note} — {note}"}
    else:
        yield {"event": "stage", "stage": "entity_resolution",
               "message": f"Resolved: {entity.canonical_title or entity.title}{status_note}"}

    # A released title's facts don't change between queries — the
    # pipeline re-doing live research on every call was producing a
    # different confidence score and even different box-office figures
    # for the same film across repeat runs, which is wrong for something
    # framed as a settled studio memo. Upcoming/unclear titles are
    # excluded on purpose: pre-release buzz genuinely moves day to day.
    canonical_title = entity.canonical_title or entity.title
    if entity.release_status == "released" and not force_refresh:
        cached = memo_cache.get(canonical_title, entity.release_year)
        if cached is not None:
            yield {"event": "stage", "stage": "cache_hit",
                   "message": f"Using the frozen memo for this released title (originally computed {cached['cached_at']}). "
                              "Released titles aren't re-researched on every query — use \"Refresh this memo\" if something genuinely new happened (re-release, anniversary spike, new data)."}
            yield {"event": "result", "data": cached}
            return
    elif entity.release_status == "released" and force_refresh:
        memo_cache.invalidate(canonical_title, entity.release_year)
        yield {"event": "stage", "stage": "cache_refresh",
               "message": "Forcing a fresh recompute for this released title, overwriting the previous frozen memo."}

    yield {"event": "stage", "stage": "fetch",
           "message": "Running web sentiment, competitive, news, cast, marketing, and YouTube agents..."}

    branches = [
        _named("web_sentiment", _run_web_sentiment_branch(entity, user_id, region_hint)),
        _named("youtube", _run_youtube_branch(entity)),
        _named("competitive", _run_competitive_branch(entity, user_id, region_hint)),
        _named("news", _run_news_branch(entity, user_id, region_hint)),
        _named("cast", _run_cast_branch(entity, user_id, region_hint)),
        _named("marketing", _run_marketing_branch(entity, user_id, region_hint)),
    ]
    branch_results: dict[str, object] = {}
    for coro in asyncio.as_completed(branches):
        name, (parsed, log) = await coro
        branch_results[name] = parsed
        yield {"event": "stage", "stage": name, "message": _FETCH_STAGE_LABELS[name]}
        yield {"event": "activity", "stage": name, "log": log}

    web_sentiment: Optional[WebSentimentResult] = branch_results.get("web_sentiment")
    youtube_data: dict = branch_results.get("youtube") or {}
    competitive: Optional[CompetitiveResult] = branch_results.get("competitive")
    news: Optional[NewsResult] = branch_results.get("news")
    cast: Optional[CastResult] = branch_results.get("cast")
    marketing: Optional[MarketingResult] = branch_results.get("marketing")

    yield {"event": "stage", "stage": "sentiment_synthesis",
           "message": "Synthesizing sentiment across sources..."}
    sentiment_prompt = build_sentiment_prompt(web_sentiment, youtube_data, entity.release_date)
    sentiment_synthesis, sentiment_log = await _run_adk_agent(
        sentiment_synthesis_agent, sentiment_prompt, user_id,
        session_id=f"sent_{uuid.uuid4().hex[:8]}", output_model=SentimentSynthesisResult,
    )
    yield {"event": "stage", "stage": "sentiment_synthesis", "message": "Sentiment synthesis complete"}
    yield {"event": "activity", "stage": "sentiment_synthesis", "log": sentiment_log}

    yield {"event": "stage", "stage": "main_synthesis", "message": "Convening the war room and writing the memo..."}
    main_prompt = build_main_synthesis_prompt(entity, sentiment_synthesis, competitive, news, cast, marketing, region_hint)
    memo, main_log = await _run_adk_agent(
        main_synthesis_agent, main_prompt, user_id,
        session_id=f"main_{uuid.uuid4().hex[:8]}", output_model=GreenlightMemo,
    )
    yield {"event": "activity", "stage": "main_synthesis", "log": main_log}

    result: dict = memo.model_dump() if memo else {
        "verdict": None, "confidence": 0, "why": [], "war_room": [], "sources_used": [],
    }
    result["entity_confidence"] = entity.confidence
    result["disambiguation_note"] = entity.disambiguation_note
    result["release_status"] = entity.release_status

    payload = {
        "entity": entity.as_dict(),
        "web_sentiment": web_sentiment.model_dump() if web_sentiment else None,
        "youtube": youtube_data,
        "competitive": competitive.model_dump() if competitive else None,
        "news": news.model_dump() if news else None,
        "cast": cast.model_dump() if cast else None,
        "marketing": marketing.model_dump() if marketing else None,
        "sentiment_synthesis": sentiment_synthesis.model_dump() if sentiment_synthesis else None,
        "result": result,
    }

    # Freeze this result for a released title so the next query for the
    # same film serves it back instead of re-researching from scratch —
    # see memo_cache.py's docstring for the reasoning and the
    # force_refresh escape hatch.
    if entity.release_status == "released":
        memo_cache.set(canonical_title, entity.release_year, payload)

    yield {"event": "result", "data": payload}


async def run_pipeline(
    title: str, user_id: str = "filmecho_user", region_hint: str = "", force_refresh: bool = False,
) -> dict:
    """Non-streaming wrapper around stream_pipeline, for the CLI and tests.

    Returns:
        The dict from the stream's final "result" event.

    Raises:
        RuntimeError: if the stream emitted an "error" event, or ended
            without ever producing a result.
    """
    final_payload = None
    async for event in stream_pipeline(title, user_id, region_hint, force_refresh=force_refresh):
        if event["event"] == "result":
            final_payload = event["data"]
        elif event["event"] == "error":
            raise RuntimeError(event["message"])
    if final_payload is None:
        raise RuntimeError("Pipeline stream ended without producing a result")
    return final_payload


if __name__ == "__main__":
    output = asyncio.run(run_pipeline("Toxic"))
    print(json.dumps(output, indent=2, default=str))
