"""Web Sentiment Agent — the hackathon's Parallel track-requirement call.

This module is the one Stage One reviewers will check first: it imports
parallel-web directly and makes a real .search() call, not just a reference
in the README. Keep that import and call unambiguous; don't bury it inside
a generic "tools.py" utility file.

The raw search is wrapped as an ADK FunctionTool so a Gemini LlmAgent can
turn unstructured excerpts into a short, source-cited sentiment brief. That
combination (Parallel for data, Gemini/ADK for reasoning) is also what
satisfies the Google Cloud AI requirement for this agent.
"""

from __future__ import annotations

import asyncio

import os

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from parallel import Parallel

from agents.entity_context import GEMINI_MODEL
from agents.schemas import WebSentimentResult

_client = Parallel(api_key=os.environ["PARALLEL_API_KEY"])


async def get_web_sentiment(title: str, release_year: str, director: str, session_id: str, release_status: str) -> dict:
    """Search the live web for critic/audience reactions to a film.

    Args:
        title: Canonical film title.
        release_year: Release year as a string, or "" if unknown.
        director: Director's name, or "" if unknown.
        session_id: Parallel session id shared across this pipeline run, so
            Parallel groups related searches for better results.
        release_status: "released", "upcoming", or "unclear". Changes what
            question gets asked: an upcoming title has no post-release
            reception to search for, and a released title's trailer
            reactions are stale compared to actual reviews.

    Returns:
        dict with key "results": a list of {url, title, excerpts}, one entry
        per matching page Parallel found.
    """
    if release_status == "released":
        objective = (
            f"What did critics and audiences ultimately think of {title}"
            + (f" ({release_year})" if release_year else "")
            + (f", directed by {director}" if director else "")
            + "? Focus on overall critical reception, audience reaction after "
            "release, and how opinion held up or shifted over time (initial "
            "reviews vs. later reappraisal), not just trailer reactions."
        )
        search_queries = [f"{title} movie review", f"{title} critical reception"]
    else:
        objective = (
            f"What are critics and audiences saying about the trailer for {title}"
            + (f" ({release_year})" if release_year else "")
            + (f", directed by {director}" if director else "")
            + "? Focus on specific praise or criticism of pacing, cast "
            "performance, visuals, and tone."
        )
        search_queries = [f"{title} trailer reaction", f"{title} trailer review"]

    search = await asyncio.to_thread(
        _client.search,
        objective=objective,
        search_queries=search_queries,
        session_id=session_id or None,
        mode="fast",
    )
    return {
        "results": [
            {"url": r.url, "title": r.title, "excerpts": r.excerpts}
            for r in search.results
        ]
    }


web_sentiment_tool = FunctionTool(func=get_web_sentiment)

web_sentiment_agent = Agent(
    name="web_sentiment_agent",
    model=GEMINI_MODEL,
    description="Synthesizes web-wide trailer sentiment via Parallel Search.",
    instruction=(
        "You have a tool, get_web_sentiment, that searches the live web for "
        "reactions to a film. You will be given title, release_year, "
        "director, session_id, and release_status as key=value pairs in the "
        "user message; parse them and call the tool exactly once with those "
        "values. Then populate the required output fields from the tool's "
        "results: an overall sentiment label; praise_points and "
        "criticism_points as attributed claims with source URLs (in your "
        "own words, not verbatim quotes); and a one-sentence tone_consensus, "
        "or empty string if there isn't one. Only report opinions that are "
        "actually present in the search excerpts, never fill gaps with your "
        "own judgment."
    ),
    tools=[web_sentiment_tool],
    output_schema=WebSentimentResult,
)
