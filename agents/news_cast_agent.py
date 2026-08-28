"""News & Cast Agent — Parallel search, extractive-only.

Deliberately different instruction style from Web Sentiment and Competitive:
this agent is not asked for an opinion or a synthesized read, only for
attributed facts. "Extractive-only" per the roadmap means the model should
behave like a clipping service, not a critic — production news, cast
changes, and release-date changes are the kind of thing a studio brief
needs to be exactly right, not just plausible-sounding.
"""

from __future__ import annotations

import asyncio

import os

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from parallel import Parallel

from agents.entity_context import GEMINI_MODEL
from agents.schemas import NewsResult

_client = Parallel(api_key=os.environ["PARALLEL_API_KEY"])


async def get_production_news(title: str, release_year: str, director: str, session_id: str) -> dict:
    """Search the web for recent production/cast/release news about a title.

    Args:
        title: Canonical film title.
        release_year: Release year as a string, or "" if unknown.
        director: Director's name, or "" if unknown.
        session_id: Parallel session id shared across this pipeline run.

    Returns:
        dict with key "results": a list of {url, title, excerpts}.
    """
    objective = (
        f"What recent production news, cast changes, or release date "
        f"changes have been reported for {title}"
        + (f" ({release_year})" if release_year else "")
        + (f", directed by {director}" if director else "")
        + "? Only factual reporting — casting announcements, filming "
        "updates, distributor decisions, confirmed date changes."
    )
    search = await asyncio.to_thread(
        _client.search,
        objective=objective,
        search_queries=[f"{title} production news", f"{title} cast news"],
        session_id=session_id or None,
        mode="fast",
    )
    return {
        "results": [
            {"url": r.url, "title": r.title, "excerpts": r.excerpts}
            for r in search.results
        ]
    }


news_cast_tool = FunctionTool(func=get_production_news)

news_cast_agent = Agent(
    name="news_cast_agent",
    model=GEMINI_MODEL,
    description="Extracts attributed production/cast/release news via Parallel Search.",
    instruction=(
        "You have a tool, get_production_news, that searches the web for "
        "production and cast news. You will be given title, release_year, "
        "director, and session_id as key=value pairs; parse them and call "
        "the tool exactly once. Then populate facts with reported facts "
        "only (each attributed to a source URL, in your own words), and "
        "rumors with anything that's speculation rather than confirmed "
        "reporting, don't merge the two lists or omit the distinction. Do "
        "not add interpretation, predictions, or opinion about whether the "
        "news is good or bad for the film. Leave both lists empty if "
        "nothing relevant was found, don't invent filler."
    ),
    tools=[news_cast_tool],
    output_schema=NewsResult,
)
