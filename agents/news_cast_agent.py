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
from google.genai import types as genai_types
from parallel import Parallel

from agents.entity_context import GEMINI_MODEL, GROUNDING_TEMPERATURE
from agents.schemas import NewsResult

_client = Parallel(api_key=os.environ["PARALLEL_API_KEY"])


async def get_production_news(
    title: str, release_year: str, director: str, session_id: str, region_hint: str = ""
) -> dict:
    """Search the web for recent production/cast/release news about a title.

    Args:
        title: Canonical film title.
        release_year: Release year as a string, or "" if unknown.
        director: Director's name, or "" if unknown.
        session_id: Parallel session id shared across this pipeline run.
        region_hint: Optional locale/timezone string from the requesting
            browser (e.g. "Asia/Kolkata, en-IN"), used to bias toward
            region-relevant release/distribution news. Empty string if
            unavailable.

    Returns:
        dict with key "results": a list of {url, title, excerpts}.
    """
    region_clause = (
        f" Include region-specific release or distribution news for "
        f"{region_hint} if available (local release date, distributor, "
        "certification), in addition to global production news."
        if region_hint else ""
    )
    objective = (
        f"What recent production news, cast changes, or release date "
        f"changes have been reported for {title}"
        + (f" ({release_year})" if release_year else "")
        + (f", directed by {director}" if director else "")
        + "? Only factual reporting — casting announcements, filming "
        "updates, distributor decisions, confirmed date changes." + region_clause
    )
    search = await asyncio.to_thread(
        _client.search,
        objective=objective,
        search_queries=[f"{title} production news", f"{title} cast news", f"{title} casting rumor confirmed denied"],
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
        "director, session_id, and region_hint as key=value pairs; parse "
        "them and call the tool exactly once (region_hint may be empty, "
        "pass it through as given). Then act as an entertainment beat "
        "reporter covering this production specifically — dates, budget, "
        "crew, filming locations, release timing, distribution deals, "
        "casting announcements. Populate facts with reported facts only "
        "(each attributed to a source URL, in your own words, categorized "
        "as casting/production/release/box_office/other), and rumors with "
        "anything that's speculation rather than confirmed reporting, "
        "don't merge the two lists or omit the distinction.\n\n"
        "CRITICAL — before filing ANYTHING as a fact, check your OWN "
        "search results for disagreement first: if two excerpts make "
        "different claims about the same specific thing (the same "
        "actor's involvement, the same date, the same budget figure), "
        "that is NOT a settled fact no matter how confidently either "
        "excerpt states it — file it under rumors instead, with a note "
        "that names the actual disagreement (e.g. 'Reports conflict on "
        "whether X returns for this specific film or only its sequel'). "
        "This matters most for casting/involvement claims specifically, "
        "since a studio's own plans can leak, change, or get reported "
        "prematurely — treat a single-outlet claim about someone's "
        "casting or involvement as provisional (note it as reported by "
        "that outlet, e.g. 'according to [source]', rather than flatly "
        "stating it) unless a second, independent excerpt corroborates "
        "it, and always check the excerpt's own language for hedging "
        "('reportedly', 'sources say', 'rumored') — a hedged source claim "
        "must not be flattened into a confident fact. A well-attributed, "
        "clearly-flagged rumor is a correct and useful output; a "
        "confidently-stated fact that turns out contested is not, even "
        "though it reads better.\n\n"
        "Your beat is "
        "PRODUCTION HISTORY, not promotional strategy (a marketing "
        "campaign's tactics belong to the marketing agent, not here, even "
        "when the same underlying event — like a trailer drop — could be "
        "framed either way; report the trailer's existence and date here, "
        "leave what the campaign was trying to accomplish to marketing) "
        "and not individual performance reception (that's the cast agent's "
        "job). Do not add interpretation, predictions, or opinion about "
        "whether the news is good or bad for the film. Leave both lists "
        "empty if nothing relevant was found, don't invent filler."
    ),
    tools=[news_cast_tool],
    output_schema=NewsResult,
    generate_content_config=genai_types.GenerateContentConfig(temperature=GROUNDING_TEMPERATURE),
)
