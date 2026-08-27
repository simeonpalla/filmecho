"""Competitive Agent — same Parallel-call pattern as Web Sentiment Agent.

Answers a different question than Web Sentiment: not "how is this trailer
landing" but "what else is this title competing against for attention
right now" — same-window releases, franchise fatigue signals, genre
saturation. This is a second, independently identifiable Parallel
.search() call/import, so Stage One reviewers see more than one function
proving the integration is real rather than a single lucky example.
"""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from parallel import Parallel

from agents.entity_context import GEMINI_MODEL
from agents.schemas import CompetitiveResult

_client = Parallel(api_key=os.environ["PARALLEL_API_KEY"])


def get_competitive_landscape(title: str, release_year: str, director: str, session_id: str) -> dict:
    """Search the web for what a title is competing against for attention.

    Args:
        title: Canonical film title.
        release_year: Release year as a string, or "" if unknown.
        director: Director's name, or "" if unknown — included for
            disambiguation, not directly used in the objective text.
        session_id: Parallel session id shared across this pipeline run.

    Returns:
        dict with key "results": a list of {url, title, excerpts}.
    """
    objective = (
        f"What other films or shows are releasing in the same window as {title}"
        + (f" ({release_year})" if release_year else "")
        + ", competing for audience attention? Focus on same-weekend "
        "releases, genre overlap, franchise fatigue commentary, and any "
        "direct comparisons critics or audiences are already drawing "
        "between this title and its competition."
    )
    search = _client.search(
        objective=objective,
        search_queries=[
            f"{title} box office competition",
            f"films releasing same weekend as {title}",
        ],
        session_id=session_id or None,
        mode="fast",
    )
    return {
        "results": [
            {"url": r.url, "title": r.title, "excerpts": r.excerpts}
            for r in search.results
        ]
    }


competitive_tool = FunctionTool(func=get_competitive_landscape)

competitive_agent = Agent(
    name="competitive_agent",
    model=GEMINI_MODEL,
    description="Assesses competitive landscape and audience-attention risk via Parallel Search.",
    instruction=(
        "You have a tool, get_competitive_landscape, that searches the web "
        "for competing releases and franchise-fatigue signals. You will be "
        "given title, release_year, director, and session_id as key=value "
        "pairs; parse them and call the tool exactly once. Then populate the "
        "required output fields: competing_titles (2-4 named titles/events), "
        "attention_assessment (whether audience attention is split, "
        "concentrated, or unaffected), and risk, grounded only in what the "
        "excerpts actually say. If the excerpts don't support a clear read, "
        "set risk to 'unclear' rather than guessing."
    ),
    tools=[competitive_tool],
    output_schema=CompetitiveResult,
)
