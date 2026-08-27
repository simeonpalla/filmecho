"""Shared entity disambiguation object for the Filmecho pipeline.

Every downstream agent (Web Sentiment, YouTube pair, Competitive, News/Cast)
needs the same canonical facts about the title in question. Rather than
re-resolving the title in each agent, this module does it once:

1. Parallel Search grounds the bare title with real web facts (this also
   counts toward the track's "imported and called" Parallel requirement,
   though the primary satisfier of that rule is web_sentiment_agent.py).
2. Gemini extracts those facts into a structured EntityContext.

Design note: this degrades gracefully. If Parallel or Gemini fail, the
EntityContext still carries the bare title, and downstream agents receive
empty strings for release_year/director rather than crashing. Build the
degrade-path in from the start, since flaky title resolution during a
live demo is a more common failure mode than a bad API key.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from parallel import Parallel

# IMPORTANT: Gemini model names change fast and this default can go stale.
# If you get a 404, run this against your own key to see what's actually
# available, and set FILMECHO_GEMINI_MODEL in .env accordingly:
#   (Invoke-RestMethod "https://generativelanguage.googleapis.com/v1beta/models?key=$env:GEMINI_API_KEY").models
#     | Where-Object { $_.supportedGenerationMethods -contains "generateContent" } | Select-Object name
GEMINI_MODEL = os.environ.get("FILMECHO_GEMINI_MODEL", "gemini-3.6-flash")


@dataclass
class EntityContext:
    """Canonical facts about a title, shared across all agents."""

    title: str
    canonical_title: Optional[str] = None
    release_year: Optional[int] = None
    director: Optional[str] = None
    cast: list[str] = field(default_factory=list)
    # Reused across Parallel calls in the same pipeline run so Parallel can
    # group related searches and improve result quality on later calls.
    session_id: str = field(default_factory=lambda: f"filmecho_{uuid.uuid4().hex[:12]}")

    def as_dict(self) -> dict:
        return {
            "title": self.canonical_title or self.title,
            "release_year": self.release_year,
            "director": self.director,
            "cast": self.cast,
            "session_id": self.session_id,
        }


_EXTRACTION_PROMPT = """You are extracting structured film/TV metadata from web search excerpts.
Return ONLY a JSON object, no markdown fences, no commentary, with exactly these keys:
{{
  "canonical_title": string,
  "release_year": integer or null,
  "director": string or null,
  "cast": array of up to 5 strings (top-billed cast, empty array if unknown)
}}

If the excerpts are ambiguous or don't clearly identify one title, set fields
you're unsure of to null rather than guessing.

Title the user gave: "{title}"

Web search excerpts:
{excerpts}
"""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text.removeprefix("json").strip()
    return text


def resolve_entity(
    title: str,
    parallel_client: Optional[Parallel] = None,
    genai_client: Optional[genai.Client] = None,
) -> EntityContext:
    """Resolve a bare title into a structured EntityContext.

    Args:
        title: The bare title as the user typed it, e.g. "Dune Part Three".
        parallel_client: Optional injected client (tests / shared pooling).
        genai_client: Optional injected client (tests / shared pooling).

    Returns:
        An EntityContext. If resolution fails at any stage, the returned
        context still has `title` set and `session_id` generated, so the
        pipeline can continue in degraded mode rather than halting.
    """
    ctx = EntityContext(title=title)
    parallel_client = parallel_client or Parallel(api_key=os.environ["PARALLEL_API_KEY"])
    genai_client = genai_client or genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    excerpts = ""
    try:
        search = parallel_client.search(
            objective=(
                f"Identify the film or TV title '{title}': confirm its exact "
                "official title, release year, director, and top-billed cast."
            ),
            search_queries=[f"{title} release year director", f"{title} cast"],
            mode="fast",
        )
        ctx.session_id = search.session_id or ctx.session_id
        excerpts = "\n\n".join(
            excerpt[:500]
            for result in search.results[:5]
            for excerpt in result.excerpts[:1]
        )
    except Exception as exc:  # noqa: BLE001 - degrade, don't crash the pipeline
        print(f"[entity_context] Parallel search failed, degrading to bare title: {exc}")

    if not excerpts:
        return ctx

    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_EXTRACTION_PROMPT.format(title=title, excerpts=excerpts),
        )
        parsed = json.loads(_strip_json_fences(response.text))
        ctx.canonical_title = parsed.get("canonical_title") or title
        ctx.release_year = parsed.get("release_year")
        ctx.director = parsed.get("director")
        ctx.cast = parsed.get("cast") or []
    except Exception as exc:  # noqa: BLE001
        print(f"[entity_context] Gemini extraction failed, using bare title: {exc}")

    return ctx
