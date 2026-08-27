"""Shared entity disambiguation object for the Filmecho pipeline.

Every downstream agent needs the same canonical facts about the title in
question. This module resolves a bare title into those facts, and unlike
the first version of this file, it now does two things a naive
single-search-single-extract approach doesn't:

1. Forces Gemini to self-report a confidence level and, when the title is
   ambiguous (multiple real films/shows share it — "Toxic" is a real
   example that surfaced during testing), explicitly say which one it
   picked and why, instead of silently guessing.
2. If confidence isn't "high", runs one additional, more targeted
   Parallel search before finalizing, rather than shipping a low-
   confidence guess downstream unflagged.

Both the Parallel search and the Gemini extraction still degrade
gracefully to the bare title on failure, so a flaky call during a live
demo doesn't crash the pipeline, it just produces a low-confidence result
that the frontend can surface honestly.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from google import genai
from google.genai import types as genai_types
from parallel import Parallel

from agents.schemas import EntityResolution

# See entity_context.py's header note in git history for why this default
# is not "gemini-3-flash": that name 404s. Verify your own project's
# available models before trusting any default here, model availability
# has already changed once during this project's build.
GEMINI_MODEL = os.environ.get("FILMECHO_GEMINI_MODEL", "gemini-2.5-flash")


@dataclass
class EntityContext:
    """Canonical facts about a title, shared across all agents."""

    title: str
    canonical_title: Optional[str] = None
    release_year: Optional[int] = None
    director: Optional[str] = None
    cast: list[str] = field(default_factory=list)
    release_status: str = "unclear"
    confidence: str = "low"
    disambiguation_note: str = ""
    session_id: str = field(default_factory=lambda: f"filmecho_{uuid.uuid4().hex[:12]}")

    def as_dict(self) -> dict:
        return {
            "title": self.canonical_title or self.title,
            "release_year": self.release_year,
            "director": self.director,
            "cast": self.cast,
            "release_status": self.release_status,
            "confidence": self.confidence,
            "disambiguation_note": self.disambiguation_note,
            "session_id": self.session_id,
        }


_EXTRACTION_INSTRUCTION = (
    "Extract structured film/TV metadata from web search excerpts about a "
    "user-provided title. If the excerpts describe more than one distinct "
    "real work sharing this title, you must set confidence to 'low' or "
    "'medium' and use disambiguation_note to say which one you picked and "
    "why, don't silently resolve the ambiguity. Determine release_status by "
    "comparing the title's release date to today's date, given below. If "
    "unsure about a field, leave it null rather than guessing."
)


def _search_excerpts(parallel_client: Parallel, title: str, extra_hint: str = "") -> tuple[str, str]:
    """Run one Parallel search and return (excerpts_text, session_id).

    Args:
        extra_hint: Appended to the objective on the disambiguation retry,
            e.g. asking specifically to distinguish between same-titled works.
    """
    objective = (
        f"Identify the film or TV title '{title}': confirm its exact "
        "official title, release year, director, and top-billed cast."
    ) + (f" {extra_hint}" if extra_hint else "")
    search = parallel_client.search(
        objective=objective,
        search_queries=[f"{title} release year director", f"{title} cast"],
        mode="fast",
    )
    excerpts = "\n\n".join(
        excerpt[:500]
        for result in search.results[:5]
        for excerpt in result.excerpts[:1]
    )
    return excerpts, (search.session_id or "")


def _extract(genai_client: genai.Client, title: str, excerpts: str) -> Optional[EntityResolution]:
    if not excerpts:
        return None
    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=(
                f"Today's date: {date.today().isoformat()}\n"
                f'Title the user gave: "{title}"\n\n'
                f"Web search excerpts:\n{excerpts}"
            ),
            config=genai_types.GenerateContentConfig(
                system_instruction=_EXTRACTION_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=EntityResolution,
            ),
        )
        return EntityResolution.model_validate_json(response.text)
    except Exception as exc:  # noqa: BLE001
        print(f"[entity_context] Gemini extraction failed: {exc}")
        return None


def resolve_entity(
    title: str,
    parallel_client: Optional[Parallel] = None,
    genai_client: Optional[genai.Client] = None,
) -> EntityContext:
    """Resolve a bare title into a structured, confidence-scored EntityContext.

    Args:
        title: The bare title as the user typed it, e.g. "Toxic".
        parallel_client: Optional injected client (tests / shared pooling).
        genai_client: Optional injected client (tests / shared pooling).

    Returns:
        An EntityContext. On any failure it still carries `title` and a
        generated `session_id`, with confidence="low", so the pipeline
        continues in a visibly degraded state rather than halting or
        (worse) presenting a guess as if it were certain.
    """
    ctx = EntityContext(title=title)
    parallel_client = parallel_client or Parallel(api_key=os.environ["PARALLEL_API_KEY"])
    # No explicit api_key/project/location: reads GOOGLE_GENAI_USE_VERTEXAI /
    # GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION (Vertex + ADC) or
    # GEMINI_API_KEY (AI Studio) from the environment.
    genai_client = genai_client or genai.Client()

    try:
        excerpts, session_id = _search_excerpts(parallel_client, title)
        ctx.session_id = session_id or ctx.session_id
    except Exception as exc:  # noqa: BLE001
        print(f"[entity_context] Parallel search failed: {exc}")
        return ctx  # bare title only, confidence stays "low"

    result = _extract(genai_client, title, excerpts)
    if result is None:
        return ctx

    # Grounding retry: don't ship a low/medium-confidence guess unflagged
    # without at least trying once to resolve the ambiguity.
    if result.confidence != "high":
        try:
            retry_excerpts, _ = _search_excerpts(
                parallel_client,
                title,
                extra_hint=(
                    "There may be multiple distinct films or shows sharing this "
                    "title. Identify the most prominent, most currently newsworthy "
                    "one, and note any other candidates you considered."
                ),
            )
            combined = (excerpts + "\n\n" + retry_excerpts).strip()
            retried = _extract(genai_client, title, combined)
            if retried is not None:
                result = retried
        except Exception as exc:  # noqa: BLE001
            print(f"[entity_context] Disambiguation retry failed, keeping first result: {exc}")

    ctx.canonical_title = result.canonical_title or title
    ctx.release_year = result.release_year
    ctx.director = result.director
    ctx.cast = result.cast
    ctx.release_status = result.release_status
    ctx.confidence = result.confidence
    ctx.disambiguation_note = result.disambiguation_note
    return ctx
