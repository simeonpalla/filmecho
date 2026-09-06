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

# Every agent in this pipeline shares this temperature to cut down on
# run-to-run variance for the SAME underlying data — e.g. the same
# Avengers: Doomsday query producing noticeably different confidence
# scores or verdicts within the same hour was traced partly to default
# sampling temperature, not just to the web genuinely changing that fast.
# Low, not zero: some structured-output tasks degrade at temperature=0
# (repetition, refusal to pick between close-call verdicts), and the
# Gemini API doesn't guarantee bit-exact reproducibility at any
# temperature — this reduces variance, it does not eliminate it. The
# remaining variance for "upcoming" titles is real and expected: Parallel
# search is live, and view counts/trailer drops/new coverage genuinely
# change between two runs an hour apart. That part isn't a bug to fix
# here, see orchestration/memo_cache.py for how "released" titles (whose
# facts truly are frozen) are handled instead.
GROUNDING_TEMPERATURE = float(os.environ.get("FILMECHO_TEMPERATURE", "0.1"))


@dataclass
class EntityContext:
    """Canonical facts about a title, shared across all agents."""

    title: str
    canonical_title: Optional[str] = None
    release_year: Optional[int] = None
    release_date: Optional[str] = None
    director: Optional[str] = None
    cast: list[str] = field(default_factory=list)
    release_status: str = "unclear"
    source_type: str = "unclear"
    based_on: Optional[str] = None
    confidence: str = "low"
    disambiguation_note: str = ""
    candidates: list[dict] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: f"filmecho_{uuid.uuid4().hex[:12]}")

    def as_dict(self) -> dict:
        return {
            "title": self.canonical_title or self.title,
            "release_year": self.release_year,
            "release_date": self.release_date,
            "director": self.director,
            "cast": self.cast,
            "release_status": self.release_status,
            "source_type": self.source_type,
            "based_on": self.based_on,
            "confidence": self.confidence,
            "disambiguation_note": self.disambiguation_note,
            "candidates": self.candidates,
            "session_id": self.session_id,
        }


_EXTRACTION_INSTRUCTION = (
    "Extract structured film/TV metadata from web search excerpts about a "
    "user-provided title. If the excerpts describe more than one distinct "
    "real work sharing this title, you must set confidence to 'low' or "
    "'medium', populate candidates with each plausible option, and use "
    "disambiguation_note to explain the ambiguity, don't silently resolve "
    "it by picking one. When ambiguous, weight recency and current "
    "prominence: if one candidate is a major upcoming release currently "
    "in the news and the other is an obscure or much older work, list the "
    "prominent one first, a person searching a bare title today is more "
    "often thinking of what's currently newsworthy than an old cult film "
    "that happens to share the name, though the excerpts' actual content "
    "should still be what decides this, not an assumption.\n\n"
    "RELEASE DATES CHANGE: an announced release date for an upcoming film "
    "is frequently pushed back or moved up after its initial announcement. "
    "If excerpts show more than one date for the same title, trust the "
    "one from the most recently published source, don't average or pick "
    "arbitrarily, and if you can't tell which is more recent, lower your "
    "confidence rather than presenting an uncertain date as settled. "
    "Determine release_status by comparing your best release_date estimate "
    "(or release_year if no exact date was found) against today's date, "
    "given below — these two fields must agree with each other. Only fill "
    "release_date if the excerpts state one, don't infer a specific date "
    "from just a year.\n\n"
    "PROVENANCE: also determine source_type and based_on — is this a "
    "remake, a book/comic adaptation, based on a true story, a "
    "sequel/reboot/spinoff, or an original work? Only claim something "
    "other than 'original'/'unclear' if the excerpts explicitly state it, "
    "don't infer from genre conventions or guess. based_on should name "
    "the specific source (the original film's title and year, the book's "
    "title and author, the comic series) when the excerpts give you that "
    "level of detail, not just a category label.\n\n"
    "If unsure about any field, leave it null rather than guessing."
)


def _search_excerpts(parallel_client: Parallel, title: str, extra_hint: str = "") -> tuple[str, str]:
    """Run one Parallel search and return (excerpts_text, session_id).

    Args:
        extra_hint: Appended to the objective on the disambiguation retry,
            e.g. asking specifically to distinguish between same-titled works.
    """
    objective = (
        f"Identify the film or TV title '{title}': confirm its exact "
        "official title, release year, director, top-billed cast, current "
        "release date, and what it's based on (original story, remake of "
        "an earlier film, adapted from a book/comic/true story, or a "
        "sequel/reboot/spinoff). Release dates for announced films "
        "frequently get pushed back or moved up after initial "
        "announcement — prioritize the most recently published source "
        "for the release date specifically, and if sources disagree on "
        "the date, note that explicitly rather than picking one silently."
    ) + (f" {extra_hint}" if extra_hint else "")
    search = parallel_client.search(
        objective=objective,
        search_queries=[
            f"{title} release year director",
            f"{title} cast",
            f"{title} release date",
            f"{title} based on remake adaptation",
        ],
        mode="fast",
    )
    excerpts = "\n\n".join(
        excerpt[:500]
        for result in search.results[:6]
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
                temperature=GROUNDING_TEMPERATURE,
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
    ctx.release_date = result.release_date
    ctx.director = result.director
    ctx.cast = result.cast
    ctx.release_status = result.release_status
    ctx.source_type = result.source_type
    ctx.based_on = result.based_on
    ctx.confidence = result.confidence
    ctx.disambiguation_note = result.disambiguation_note
    ctx.candidates = [c.model_dump() for c in result.candidates]
    return ctx
