"""Detects an agent writing a meta-refusal into a plain-text field instead
of actually doing the research task it was asked to do.

Observed real failure: cast_agent, given an empty `cast` list for a large
ensemble film where entity resolution couldn't confidently pick a
top-5 (e.g. a multi-hero crossover with a dozen-plus co-leads), wrote
"I can't provide a cast reception for this film as I need to know the
cast members first" into `overall_cast_reception` instead of identifying
actors from its own search results. See agents/cast_agent.py's docstring
and instruction for the actual fix to that agent's behavior — this module
is the safety net in case it, or any other agent, ever does the same
thing again.

The reason this needs a separate check at all: `output_schema` (Pydantic)
enforces SHAPE, not content. An agent can write literally any string into
a plain text field and still pass schema validation cleanly — a refusal
sentence is a perfectly valid `str`. This module is a second, content-
level check layered on top of that shape check.
"""

from __future__ import annotations

import re
from typing import Optional, TypeVar

from pydantic import BaseModel

_ModelT = TypeVar("_ModelT", bound=BaseModel)

REFUSAL_PATTERN = re.compile(
    r"\bi (can't|cannot|don't have|need to know|would need|require)\b|"
    r"\bas an ai\b|\bi'm unable to\b|\bi am unable to\b",
    re.IGNORECASE,
)


def contains_refusal_text(value: object) -> bool:
    """Recursively scan a (possibly nested) dumped Pydantic model's values
    for refusal-shaped text. Takes the plain dict/list/str tree from
    `model.model_dump()`, not a model instance, so it has no dependency
    on any particular schema."""
    if isinstance(value, str):
        return bool(REFUSAL_PATTERN.search(value))
    if isinstance(value, dict):
        return any(contains_refusal_text(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_refusal_text(v) for v in value)
    return False


def drop_if_refusal(name: str, model: Optional[_ModelT]) -> Optional[_ModelT]:
    """Degrade a branch to None if its output contains refusal-shaped
    text — the same degradation a schema-validation failure or an API
    error already produces for that branch, since a meta-refusal is just
    as unusable as no output at all, and letting it reach the final memo
    would look far worse than the branch simply being marked unavailable.
    """
    if model is None:
        return None
    if contains_refusal_text(model.model_dump()):
        print(f"[pipeline] '{name}' branch contained refusal-shaped text, discarding it: {model!r}")
        return None
    return model
