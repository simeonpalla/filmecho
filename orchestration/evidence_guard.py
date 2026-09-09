"""Deterministic evidence-threshold guardrail for the Greenlight Memo.

Why this exists: main_synthesis_agent is INSTRUCTED to reflect low
evidence honestly (lower confidence, name what's missing), and it does —
a real production run with only 2 of 5 upstream sources available still
correctly dropped confidence to 35% and said so in confidence_rationale.
But the VERDICT BADGE itself doesn't know how much evidence backed it.
That run still produced a bold "GREENLIGHT" stamp sitting right next to
"35% confidence, 3 sources missing" — which is a worse failure than the
low number alone, because a verdict badge reads as a recommendation
regardless of the fine print next to it.

An instruction telling the model "don't do this" is not a guardrail — an
LLM can and did produce exactly the output the instruction said not to.
This module is the actual guardrail: a plain, deterministic Python
check, run after main_synthesis returns, that overrides the verdict when
there genuinely isn't enough evidence to support one — regardless of
what the model decided. It cannot be reasoned around by a prompt change
or a particularly confident-sounding model response.
"""

from __future__ import annotations

from typing import Optional

# Out of 5 possible upstream sources (sentiment, competitive, news, cast,
# marketing) — see main_synthesis.py's build_main_synthesis_prompt args.
# Below this many actually available, no verdict badge is shown at all.
MIN_SOURCES_FOR_VERDICT = 3

INSUFFICIENT_DATA_HEADLINE = "Insufficient data for a confident recommendation this run."


def count_available_sources(*sources: Optional[object]) -> int:
    """How many of the given upstream results are non-None (i.e. actually
    returned data, weren't dropped by an API failure, schema validation
    failure, or the refusal guard)."""
    return sum(1 for s in sources if s is not None)


def enforce_evidence_threshold(result: dict, available_count: int) -> dict:
    """Overwrite verdict/confidence/headline if evidence is too thin,
    regardless of what main_synthesis produced.

    Args:
        result: The dumped GreenlightMemo dict (or the degraded default
            dict used when main_synthesis itself failed/was dropped).
        available_count: How many of the 5 upstream sources had data —
            pass count_available_sources(sentiment_synthesis, competitive,
            news, cast, marketing).

    Returns:
        The same dict, mutated in place and also returned for
        convenience. Fields that reflect genuinely-available partial
        evidence (why, war_room, biggest_opportunity, etc.) are left
        alone — they can still be real and useful even when there isn't
        enough for an overall verdict. Only the fields that present
        themselves as a confident top-line recommendation are touched.
    """
    if available_count >= MIN_SOURCES_FOR_VERDICT:
        return result

    original_verdict = result.get("verdict")
    result["verdict"] = "insufficient_data"
    result["confidence"] = min(result.get("confidence") or 0, 25)
    result["headline"] = INSUFFICIENT_DATA_HEADLINE
    result["confidence_rationale"] = (
        f"Only {available_count} of 5 upstream sources returned data this run — below the "
        "minimum needed for a real recommendation. "
        + (
            f"The synthesis agent's own verdict ('{original_verdict}') was overridden by this "
            "check rather than shown, since a normal-looking verdict badge on this little "
            "evidence would misrepresent how much was actually analyzed."
            if original_verdict and original_verdict != "insufficient_data"
            else "Try again shortly, or use \"Refresh this memo\" once more sources are likely available."
        )
    )
    return result
