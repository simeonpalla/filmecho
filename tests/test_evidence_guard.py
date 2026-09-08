"""Tests for orchestration/evidence_guard.py.

Direct regression test for the observed production failure: a real run
with only 2 of 5 upstream sources available still produced a bold
"GREENLIGHT" verdict badge — main_synthesis correctly lowered confidence
to 35% and explained the gap in confidence_rationale, but nothing
stopped the verdict itself from looking like a normal recommendation.
This is the code-level check that makes it structurally impossible for
that to happen again, regardless of what the model outputs.
"""

import pytest

from orchestration.evidence_guard import (
    MIN_SOURCES_FOR_VERDICT,
    count_available_sources,
    enforce_evidence_threshold,
)


class TestCountAvailableSources:
    def test_counts_only_non_none(self):
        assert count_available_sources("a", None, "b", None, None) == 2

    def test_all_present(self):
        assert count_available_sources(1, 2, 3, 4, 5) == 5

    def test_all_none(self):
        assert count_available_sources(None, None, None, None, None) == 0


class TestEnforceEvidenceThreshold:
    def test_below_threshold_overrides_confident_verdict(self):
        """Direct regression test for the actual observed failure: 2 of 5
        sources, but the model still said 'greenlight'."""
        result = {
            "verdict": "greenlight",
            "confidence": 35,
            "headline": "Greenlight Memo: 'Avengers Doomsday' (Low Confidence Title Match)",
            "confidence_rationale": "Based on 2 of 5 sources, which broadly agreed.",
            "why": ["Some real partial evidence"],
        }

        out = enforce_evidence_threshold(result, available_count=2)

        assert out["verdict"] == "insufficient_data"
        assert out["confidence"] <= 25
        assert "insufficient" in out["headline"].lower() or "not enough" in out["headline"].lower()
        assert "2 of 5" in out["confidence_rationale"]

    def test_at_or_above_threshold_leaves_result_untouched(self):
        result = {"verdict": "hold", "confidence": 60, "headline": "A real memo", "why": ["x"]}
        out = enforce_evidence_threshold(dict(result), available_count=MIN_SOURCES_FOR_VERDICT)
        assert out == result

    def test_above_threshold_leaves_result_untouched(self):
        result = {"verdict": "greenlight", "confidence": 90, "headline": "A real memo"}
        out = enforce_evidence_threshold(dict(result), available_count=5)
        assert out["verdict"] == "greenlight"
        assert out["confidence"] == 90

    def test_partial_evidence_fields_are_not_wiped(self):
        """Only the top-line recommendation fields get overridden — a
        genuinely-grounded 'why' bullet from the 2 available sources is
        still real information and shouldn't be discarded."""
        result = {
            "verdict": "hold", "confidence": 40, "headline": "x",
            "why": ["Production has genuinely begun, confirmed by two sources."],
            "war_room": [{"role": "director", "insight": "real insight"}],
        }
        out = enforce_evidence_threshold(result, available_count=1)
        assert out["why"] == ["Production has genuinely begun, confirmed by two sources."]
        assert out["war_room"] == [{"role": "director", "insight": "real insight"}]

    def test_handles_missing_confidence_key_gracefully(self):
        result = {"verdict": "greenlight", "headline": "x"}
        out = enforce_evidence_threshold(result, available_count=0)
        assert out["confidence"] == 0

    def test_degraded_default_result_also_gets_capped(self):
        """When main_synthesis itself failed entirely (memo is None),
        pipeline.py's fallback dict has verdict=None — this must not
        crash and should still land on insufficient_data."""
        result = {"verdict": None, "confidence": 0, "why": [], "war_room": [], "sources_used": []}
        out = enforce_evidence_threshold(result, available_count=0)
        assert out["verdict"] == "insufficient_data"

    def test_zero_available_sources(self):
        result = {"verdict": "pass", "confidence": 20, "headline": "x"}
        out = enforce_evidence_threshold(result, available_count=0)
        assert out["verdict"] == "insufficient_data"
        assert out["confidence"] <= 25
