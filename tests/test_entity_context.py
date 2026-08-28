"""Tests for agents/entity_context.py's resolve_entity degrade paths.

resolve_entity accepts parallel_client and genai_client as optional
injected parameters specifically so this is testable without real API
keys, credentials, or network access — these tests exercise that
injection point directly with mocks.

What's tested:
- The happy path produces a fully populated, high-confidence EntityContext.
- A Parallel search failure degrades to a bare-title, low-confidence
  context rather than raising.
- A Gemini extraction failure degrades the same way.
- A non-"high" confidence result triggers exactly one retry search, not
  zero and not more than one (the retry is explicitly capped).
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents.entity_context import resolve_entity


def _mock_parallel_client(excerpts_text="Toxic is a 2025 Malayalam film directed by Geetu Mohandas, starring Yash."):
    """A Parallel client whose .search() returns one result with one excerpt."""
    client = MagicMock()
    result = SimpleNamespace(url="https://example.com", title="Toxic", excerpts=[excerpts_text])
    client.search.return_value = SimpleNamespace(results=[result], session_id="session_test123")
    return client


def _mock_genai_client(payload: dict):
    """A genai client whose generate_content() returns valid JSON text."""
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(payload))
    return client


HIGH_CONFIDENCE_PAYLOAD = {
    "canonical_title": "Toxic",
    "release_year": 2025,
    "director": "Geetu Mohandas",
    "cast": ["Yash"],
    "release_status": "released",
    "confidence": "high",
    "disambiguation_note": "",
}

LOW_CONFIDENCE_PAYLOAD = {
    "canonical_title": "Toxic",
    "release_year": 2025,
    "director": "Geetu Mohandas",
    "cast": ["Yash"],
    "release_status": "released",
    "confidence": "low",
    "disambiguation_note": "Multiple films share this title; picked the most prominent.",
}


class TestHappyPath:
    def test_full_resolution_high_confidence(self):
        parallel = _mock_parallel_client()
        genai = _mock_genai_client(HIGH_CONFIDENCE_PAYLOAD)

        ctx = resolve_entity("Toxic", parallel_client=parallel, genai_client=genai)

        assert ctx.canonical_title == "Toxic"
        assert ctx.release_year == 2025
        assert ctx.director == "Geetu Mohandas"
        assert ctx.confidence == "high"
        assert ctx.release_status == "released"
        # High confidence should NOT trigger a retry search.
        assert parallel.search.call_count == 1
        assert genai.models.generate_content.call_count == 1


class TestDegradePaths:
    def test_parallel_search_failure_degrades_to_bare_title(self):
        parallel = MagicMock()
        parallel.search.side_effect = ConnectionError("network down")
        genai = _mock_genai_client(HIGH_CONFIDENCE_PAYLOAD)

        ctx = resolve_entity("Some Title", parallel_client=parallel, genai_client=genai)

        assert ctx.title == "Some Title"
        assert ctx.canonical_title is None
        assert ctx.confidence == "low"  # dataclass default, never upgraded
        # Gemini should never be called if there were no excerpts to feed it.
        genai.models.generate_content.assert_not_called()

    def test_gemini_failure_degrades_to_bare_title(self):
        parallel = _mock_parallel_client()
        genai = MagicMock()
        genai.models.generate_content.side_effect = RuntimeError("quota exceeded")

        ctx = resolve_entity("Some Title", parallel_client=parallel, genai_client=genai)

        assert ctx.title == "Some Title"
        assert ctx.canonical_title is None
        assert ctx.confidence == "low"

    def test_empty_excerpts_skips_gemini_entirely(self):
        parallel = MagicMock()
        parallel.search.return_value = SimpleNamespace(results=[], session_id="s1")
        genai = _mock_genai_client(HIGH_CONFIDENCE_PAYLOAD)

        ctx = resolve_entity("Some Title", parallel_client=parallel, genai_client=genai)

        assert ctx.canonical_title is None
        genai.models.generate_content.assert_not_called()


class TestDisambiguationRetry:
    def test_low_confidence_triggers_exactly_one_retry(self):
        parallel = _mock_parallel_client()
        genai = MagicMock()
        # First call returns low confidence, second (retry) call returns high.
        genai.models.generate_content.side_effect = [
            SimpleNamespace(text=json.dumps(LOW_CONFIDENCE_PAYLOAD)),
            SimpleNamespace(text=json.dumps(HIGH_CONFIDENCE_PAYLOAD)),
        ]

        ctx = resolve_entity("Toxic", parallel_client=parallel, genai_client=genai)

        # One initial search + one disambiguation retry search = 2 total.
        assert parallel.search.call_count == 2
        assert genai.models.generate_content.call_count == 2
        # Final result should be the retry's (higher-confidence) result.
        assert ctx.confidence == "high"

    def test_retry_failure_keeps_first_low_confidence_result(self):
        """If the retry search itself fails, the pipeline should still
        return the first (low-confidence) result rather than losing all
        data — this is the graceful-degradation contract, not a crash."""
        parallel = MagicMock()
        first_result = SimpleNamespace(url="https://x.com", title="Toxic", excerpts=["some excerpt"])
        parallel.search.side_effect = [
            SimpleNamespace(results=[first_result], session_id="s1"),
            ConnectionError("retry search failed"),
        ]
        genai = _mock_genai_client(LOW_CONFIDENCE_PAYLOAD)

        ctx = resolve_entity("Toxic", parallel_client=parallel, genai_client=genai)

        assert ctx.confidence == "low"
        assert ctx.canonical_title == "Toxic"  # first result still applied, not lost
