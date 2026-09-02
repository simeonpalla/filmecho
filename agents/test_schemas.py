"""Tests for agents/schemas.py.

These don't need any API keys, they test that the Pydantic models
actually enforce what the rest of the codebase assumes they enforce:
that invalid enum values are rejected, that required fields are
required, and that the fields the frontend reads by name actually exist
where it expects them. This last part is exactly the class of bug that
caused the earlier "sources_used nested under fan_pulse" drift, before
output_schema was in place, tests here are meant to catch a regression
of that same shape.
"""

import pytest
from pydantic import ValidationError

from agents.schemas import (
    CastPerformanceNote,
    CastPersonalNote,
    CastResult,
    CompetitiveResult,
    EntityResolution,
    FanPulse,
    FinalBrief,
    MarketingResult,
    NewsResult,
    SentimentSynthesisResult,
    SourceExcerpt,
    StudioBrief,
    WebSentimentResult,
)


class TestEntityResolution:
    def test_accepts_valid_data(self):
        e = EntityResolution(
            canonical_title="Toxic",
            release_year=2025,
            director="Geetu Mohandas",
            cast=["Yash"],
            release_status="released",
            confidence="high",
        )
        assert e.confidence == "high"
        assert e.disambiguation_note == ""  # default

    def test_rejects_invalid_confidence(self):
        with pytest.raises(ValidationError):
            EntityResolution(canonical_title="X", confidence="super-high", release_status="released")

    def test_rejects_invalid_release_status(self):
        with pytest.raises(ValidationError):
            EntityResolution(canonical_title="X", confidence="high", release_status="definitely-out")

    def test_optional_fields_default_sensibly(self):
        e = EntityResolution(canonical_title="X", confidence="low", release_status="unclear")
        assert e.release_year is None
        assert e.director is None
        assert e.cast == []


class TestMarketingResult:
    def test_lessons_learned_defaults_empty(self):
        """Regression guard for the upcoming-title framing bug: nothing
        should force lessons_learned to be populated, an upcoming title's
        marketing_agent run should be able to leave it empty."""
        m = MarketingResult(strategies_observed=[SourceExcerpt(url="https://x.com", claim="teaser drop")])
        assert m.lessons_learned == []

    def test_accepts_full_retrospective_data(self):
        m = MarketingResult(
            strategies_observed=[
                SourceExcerpt(url="https://x.com", claim="trailer"),
                SourceExcerpt(url="https://x.com", claim="influencer push"),
            ],
            what_worked=[SourceExcerpt(url="https://x.com", claim="Strong trailer response")],
            what_underperformed=[SourceExcerpt(url="https://x.com", claim="Poster criticized")],
            lessons_learned=["Front-load reveals earlier next time"],
        )
        assert len(m.what_worked) == 1
        assert m.lessons_learned == ["Front-load reveals earlier next time"]

    def test_lists_are_length_capped(self):
        """Regression guard for the fabricated-statistics run: an agent
        should not be able to return an unbounded wall of claims."""
        too_many = [SourceExcerpt(url="https://x.com", claim=f"claim {i}") for i in range(10)]
        with pytest.raises(ValidationError):
            MarketingResult(what_worked=too_many)


class TestCastResult:
    def test_performances_and_personal_updates_are_separate(self):
        c = CastResult(
            performances=[CastPerformanceNote(actor="Yash", note="Praised for dual role")],
            personal_updates=[CastPersonalNote(actor="Yash", headline="Reported wrist injury during reshoots")],
        )
        assert c.performances[0].actor == "Yash"
        assert c.personal_updates[0].headline == "Reported wrist injury during reshoots"
        # These are meant to stay in different fields — a regression that
        # merged them back together would still pass Pydantic validation
        # but would silently break the Studio/Fan tab split in the UI, so
        # this test exists to make that split an explicit contract.
        assert CastResult.model_fields["performances"] is not CastResult.model_fields["personal_updates"]


class TestFinalBrief:
    def test_sources_used_stays_top_level(self):
        """Direct regression test for the original schema-drift bug:
        sources_used must be a sibling of studio_brief/fan_pulse, not
        nested inside either one."""
        fb = FinalBrief(
            studio_brief=StudioBrief(
                headline="h", sentiment_summary="s", competitive_risk="high", recommendation="r"
            ),
            fan_pulse=FanPulse(headline="h2", excitement_level="high", excitement_reason="Reunion of the original cast"),
            sources_used=["web", "cast", "marketing"],
        )
        dumped = fb.model_dump()
        assert "sources_used" in dumped
        assert "sources_used" not in dumped["fan_pulse"]
        assert "sources_used" not in dumped["studio_brief"]

    def test_studio_brief_lessons_learned_defaults_empty(self):
        sb = StudioBrief(headline="h", sentiment_summary="s", competitive_risk="low", recommendation="r")
        assert sb.lessons_learned == []

    def test_fan_pulse_worth_watching_defaults_empty(self):
        fp = FanPulse(headline="h", excitement_level="mixed", excitement_reason="Mixed early reactions to the trailer")
        assert fp.worth_watching == ""

    def test_fan_pulse_requires_excitement_reason(self):
        """Regression guard: excitement_level alone, with no reason, was
        exactly the 'blank, unexplained' Fan Pulse a real run produced."""
        with pytest.raises(ValidationError):
            FanPulse(headline="h", excitement_level="high")


class TestWebSentimentAndCompetitive:
    def test_web_sentiment_requires_overall_sentiment(self):
        with pytest.raises(ValidationError):
            WebSentimentResult()  # overall_sentiment has no default

    def test_competitive_result_risk_enum(self):
        with pytest.raises(ValidationError):
            CompetitiveResult(risk="catastrophic")

    def test_news_result_facts_and_rumors_independent(self):
        n = NewsResult(
            facts=[SourceExcerpt(url="https://x.com", claim="Confirmed December release")],
            rumors=[SourceExcerpt(url="https://x.com", claim="Possible sequel in talks")],
        )
        assert len(n.facts) == 1
        assert len(n.rumors) == 1

    def test_sentiment_synthesis_sources_available_required(self):
        with pytest.raises(ValidationError):
            SentimentSynthesisResult(overall_sentiment="positive", justification="j", agreement_note="a")
