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
    BoxOfficeInfo,
    CastPerformanceNote,
    CastPersonalNote,
    CastResult,
    CompetitiveResult,
    CompetitorInfo,
    EntityResolution,
    GreenlightMemo,
    MarketingResult,
    NewsItem,
    NewsResult,
    ReleaseWindowAdvice,
    SentimentSynthesisResult,
    SourceExcerpt,
    WarRoomVoice,
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


def _six_voices():
    """Helper: a valid, minimal set of the 6 required war_room voices."""
    return [
        WarRoomVoice(role="director", insight="Audience responding to the character arc."),
        WarRoomVoice(role="producer", insight="Competitive risk is moderate."),
        WarRoomVoice(role="marketing_chief", insight="Trailer leans on spectacle."),
        WarRoomVoice(role="casting_executive", insight="Lead actor generating outsized buzz."),
        WarRoomVoice(role="distribution_executive", insight="Window overlaps a major rival."),
        WarRoomVoice(role="analyst", insight="Confidence would rise with more box-office data."),
    ]


class TestGreenlightMemo:
    def test_sources_used_stays_top_level(self):
        """Direct regression test for the original schema-drift bug:
        sources_used must be a sibling of the other memo fields, not
        nested inside a sub-object."""
        memo = GreenlightMemo(
            headline="h", verdict="greenlight_with_changes", confidence=74,
            why=["Strong audience interest", "Moderate competitive risk"],
            biggest_opportunity="Character-driven marketing angle",
            biggest_risk="Release-window congestion",
            recommended_action="Reposition trailer around character conflict",
            war_room=_six_voices(),
            sources_used=["web", "cast", "marketing"],
        )
        dumped = memo.model_dump()
        assert "sources_used" in dumped
        assert dumped["sources_used"] == ["web", "cast", "marketing"]

    def test_war_room_requires_exactly_six_voices(self):
        with pytest.raises(ValidationError):
            GreenlightMemo(
                headline="h", verdict="greenlight", confidence=80, why=["x"],
                biggest_opportunity="o", biggest_risk="r", recommended_action="a",
                war_room=_six_voices()[:5],  # only 5
            )

    def test_lessons_learned_defaults_empty(self):
        memo = GreenlightMemo(
            headline="h", verdict="greenlight", confidence=80, why=["x"],
            biggest_opportunity="o", biggest_risk="r", recommended_action="a",
            war_room=_six_voices(),
        )
        assert memo.lessons_learned == []

    def test_what_we_would_change_defaults_empty(self):
        """Regression guard: released titles should be constructible with
        zero forward-looking changes — there's nothing left to change
        about something that already happened."""
        memo = GreenlightMemo(
            headline="h", verdict="hold", confidence=60, why=["x"],
            biggest_opportunity="o", biggest_risk="r", recommended_action="a",
            war_room=_six_voices(),
        )
        assert memo.what_we_would_change == []

    def test_confidence_bounded_0_to_100(self):
        with pytest.raises(ValidationError):
            GreenlightMemo(
                headline="h", verdict="greenlight", confidence=150, why=["x"],
                biggest_opportunity="o", biggest_risk="r", recommended_action="a",
                war_room=_six_voices(),
            )

    def test_verdict_enum_rejects_invalid(self):
        with pytest.raises(ValidationError):
            GreenlightMemo(
                headline="h", verdict="maybe", confidence=50, why=["x"],
                biggest_opportunity="o", biggest_risk="r", recommended_action="a",
                war_room=_six_voices(),
            )

    def test_war_room_voice_role_enum(self):
        with pytest.raises(ValidationError):
            WarRoomVoice(role="intern", insight="x")


class TestReleaseWindowAdvice:
    def test_never_implies_hypothetical_outcome_by_default(self):
        """This field exists specifically to hold a grounded, directional
        read — not a fabricated counterfactual. Defaults must be inert."""
        advice = ReleaseWindowAdvice(congestion="unclear", reasoning="", suggested_direction="unclear")
        assert advice.suggested_direction == "unclear"

    def test_rejects_invalid_direction(self):
        with pytest.raises(ValidationError):
            ReleaseWindowAdvice(congestion="high", reasoning="x", suggested_direction="move_to_december_15")

    def test_competitive_result_defaults_release_window_unclear(self):
        c = CompetitiveResult(risk="unclear", attention_assessment="No data.")
        assert c.release_window.suggested_direction == "unclear"


class TestWebSentimentAndCompetitive:
    def test_web_sentiment_requires_overall_sentiment(self):
        with pytest.raises(ValidationError):
            WebSentimentResult()  # overall_sentiment has no default

    def test_competitive_result_risk_enum(self):
        with pytest.raises(ValidationError):
            CompetitiveResult(risk="catastrophic")

    def test_box_office_defaults_to_unclear_not_fabricated(self):
        """Regression guard: an upcoming title's CompetitiveResult must be
        constructible with zero box office data, not forced to invent
        figures for something that hasn't released."""
        c = CompetitiveResult(risk="unclear", attention_assessment="No competition data.")
        assert c.box_office.verdict == "unclear"
        assert c.box_office.worldwide_gross is None

    def test_box_office_accepts_full_released_data(self):
        c = CompetitiveResult(
            risk="low", attention_assessment="Held its own against rivals.",
            box_office=BoxOfficeInfo(
                budget="$200 million", worldwide_gross="$900 million",
                verdict="hit", verdict_basis="4.5x budget-to-gross ratio",
            ),
        )
        assert c.box_office.verdict == "hit"

    def test_competitor_gross_estimate_optional(self):
        comp = CompetitorInfo(title="Rival Film", strength_vs_searched="bigger franchise")
        assert comp.gross_estimate is None
        comp_with_gross = CompetitorInfo(title="Rival Film", strength_vs_searched="bigger franchise", gross_estimate="$1.2B")
        assert comp_with_gross.gross_estimate == "$1.2B"

    def test_news_result_facts_and_rumors_independent(self):
        n = NewsResult(
            facts=[NewsItem(url="https://x.com", claim="Confirmed December release", category="release")],
            rumors=[NewsItem(url="https://x.com", claim="Possible sequel in talks", category="other")],
        )
        assert len(n.facts) == 1
        assert len(n.rumors) == 1
        assert n.facts[0].category == "release"

    def test_news_item_rejects_invalid_category(self):
        with pytest.raises(ValidationError):
            NewsItem(url="https://x.com", claim="x", category="gossip")

    def test_sentiment_synthesis_sources_available_required(self):
        with pytest.raises(ValidationError):
            SentimentSynthesisResult(overall_sentiment="positive", justification="j", agreement_note="a")

    def test_timeline_defaults_empty_not_fabricated(self):
        """Regression guard: sentiment_synthesis must be constructible
        with zero timeline entries — this is a bonus signal only
        available when real dates exist, never a required guess."""
        s = SentimentSynthesisResult(
            overall_sentiment="positive", justification="j", agreement_note="a", sources_available=["web"],
        )
        assert s.timeline == []

    def test_timeline_accepts_valid_entries(self):
        from agents.schemas import ReputationTimelineEntry
        s = SentimentSynthesisResult(
            overall_sentiment="mixed", justification="j", agreement_note="a", sources_available=["youtube"],
            timeline=[
                ReputationTimelineEntry(milestone="Trailer", date="2021-12-09", sentiment="positive", note="Trailer response was strongly positive."),
                ReputationTimelineEntry(milestone="Opening Weekend", date="2022-03-25", sentiment="mixed", note="Reaction cooled slightly after release."),
            ],
        )
        assert len(s.timeline) == 2
        assert s.timeline[0].milestone == "Trailer"
        assert s.timeline[0].date == "2021-12-09"

    def test_timeline_milestone_is_free_form_not_a_fixed_enum(self):
        """Regression guard for the too-narrow-coverage bug: the timeline
        used to be capped to a fixed 5-phase enum (pre_release/trailer/
        opening_weekend/week_two/long_tail), which meant an announcement
        or casting reveal could never appear on it even when a real date
        was available. milestone is deliberately a free string now, not
        an enum, so any accurately-named, dated event is valid."""
        from agents.schemas import ReputationTimelineEntry
        entry = ReputationTimelineEntry(
            milestone="Casting Announcement", date="2025-03-26",
            sentiment="positive", note="Cast reveal livestream drew record viewership.",
        )
        assert entry.milestone == "Casting Announcement"

    def test_timeline_entry_requires_a_date(self):
        from agents.schemas import ReputationTimelineEntry
        with pytest.raises(ValidationError):
            ReputationTimelineEntry(milestone="Trailer", sentiment="positive", note="x")

    def test_timeline_entry_caps_at_eight(self):
        from agents.schemas import ReputationTimelineEntry
        nine = [
            ReputationTimelineEntry(milestone=f"Milestone {i}", date="2022-01-01", sentiment="unclear", note="x")
            for i in range(9)
        ]
        with pytest.raises(ValidationError):
            SentimentSynthesisResult(
                overall_sentiment="unclear", justification="j", agreement_note="a",
                sources_available=["web"], timeline=nine,
            )
