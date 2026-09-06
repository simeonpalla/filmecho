"""Tests for orchestration/refusal_guard.py.

Direct regression test for the observed cast_agent failure: given an
empty cast list (normal for large ensemble films), it wrote "I can't
provide a cast reception... I need to know the cast members first" into
a plain string field instead of researching the cast from its own search
results. output_schema enforces shape, not content, so that sentence
passed Pydantic validation cleanly — these tests are for the
content-level check layered on top.
"""

from pydantic import BaseModel

from orchestration.refusal_guard import contains_refusal_text, drop_if_refusal


class _Dummy(BaseModel):
    overall_cast_reception: str = ""
    nested: dict = {}
    items: list = []


class TestContainsRefusalText:
    def test_detects_the_actual_observed_failure_sentence(self):
        text = "I can't provide a cast reception for 'Avengers: Doomsday' as I need to know the cast members first."
        assert contains_refusal_text(text) is True

    def test_detects_as_an_ai_disclaimer(self):
        assert contains_refusal_text("As an AI, I cannot speculate on unreleased box office figures.") is True

    def test_detects_unable_to_phrasing(self):
        assert contains_refusal_text("I'm unable to determine the cast without more information.") is True

    def test_normal_content_is_not_flagged(self):
        text = "Ram Charan and Jr NTR received universal critical acclaim for their performances."
        assert contains_refusal_text(text) is False

    def test_case_insensitive(self):
        assert contains_refusal_text("I CANNOT provide this without a cast list.") is True

    def test_recurses_into_nested_dicts(self):
        value = {"performances": [], "overall_cast_reception": "I need to know the cast members first."}
        assert contains_refusal_text(value) is True

    def test_recurses_into_lists(self):
        value = ["a normal claim", "I don't have enough information to answer this."]
        assert contains_refusal_text(value) is True

    def test_empty_and_none_are_not_flagged(self):
        assert contains_refusal_text("") is False
        assert contains_refusal_text(None) is False
        assert contains_refusal_text({}) is False
        assert contains_refusal_text([]) is False


class TestDropIfRefusal:
    def test_none_input_stays_none(self):
        assert drop_if_refusal("cast", None) is None

    def test_clean_model_passes_through_unchanged(self):
        model = _Dummy(overall_cast_reception="A strong ensemble performance overall.")
        assert drop_if_refusal("cast", model) is model

    def test_refusal_model_is_dropped_to_none(self):
        model = _Dummy(overall_cast_reception="I can't provide a cast reception as I need to know the cast members first.")
        assert drop_if_refusal("cast", model) is None

    def test_refusal_buried_in_nested_field_is_still_caught(self):
        model = _Dummy(nested={"note": "I cannot determine this without more data."})
        assert drop_if_refusal("competitive", model) is None
