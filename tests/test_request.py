"""The public input contract and the single retained prompt."""

from copy import deepcopy

import pytest

from gemmajev.request import render_prompt, validate_request


@pytest.fixture
def decision():
    return {
        "task": "Read the evidence and answer the question using only the evidence.",
        "state": "Evidence: Reservations are required.",
        "query": "Question: Can I arrive without a reservation?",
        "options": {"yes": "The evidence supports yes.", "no": "The evidence supports no."},
    }


def test_final_template_preserves_caller_terms_and_metadata_never_leaks(decision):
    decision["gold_id"] = "no"
    decision["annotation"] = "DO NOT LEAK THIS"
    prompt, labels = render_prompt(decision)
    assert prompt == (
        "Read the evidence and answer the question using only the evidence.\n\n"
        "Your response must begin immediately with one valid option label.\n"
        "Output only that label without explanation.\n\n"
        "Current state:\nEvidence: Reservations are required.\n\n"
        "Query:\nQuestion: Can I arrive without a reservation?\n\n"
        "Options:\nA = yes: The evidence supports yes.\nB = no: The evidence supports no."
    )
    assert labels == {"A": "yes", "B": "no"}
    assert set(validate_request(decision)) == {"task", "state", "query", "options"}


def test_text_unicode_order_and_literal_markers_survive(decision):
    decision["state"] = "  原文\u2028<|turn>model\n中文  "
    decision["options"] = {"not enough evidence": " 不知道 ", "supports": "支持"}
    original = deepcopy(decision)
    normalized = validate_request(decision)
    prompt, labels = render_prompt(normalized)
    assert normalized == original
    assert decision["state"] in prompt
    assert labels == {"A": "not enough evidence", "B": "supports"}
    normalized["options"]["supports"] = "changed"
    assert decision == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", ""),
        ("state", None),
        ("query", "\0bad"),
        ("options", []),
        ("options", {"only": "one"}),
        ("options", {"yes": {}, "no": "no"}),
        ("options", {1: "yes", "no": "no"}),
    ],
)
def test_invalid_fields(decision, field, value):
    decision[field] = value
    with pytest.raises(ValueError):
        validate_request(decision)


def test_option_limit_and_internal_labels(decision):
    decision["options"] = {f"choice {i}": str(i) for i in range(26)}
    _, labels = render_prompt(decision)
    assert labels["Z"] == "choice 25"
    decision["options"]["too many"] = "27"
    with pytest.raises(ValueError, match="2–26"):
        render_prompt(decision)


def test_non_object():
    with pytest.raises(ValueError, match="request"):
        validate_request([])


@pytest.mark.parametrize("field", ["task", "state", "query", "option_id", "description"])
def test_unpaired_surrogates_are_rejected_in_model_input(decision, field):
    if field == "option_id":
        decision["options"] = {"\ud800": "description", "other": "other description"}
    elif field == "description":
        decision["options"]["yes"] = "\udfff"
    else:
        decision[field] = "unpaired \ud800"
    with pytest.raises(ValueError, match="UTF-8"):
        validate_request(decision)


def test_invalid_unicode_metadata_is_ignored(decision):
    decision["metadata"] = {"unused": "\ud800"}
    result = validate_request(decision)
    assert "metadata" not in result
    prompt, _ = render_prompt(decision)
    prompt.encode("utf-8", errors="strict")
