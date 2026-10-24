from __future__ import annotations

import json

from jsonschema import Draft202012Validator

from scripts.slm_dataset import (
    ACTION_BRANCHES,
    ACTION_SCHEMA,
    ACTION_VALIDATOR,
    CHAT_VALIDATOR,
    OPERATIONAL_ACTIONS,
    build_label,
    contains_sensitive_text,
    deduplicate_records,
    parse_utterances,
    parse_phase3_legacy,
    sample_parameters,
    safe_error,
    split_by_template_family,
)


def test_label_builder_produces_schema_valid_chat_and_action() -> None:
    label = json.loads(build_label("set_rounds", {"rounds": 100}, "normal"))
    CHAT_VALIDATOR.validate(label)
    ACTION_VALIDATOR.validate(label["action"])
    assert label["action"] == {
        "action": "set_rounds",
        "parameters": {"rounds": 100},
    }


def test_conversational_label_templates_match_category_behavior() -> None:
    ambiguous = json.loads(build_label("clarify", {}, "ambiguous"))
    compound = json.loads(build_label("clarify", {}, "compound"))
    invalid = json.loads(build_label("refuse", {}, "invalid_value"))
    unsafe = json.loads(build_label("refuse", {}, "unsafe"))
    redirect = json.loads(build_label("redirect", {}, "chit_chat"))
    for label in (ambiguous, compound, invalid, unsafe, redirect):
        CHAT_VALIDATOR.validate(label)
    assert ambiguous["action"]["action"] == "clarify"
    assert "clarify" in ambiguous["reply"].casefold()
    assert "first" in compound["reply"]
    assert "1–100" in invalid["reply"]
    assert unsafe["action"]["action"] == "refuse"
    assert redirect["action"]["action"] == "redirect"


def test_all_schema_actions_can_sample_valid_parameters() -> None:
    Draft202012Validator.check_schema(ACTION_SCHEMA)
    assert len(ACTION_BRANCHES) == 18
    for action, branch in ACTION_BRANCHES.items():
        parameters = sample_parameters(action, 0)
        ACTION_VALIDATOR.validate({"action": action, "parameters": parameters})
        if action in {"clarify", "refuse", "redirect"}:
            assert parameters == {}


def test_parameter_sampler_covers_schema_edges() -> None:
    rounds = [sample_parameters("set_rounds", index)["rounds"] for index in range(4)]
    epochs = [sample_parameters("set_epochs", index)["epochs"] for index in range(4)]
    assert rounds[:2] == [1, 100]
    assert epochs[:2] == [1, 50]
    assert len(OPERATIONAL_ACTIONS) == len(ACTION_BRANCHES) - 3


def test_duplicate_removal_handles_exact_normalized_and_fuzzy_text() -> None:
    rows = [
        {"user": "Check master status!", "id": "1"},
        {"user": "check master status", "id": "2"},
        {"user": "Check the master status", "id": "3"},
        {"user": "Set rounds to ten", "id": "4"},
    ]
    kept, exact, near = deduplicate_records(rows, threshold=0.8)
    assert exact == 1
    assert near == 1
    assert [row["id"] for row in kept] == ["1", "4"]


def test_family_split_has_no_leakage_and_approximate_proportions() -> None:
    rows = []
    for family in range(20):
        for sample in range(10):
            rows.append(
                {
                    "id": f"{family}-{sample}",
                    "template_family": f"family-{family}",
                }
            )
    splits = split_by_template_family(rows, seed=7)
    family_sets = {
        name: {row["template_family"] for row in values}
        for name, values in splits.items()
    }
    assert family_sets["train"].isdisjoint(family_sets["validation"])
    assert family_sets["train"].isdisjoint(family_sets["test"])
    assert family_sets["validation"].isdisjoint(family_sets["test"])
    assert len(splits["train"]) == 160
    assert abs(len(splits["validation"]) - 20) <= 10
    assert abs(len(splits["test"]) - 20) <= 10


def test_family_split_keeps_each_category_in_validation_and_test() -> None:
    rows = [
        {
            "id": f"{category}-{family}-{sample}",
            "category": category,
            "template_family": f"{category}-{family}",
        }
        for category in ("normal", "unsafe", "chit_chat")
        for family in range(5)
        for sample in range(10)
    ]
    splits = split_by_template_family(rows, seed=7)
    for split_name in ("validation", "test"):
        categories = {row["category"] for row in splits[split_name]}
        assert categories == {"normal", "unsafe", "chit_chat"}


def test_sensitive_text_filter_catches_secrets_and_personal_data() -> None:
    assert contains_sensitive_text("email me at person@example.com")
    assert contains_sensitive_text("token=supersecretcredential123")
    assert contains_sensitive_text("server 192.168.1.14")
    assert not contains_sensitive_text("Set the training run to 100 rounds.")


def test_utterance_parser_accepts_short_batch_but_rejects_truncation() -> None:
    text = json.dumps({"utterances": [f"synthetic request {index}" for index in range(19)]})
    assert len(parse_utterances(text)) == 19
    try:
        parse_utterances('{"utterances": ["incomplete"]')
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("Malformed/truncated JSON was accepted")


def test_api_error_redaction_removes_secrets_and_generated_payloads() -> None:
    error = RuntimeError(
        "invalid request; failed_generation: generated prompt text; gsk_test_secret_1234567890"
    )
    message = safe_error(error, {"GROQ_API_KEY1": "gsk_test_secret_1234567890"})
    assert "generated prompt text" not in message
    assert "gsk_test_secret_1234567890" not in message
    assert "[REDACTED]" in message


def test_legacy_prompts_are_loaded_from_phase3_report() -> None:
    rows = parse_phase3_legacy()
    assert len(rows) == 20
    assert rows[0]["action"] == "get_status"
    assert rows[5]["action"] == "get_logs"
    assert rows[4]["params"] == {}
    assert rows[17]["params"] == {"run_ids": ["run_001", "run_002"]}
