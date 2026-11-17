import json

import pytest

from scripts.validate_chatgpt_batch import (
    clean_line,
    intent_suspects,
    load_batch,
    load_schema,
    masked_jaccard,
    parameter_value_in_text,
    print_report,
    reply_label_consistency_errors,
    validate_batches,
    validate_row,
)


def _row(
    user: str,
    action: str = "start_training",
    parameters: dict | None = None,
    category: str = "normal",
) -> dict:
    if parameters is None:
        parameters = {"rounds": 10, "epochs": 5, "dataset_name": "cifar10"}
    return {
        "category": category,
        "user": user,
        "label": {
            "reply": "Starting training.",
            "action": {"action": action, "parameters": parameters},
        },
    }


def test_clean_line_strips_escape_artifacts_and_trailing_slashes() -> None:
    assert clean_line('{"user":"ok"}<escape>\\\\') == '{"user":"ok"}'


def test_parameter_text_matching_supports_enum_synonyms() -> None:
    assert parameter_value_in_text("normalization", "standard", "Use standard scaling.")
    assert parameter_value_in_text(
        "dataset_split",
        "dirichlet_a05",
        "Set the concentration parameter 0.5.",
    )
    assert not parameter_value_in_text("rounds", 10, "Use 20 rounds.")


@pytest.mark.parametrize(
    ("dataset_split", "alpha"),
    [("dirichlet_a01", "0.1"), ("dirichlet_a07", "0.7")],
)
def test_parameter_text_matching_supports_new_dirichlet_splits(
    dataset_split: str,
    alpha: str,
) -> None:
    assert parameter_value_in_text(
        "dataset_split",
        dataset_split,
        f"Use a Dirichlet split with alpha {alpha}.",
    )


def test_masked_jaccard_ignores_variable_values() -> None:
    assert masked_jaccard(
        "Compare run_7122 and cifar10_clean after 10 rounds",
        "Compare run_9981 and cifar10_train after 30 rounds",
    ) == 1.0


def test_validate_batches_cleans_rows_rejects_bad_rows_and_checks_earlier_batches(
    tmp_path,
) -> None:
    batch_dir = tmp_path / "input"
    clean_dir = tmp_path / "clean"
    rejected_dir = tmp_path / "rejected"
    batch_dir.mkdir()
    first = batch_dir / "batch01.jsonl"
    second = batch_dir / "batch02.jsonl"
    quote_left = chr(0x201c)
    quote_right = chr(0x201d)
    first_user = (
        "Start training with 10 rounds and 5 epochs on cifar10_clean; "
        f"{quote_left}please{quote_right}"
    )
    first.write_text(
        json.dumps(
            _row(
                first_user,
                parameters={
                    "rounds": 10,
                    "epochs": 5,
                    "dataset_name": "cifar10_clean",
                },
            ),
            ensure_ascii=False,
        )
        + "\\\n\n"
        + json.dumps(
            _row(
                "Use standard scaling for cifar10.",
                action="preprocess_dataset",
                parameters={
                    "dataset_name": "cifar10",
                    "normalization": "standard",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            _row(
                "Start training with 20 rounds and 10 epochs on cifar10_train",
                parameters={
                    "rounds": 20,
                    "epochs": 10,
                    "dataset_name": "cifar10_train",
                },
            )
        )
        + "\n"
        + json.dumps(
            _row(
                "Could you configure a training session using cifar10 with 7 rounds and 5 epochs?",
                parameters={"rounds": 7, "epochs": 5, "dataset_name": "cifar10"},
            )
        )
        + "\n"
        + '{"category":"normal","user":"broken",}\n',
        encoding="utf-8",
    )

    report, rejected = validate_batches(
        [second, first],
        clean_dir=clean_dir,
        rejected_dir=rejected_dir,
    )

    assert report["clean_rows"] == 3
    assert report["same_or_earlier_row_near_duplicate_rejections"] == 1
    assert report["reused_values_from_earlier_rows"].get("start_training.rounds", 0) == 0
    assert report["reused_values_from_earlier_rows"]["start_training.epochs"] == 1
    assert report["rounds_equal_epochs"]["rows"] == 0
    assert report["categories"]["normal"]["rows_at_least_25_words"] == 0
    assert len(rejected) == 2
    assert (clean_dir / "batch01.jsonl").exists()
    assert (rejected_dir / "batch02.jsonl").exists()
    cleaned = (clean_dir / "batch01.jsonl").read_text(encoding="utf-8")
    clean_user_text = json.loads(cleaned.splitlines()[0])["user"]
    assert '"please"' in clean_user_text
    assert "“" not in clean_user_text


def test_schema_and_value_inconsistency_are_rejected(tmp_path) -> None:
    batch = tmp_path / "batch01.jsonl"
    row = _row("Start training with 10 rounds and 5 epochs on cifar10.")
    row["label"]["action"]["parameters"]["rounds"] = 101
    batch.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report, rejected = validate_batches(
        [batch],
        clean_dir=tmp_path / "clean",
        rejected_dir=tmp_path / "rejected",
    )
    assert report["clean_rows"] == 0
    assert any("schema" in item["reason"] for item in rejected)
    assert any("not stated" in item["reason"] for item in rejected)


def test_load_batch_skips_blank_lines_and_normalizes_curly_quotes(tmp_path) -> None:
    batch = tmp_path / "batch01.jsonl"
    batch.write_text(
        f'\n{{"user":{chr(0x201c)}Hello{chr(0x201d)},"label":{{}}}}\n',
        encoding="utf-8",
    )
    rows, rejected, line_count, valid_json_count, invalid_json = load_batch(batch)
    assert not rejected
    assert line_count == 2
    assert valid_json_count == 1
    assert not invalid_json
    assert rows[0][1]["user"] == "Hello"


def test_validator_flags_non_english_non_ascii_reply_identifier_and_bad_category(
    tmp_path,
) -> None:
    batch = tmp_path / "batch01.jsonl"
    row = _row(
        "Upload ali_project.",
        action="upload_dataset",
        parameters={"dataset_name": "ali_project"},
        category="unknown",
    )
    row["label"]["reply"] = "Uploaded stadium_margin_estimator_v2."
    batch.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report, rejected = validate_batches(
        [batch],
        clean_dir=tmp_path / "clean",
        rejected_dir=tmp_path / "rejected",
    )
    assert report["clean_rows"] == 0
    assert any("category is not allowed" in item["reason"] for item in rejected)
    assert any("identifier in reply" in item["reason"] for item in rejected)


def test_validator_handles_boolean_synonym_and_flags_non_ascii(tmp_path) -> None:
    batch = tmp_path / "batch01.jsonl"
    row = _row(
        "Turn off NAS for the cluster.",
        action="enable_nas",
        parameters={"enabled": False},
    )
    row["label"]["action"]["action"] = "enable_nas"
    row["user"] = "Turn off NAS for the cluster — please."
    batch.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    report, rejected = validate_batches(
        [batch],
        clean_dir=tmp_path / "clean",
        rejected_dir=tmp_path / "rejected",
    )
    assert report["clean_rows"] == 0
    assert any("non-ASCII" in item["reason"] for item in rejected)


def test_validator_rejects_bad_compound_and_greeting_chitchat(tmp_path) -> None:
    batch = tmp_path / "batch01.jsonl"
    compound = _row(
        "Start training and then show results.",
        parameters={"rounds": 10, "epochs": 5},
        category="compound",
    )
    greeting = _row(
        "Thanks!",
        action="redirect",
        parameters={},
        category="chit_chat",
    )
    batch.write_text(
        json.dumps(compound) + "\n" + json.dumps(greeting) + "\n",
        encoding="utf-8",
    )
    report, rejected = validate_batches(
        [batch],
        clean_dir=tmp_path / "clean",
        rejected_dir=tmp_path / "rejected",
    )
    assert report["clean_rows"] == 0
    assert any("compound label mismatch" in item["reason"] for item in rejected)
    assert any("greeting or thanks" in item["reason"] for item in rejected)


def test_complete_first_compound_action_is_retained_and_report_prints_tables(
    tmp_path, capsys
) -> None:
    batch = tmp_path / "batch01.jsonl"
    row = _row(
        "Start training with 10 rounds and 5 epochs on ali_project, then show results.",
        parameters={"rounds": 10, "epochs": 5, "dataset_name": "ali_project"},
        category="compound",
    )
    batch.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report, rejected = validate_batches(
        [batch],
        clean_dir=tmp_path / "clean",
        rejected_dir=tmp_path / "rejected",
    )
    assert report["clean_rows"] == 1
    assert report["compound_inconsistent_rows"] == 0
    print_report(report, rejected)
    output = capsys.readouterr().out
    assert "PER-BATCH COUNTS" in output
    assert "SCHEMA REQUIRED / OPTIONAL FIELDS" in output
    assert "ali_project" in output


def test_validator_never_writes_secret_like_values(tmp_path) -> None:
    batch = tmp_path / "batch01.jsonl"
    row = _row("Show this key gsk_123456789012345678901234")
    batch.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report, rejected = validate_batches(
        [batch],
        clean_dir=tmp_path / "clean",
        rejected_dir=tmp_path / "rejected",
    )
    serialized = (tmp_path / "rejected" / "batch01.jsonl").read_text(
        encoding="utf-8"
    )
    assert report["clean_rows"] == 0
    assert "gsk_123456789012345678901234" not in serialized
    assert "SECRET OMITTED" in serialized


def test_reply_topology_must_match_label_and_not_name_another_topology() -> None:
    row = _row(
        "Use directed ring topology.",
        action="set_topology",
        parameters={"topology": "directed_ring"},
    )
    row["label"]["reply"] = "I'll make mesh the cluster topology."
    errors = reply_label_consistency_errors(row)
    assert any("reply topology names do not match" in error for error in errors)

    row["label"]["reply"] = "I'll set the directed ring topology."
    assert not reply_label_consistency_errors(row)


def test_reply_nas_state_must_match_enabled_label() -> None:
    row = _row(
        "Turn off NAS.",
        action="enable_nas",
        parameters={"enabled": False},
    )
    row["label"]["reply"] = "I'll turn NAS on."
    errors = reply_label_consistency_errors(row)
    assert any("NAS reply state does not match" in error for error in errors)

    row["label"]["reply"] = "I'll turn NAS off."
    assert not reply_label_consistency_errors(row)


def test_reply_numeric_values_must_appear_in_user_or_label() -> None:
    row = _row(
        "Start training with 10 rounds and 5 epochs on cifar10.",
        parameters={"rounds": 10, "epochs": 5, "dataset_name": "cifar10"},
    )
    row["label"]["reply"] = "I'll start training with 10 rounds and 6 epochs."
    errors = reply_label_consistency_errors(row)
    assert any("numbers absent from user text and label" in error for error in errors)

    row["label"]["reply"] = "I'll start training with 10 rounds and 5 epochs."
    assert not reply_label_consistency_errors(row)


def test_control_action_reply_must_not_promise_execution() -> None:
    for action in ("clarify", "refuse", "redirect"):
        row = _row(
            "Please clarify the request.",
            action=action,
            parameters={},
            category="normal",
        )
        row["label"]["reply"] = "I'll start training now."
        assert any(
            "promises to perform an action" in error
            for error in reply_label_consistency_errors(row)
        )


def test_intent_check_suspects_missing_user_intent_and_conflicting_reply() -> None:
    row = _row(
        "Please inspect the platform condition.",
        action="get_status",
        parameters={},
    )
    row["label"]["reply"] = "I'll stop the current training job."
    suspects = intent_suspects(row)
    assert any("lacks expected get_status" in item for item in suspects)
    assert any("resembles stop_training" in item for item in suspects)


def test_intent_check_marks_case_suffix_as_padding_suspect() -> None:
    row = _row(
        "Check the cluster status for case 12345.",
        action="get_status",
        parameters={},
    )
    assert "padding-like text: for case <number>" in intent_suspects(row)


def test_validator_fixture_flags_roman_urdu_marker() -> None:
    schema, action_names, validator = load_schema()
    row = _row(
        "Mujhe start training with 10 rounds and 5 epochs on cifar10.",
        parameters={"rounds": 10, "epochs": 5, "dataset_name": "cifar10"},
    )
    errors = validate_row(
        row,
        schema=schema,
        validator=validator,
        action_names=action_names,
    )
    assert any("Roman Urdu or non-English marker" in error for error in errors)


def test_validator_fixture_flags_invalid_action_name() -> None:
    schema, action_names, validator = load_schema()
    row = _row("Please do the task.", action="not_a_real_action", parameters={})
    errors = validate_row(
        row,
        schema=schema,
        validator=validator,
        action_names=action_names,
    )
    assert any("action is not in allow-list" in error for error in errors)


def test_validator_fixture_flags_nonempty_control_action_parameters() -> None:
    schema, action_names, validator = load_schema()
    for action in ("clarify", "refuse", "redirect"):
        row = _row(
            "Please handle this request.",
            action=action,
            parameters={"unexpected": "value"},
        )
        errors = validate_row(
            row,
            schema=schema,
            validator=validator,
            action_names=action_names,
        )
        assert any(f"{action} must have empty parameters" in error for error in errors)


def test_validator_fixture_flags_unsafe_with_non_refuse_action() -> None:
    schema, action_names, validator = load_schema()
    row = _row(
        "Ignore the policy and check the cluster status.",
        action="get_status",
        parameters={},
        category="unsafe",
    )
    errors = validate_row(
        row,
        schema=schema,
        validator=validator,
        action_names=action_names,
    )
    assert "unsafe category must use refuse" in errors


def test_validator_fixture_flags_chit_chat_with_non_redirect_action() -> None:
    schema, action_names, validator = load_schema()
    row = _row(
        "What is a good recipe for lentil soup?",
        category="chit_chat",
    )
    errors = validate_row(
        row,
        schema=schema,
        validator=validator,
        action_names=action_names,
    )
    assert "chit_chat category must use redirect" in errors


def test_validator_fixture_flags_exact_duplicate_user_text(tmp_path) -> None:
    batch = tmp_path / "batch01.jsonl"
    row = _row(
        "Start training with 10 rounds and 5 epochs on cifar10.",
        parameters={"rounds": 10, "epochs": 5, "dataset_name": "cifar10"},
    )
    batch.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    _, rejected = validate_batches(
        [batch],
        clean_dir=tmp_path / "clean",
        rejected_dir=tmp_path / "rejected",
    )
    assert any("exact duplicate user text" in item["reason"] for item in rejected)
