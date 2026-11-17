import json
import sys
from io import StringIO
from types import SimpleNamespace

import pytest

from master import reply_inference
from master.reply_service import ReplyInputError, ReplyModel, build_user_prompt


def test_reply_prompt_includes_verified_context_as_json() -> None:
    prompt = build_user_prompt(
        "What should I do next?",
        {"stored_mnist_results": "Validation accuracy was 82%."},
    )

    assert prompt == (
        'What should I do next?\n\nVerified DTN context:\n'
        '{"stored_mnist_results":"Validation accuracy was 82%."}'
    )


def test_reply_instructions_restrict_next_steps_and_keep_them_concise() -> None:
    from master.reply_service import MAX_INPUT_TOKENS, MAX_NEW_TOKENS, SYSTEM_PROMPT

    assert "'continue current action'" in SYSTEM_PROMPT
    assert "'perform NAS'" in SYSTEM_PROMPT
    assert "'perform topology learning'" in SYSTEM_PROMPT
    assert "'perform training'" in SYSTEM_PROMPT
    assert "recommend exactly one action" in SYSTEM_PROMPT
    assert "Never ask for more context" in SYSTEM_PROMPT
    assert MAX_INPUT_TOKENS == 2048
    assert MAX_NEW_TOKENS == 80


@pytest.mark.parametrize(
    ("message", "context"),
    [
        ("", {}),
        ("   ", {}),
        ("x" * 4001, {}),
        ("What next?", {"context": "x" * 8001}),
        ("What next?", {"not_json": object()}),
    ],
)
def test_reply_prompt_rejects_invalid_or_oversized_input(message, context) -> None:
    with pytest.raises(ValueError):
        build_user_prompt(message, context)


def reply_model(tmp_path, monkeypatch) -> ReplyModel:
    base_model = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base_model.mkdir()
    adapter.mkdir()
    report = tmp_path / "training_report.json"
    report.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    monkeypatch.setenv("DFL_REPLY_API_TOKEN", "a" * 48)
    monkeypatch.setenv("DFL_REPLY_BASE_MODEL", str(base_model))
    monkeypatch.setenv("DFL_REPLY_ADAPTER", str(adapter))
    monkeypatch.setenv("DFL_REPLY_REPORT_PATH", str(report))
    return ReplyModel()


def test_reply_generation_uses_a_bounded_one_shot_worker(
    tmp_path,
    monkeypatch,
) -> None:
    model = reply_model(tmp_path, monkeypatch)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"reply": "I recommend reviewing validation results."}),
            stderr="",
        )

    monkeypatch.setattr("master.reply_service.subprocess.run", fake_run)
    reply, latency_ms = model.generate(
        "What next?",
        {"stored_mnist_results": "Validation accuracy was 82%."},
    )

    assert reply == "I recommend reviewing validation results."
    assert latency_ms >= 0
    command, kwargs = calls[0]
    assert command == [sys.executable, "-m", "master.reply_inference"]
    assert kwargs["timeout"] == 120
    assert kwargs["check"] is False
    assert json.loads(kwargs["input"])["context"] == {
        "stored_mnist_results": "Validation accuracy was 82%."
    }


def test_reply_generation_surfaces_worker_failures(tmp_path, monkeypatch) -> None:
    model = reply_model(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "master.reply_service.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="OOM"),
    )

    with pytest.raises(RuntimeError, match="inference process failed"):
        model.generate("What next?", {})


def test_reply_generation_classifies_worker_input_errors(tmp_path, monkeypatch) -> None:
    model = reply_model(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "master.reply_service.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"error": "request exceeds the 2048-token fine-tuned model input limit"}
            ),
            stderr="",
        ),
    )

    with pytest.raises(ValueError, match="2048-token"):
        model.generate("What next?", {})


def test_inference_worker_serializes_request_validation_errors(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        reply_inference.sys,
        "stdin",
        StringIO(json.dumps({"message": "What next?", "context": {}})),
    )

    def reject_request(_message, _context):
        raise ReplyInputError("request exceeds the 2048-token fine-tuned model input limit")

    monkeypatch.setattr(reply_inference, "generate", reject_request)
    reply_inference.main()

    assert json.loads(capsys.readouterr().out) == {
        "error": "request exceeds the 2048-token fine-tuned model input limit"
    }
