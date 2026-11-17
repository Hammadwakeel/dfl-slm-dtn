"""Run TASK K Steps 4-5 after the model and NAS acceptance have passed."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import Settings  # noqa: E402
from m5_feedback_loop import BackendClient, _redact_tokens  # noqa: E402
from mnist_task_k_acceptance import (  # noqa: E402
    DATASET_NAME,
    TaskKError,
    _feedback_acceptance,
    _preflight,
    _training_acceptance,
    _workers,
)

OUTPUT_PATH = ROOT / "data" / "mnist_task_k_training_feedback.json"


def main() -> None:
    settings = Settings.from_env()
    client = BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)
    workers = _workers(settings)
    result: dict[str, Any] = {
        "dataset_name": DATASET_NAME,
        "status": "RUNNING",
        "feedback_rule_fallback_enabled": settings.feedback_rule_fallback_enabled,
        "steps": {},
    }
    try:
        if settings.feedback_rule_fallback_enabled:
            raise TaskKError("Feedback rule fallback must remain disabled")
        _preflight(client, settings, workers)
        training, training_decision, group = _training_acceptance(
            client,
            settings,
            workers,
        )
        result["steps"]["training"] = training
        feedback, feedback_decision = _feedback_acceptance(
            client,
            settings,
            workers,
            group,
        )
        result["steps"]["feedback"] = feedback
        result["decision_history"] = [
            {
                "decision_id": training_decision,
                "events": training["decision_history"],
            },
            {
                "decision_id": feedback_decision,
                "events": (
                    ["created", "approved"] if feedback_decision is not None else []
                ),
            },
        ]
        result["status"] = "PASS" if feedback.get("status") == "PASS" else "PARTIAL"
    except Exception as error:
        result["status"] = "FAIL"
        result["error_type"] = type(error).__name__
        result["error"] = str(_redact_tokens(str(error), settings))
        raise
    finally:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        safe_result = _redact_tokens(result, settings)
        OUTPUT_PATH.write_text(
            json.dumps(safe_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"training_feedback_json={OUTPUT_PATH}")
        print(json.dumps(safe_result, sort_keys=True))


if __name__ == "__main__":
    main()
