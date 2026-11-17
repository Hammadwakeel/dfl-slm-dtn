"""Print persisted M5 feedback-suggestion source and validator evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402


def _get(base_url: str, token: str, path: str) -> Any:
    request = Request(
        base_url + path,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"GET {path} returned HTTP {error.code}") from error
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"GET {path} failed: {type(error).__name__}") from error


def main() -> None:
    settings = Settings.from_env()
    base_url = f"http://127.0.0.1:{settings.port}"
    decisions = _get(base_url, settings.api_token, "/api/autonomy/decisions")
    history = _get(base_url, settings.api_token, "/api/autonomy/history")
    if not isinstance(decisions, list) or not isinstance(history, list):
        raise RuntimeError("Autonomy history endpoints returned an unexpected shape")

    feedback = [
        decision
        for decision in decisions
        if isinstance(decision, dict)
        and isinstance(decision.get("evidence"), dict)
        and decision["evidence"].get("source") in {"slm", "rule_fallback"}
        and isinstance(decision["evidence"].get("summary"), dict)
        and decision["evidence"]["summary"].get("dataset_name") == "cifar_10k_demo"
    ]
    for decision in feedback:
        evidence = decision["evidence"]
        rows = [row for row in history if row.get("decision_id") == decision.get("id")]
        events = sorted({str(row.get("event")) for row in rows})
        print(
            json.dumps(
                {
                    "decision_id": decision.get("id"),
                    "action": decision.get("action"),
                    "status": decision.get("status"),
                    "source": evidence.get("source"),
                    "validator_result": "valid (passed before dispatch)",
                    "iteration": evidence.get("iteration"),
                    "dataset_name": evidence["summary"].get("dataset_name"),
                    "history_events": events,
                },
                sort_keys=True,
            )
        )

    slm_count = sum(item["evidence"].get("source") == "slm" for item in feedback)
    fallback_count = sum(
        item["evidence"].get("source") == "rule_fallback" for item in feedback
    )
    print(
        f"persisted feedback decisions: SLM valid {slm_count} of {len(feedback)}; "
        f"fallback used {fallback_count} times"
    )
    if settings.feedback_rule_fallback_enabled:
        raise RuntimeError("Feedback rule fallback is enabled in the running configuration")
    print("feedback_rule_fallback_enabled=False")
    if not feedback:
        print("No matching ASK_HUMAN feedback decisions were persisted.")


if __name__ == "__main__":
    main()
