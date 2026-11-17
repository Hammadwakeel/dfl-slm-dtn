"""Read-only demo of saved metrics and local autonomy state; never dispatches training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
LOCAL_API = "http://127.0.0.1:8000"
LOCAL_GET_PATHS = frozenset({"/api/autonomy/state", "/api/autonomy/history"})
RESULT_FILES = (
    ROOT / "data" / "mnist_final_test_eval.json",
    ROOT / "data" / "m8_centralized_nas_baseline.json",
)
RESULT_REPORT = ROOT / "docs" / "results-report.md"


def _percent_summary(summary: dict, key: str) -> str:
    metric = summary[key]
    mean = float(metric["mean"])
    std = float(metric["sample_std_ddof_1"])
    return f"{mean * 100:.2f}% +/- {std * 100:.2f} pp"


def _show_saved_json(path: Path) -> None:
    if not path.is_file():
        print(f"saved_result=missing ({path.relative_to(ROOT)})")
        return
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"saved_result=unavailable ({path.name}; {type(error).__name__})")
        return

    if path.name == "mnist_final_test_eval.json":
        print(
            "saved_result=MNIST test; rows="
            f"{result['data_integrity']['test_rows']}; federated_accuracy="
            f"{_percent_summary(result['federated_test']['summary'], 'accuracy')}; "
            f"centralized_accuracy={_percent_summary(result['centralized_test']['summary'], 'accuracy')}"
        )
    elif path.name == "m8_centralized_nas_baseline.json":
        print(
            "saved_result=M8 CIFAR-10; test_rows="
            f"{result['phase6_configuration']['test_rows']}; accuracy="
            f"{_percent_summary(result['centralized_test']['summary'], 'accuracy')}"
        )
    else:
        print(f"saved_result=present ({path.relative_to(ROOT)})")


def _show_report_rows() -> None:
    if not RESULT_REPORT.is_file():
        print("saved_report=missing (docs/results-report.md)")
        return
    labels = (
        "CIFAR-10 DFL",
        "CIFAR-10 centralized",
        "CIFAR-10 NAS DFL",
        "M8 centralized NAS",
        "MNIST first hardware",
        "MNIST corrected hardware",
        "MNIST federated PC evaluation",
        "MNIST centralized PC evaluation",
    )
    rows = [
        line.strip()
        for line in RESULT_REPORT.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("|") and any(label in line for label in labels)
    ]
    for row in rows:
        print("saved_report_row=" + row)


def _get_json(path: str, token: str) -> object:
    if path not in LOCAL_GET_PATHS:
        raise ValueError("Demo permits only the local autonomy state/history GET routes")
    request = Request(
        LOCAL_API + path,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(type(error).__name__) from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Invalid JSON response") from error


def _show_local_autonomy() -> None:
    token = os.environ.get("DTN_API_TOKEN", "")
    if not token:
        print("autonomy_get=skipped (DTN_API_TOKEN is not set in this process environment)")
        return
    try:
        state = _get_json("/api/autonomy/state", token)
        if not isinstance(state, dict):
            raise RuntimeError("Unexpected state response")
        loops = state.get("loops", [])
        running_loops = sum(
            isinstance(loop, dict) and loop.get("status") == "RUNNING"
            for loop in loops
        ) if isinstance(loops, list) else "unknown"
        print(
            f"autonomy_state_get=OK; mode={state.get('mode')}; "
            f"STOP={state.get('stop_requested')}; pending={state.get('pending_decisions')}; "
            f"running_loops={running_loops}"
        )
    except RuntimeError as error:
        print(f"autonomy_state_get=unavailable ({error})")
        return

    try:
        history = _get_json("/api/autonomy/history", token)
        if not isinstance(history, list):
            raise RuntimeError("Unexpected history response")
        automatic = [
            row for row in history
            if isinstance(row, dict) and row.get("event") == "executed automatically"
        ]
        automatic.sort(key=lambda row: str(row.get("timestamp", row.get("created_at", ""))))
        for row in automatic[-5:]:
            visible = {
                key: row.get(key)
                for key in ("event", "id", "mode", "action", "result_status", "dataset_name")
                if key in row
            }
            print("autonomy_history_get=" + json.dumps(visible, sort_keys=True))
    except RuntimeError as error:
        print(f"autonomy_history_get=unavailable ({error})")


def main() -> None:
    print("demo=READ_ONLY; network_scope=127.0.0.1:8000; HTTP_methods=GET; cluster_calls=0")
    for path in RESULT_FILES:
        _show_saved_json(path)
    _show_report_rows()
    _show_local_autonomy()


if __name__ == "__main__":
    main()