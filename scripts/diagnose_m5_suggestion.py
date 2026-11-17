"""Reproduce one read-only feedback interpretation and report validation facts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402
from app.routers.slm import _feedback_prompt, _validate_feedback_suggestion  # noqa: E402
from app.services.feedback import build_training_feedback_summary  # noqa: E402


GROUP_ID = "bidirectional_ring-1793174593-1a4d43"


class DiagnosisError(RuntimeError):
    pass


def _request(
    url: str,
    *,
    token: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 180,
) -> Any:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    data = None
    method = "GET"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise DiagnosisError(f"Request returned HTTP {error.code}") from error
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiagnosisError(f"Request failed: {type(error).__name__}") from error


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        redacted = re.sub(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*",
            "Bearer [REDACTED]",
            redacted,
        )
        return redacted
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    return value


def main() -> None:
    settings = Settings.from_env()
    results = _request(
        f"http://127.0.0.1:{settings.port}/api/slm/results",
        token=settings.api_token,
    )
    groups = results.get("groups")
    if not isinstance(groups, list):
        raise DiagnosisError("Backend results did not contain a groups list")
    group = next(
        (
            item
            for item in groups
            if isinstance(item, dict) and item.get("group_id") == GROUP_ID
        ),
        None,
    )
    if group is None:
        raise DiagnosisError(f"Completed group {GROUP_ID} is not available")
    summary = build_training_feedback_summary(
        group,
        iteration=1,
        previous_accuracy=None,
    )
    prompt = _feedback_prompt(summary)
    raw_output = _request(
        f"{settings.master_slm_url}/interpret",
        token=settings.master_slm_token,
        payload={"message": prompt},
        timeout=settings.slm_timeout_seconds,
    )
    tokens = (
        settings.api_token,
        settings.master_slm_token,
        *(node.token for node in settings.nodes),
    )
    safe_output = _redact(raw_output, tokens)
    safe_prompt = _redact(prompt, tokens)
    _, _, reason = _validate_feedback_suggestion(raw_output, summary)
    print("prompt_sent_to_slm:")
    print(safe_prompt)
    print("raw_model_output:")
    print(json.dumps(safe_output, sort_keys=True, indent=2))
    print(f"validator_result: {reason or 'valid'}")


if __name__ == "__main__":
    main()
