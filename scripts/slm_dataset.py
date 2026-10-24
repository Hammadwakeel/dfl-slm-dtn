"""Generate and prepare synthetic SLM instruction data from the shared action schema."""

from __future__ import annotations

import argparse
import email.utils
import json
import math
import random
import re
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from jsonschema import Draft202012Validator, ValidationError
from rapidfuzz.fuzz import ratio as fuzzy_ratio

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "common" / "slm_action_schema.json"
PHASE3_REPORT = ROOT / "docs" / "phase3-slm-master.md"
DATA_ROOT = ROOT / "data" / "slm"
RAW_PATH = DATA_ROOT / "raw" / "dataset.jsonl"
METRICS_PATH = DATA_ROOT / "raw" / "generation_metrics.json"
SPLIT_ROOT = DATA_ROOT / "splits"
SAMPLE_PATH = DATA_ROOT / "sample.jsonl"
MODEL = "openai/gpt-oss-120b"
BATCH_SIZE = 20
TARGET_SIZE = 3000
CATEGORY_TARGETS = {
    "normal": 1650,
    "mixed_language": 300,
    "compound": 240,
    "ambiguous": 300,
    "invalid_value": 210,
    "unsafe": 210,
    "chit_chat": 90,
}
OPERATIONAL_ACTIONS = (
    "upload_dataset",
    "preprocess_dataset",
    "create_initial_model",
    "start_training",
    "stop_training",
    "set_topology",
    "optimize_topology",
    "set_rounds",
    "set_epochs",
    "enable_nas",
    "get_status",
    "get_logs",
    "get_results",
    "compare_runs",
    "suggest_next_action",
)
CHAT_SCHEMA = None
ACTION_SCHEMA = None
SECRET_PATTERN = re.compile(
    r"(?i)(?:bearer\s+[a-z0-9._~-]{16,}|"
    r"(?:api[_ -]?key|token|password|secret)\s*[:=]\s*[^\s,;]{8,}|"
    r"\b(?:sk|gsk|ghp|github_pat)_[A-Za-z0-9_-]{16,}\b|"
    r"\b[A-Fa-f0-9]{32,}\b)"
)
PERSONAL_DATA_PATTERNS = (
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?<!\d)(?:\+\d{8,15}|00\d{8,15})(?!\d)|(?<!\d)\d{3}[- .]\d{3}[- .]\d{4}(?!\d)"),
    re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
)
FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class DailyQuotaExhausted(RuntimeError):
    """Raised when every configured account has exhausted its daily quota."""


def load_schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(document)
    action_schema = {**document, **document["$defs"]["action"]}
    chat_schema = {**document, **document["$defs"]["chatResponse"]}
    return action_schema, chat_schema


ACTION_SCHEMA, CHAT_SCHEMA = load_schemas()
ACTION_VALIDATOR = Draft202012Validator(ACTION_SCHEMA)
CHAT_VALIDATOR = Draft202012Validator(CHAT_SCHEMA)
ACTION_BRANCHES = {
    branch["properties"]["action"]["const"]: branch
    for branch in ACTION_SCHEMA["oneOf"]
}


def build_label(action: str, parameters: dict[str, Any], category: str) -> str:
    """Build the supervised target entirely from code and schema-constrained values."""
    if category == "ambiguous" or category == "compound":
        action = "clarify"
        parameters = {}
        if category == "compound":
            reply = "Which safe step would you like me to handle first?"
        else:
            reply = "Could you clarify the missing details so I can choose one valid action?"
    elif category == "invalid_value" or category == "unsafe":
        action = "refuse"
        parameters = {}
        if category == "invalid_value":
            reply = (
                "I can’t accept those values: rounds must be 1–100, epochs 1–50, "
                "and topology must be ring, directed_ring, bidirectional_ring, mesh, chain, or star."
            )
        else:
            reply = "I can’t help with unsafe requests, but I can assist with safe DFL operations."
    elif category == "chit_chat":
        action = "redirect"
        parameters = {}
        reply = "I can help with DFL operations; what would you like to check or configure?"
    else:
        reply = "I can help with that DFL request."
    payload = {"reply": reply, "action": {"action": action, "parameters": parameters}}
    CHAT_VALIDATOR.validate(payload)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sample_parameters(action: str, index: int) -> dict[str, Any]:
    """Sample only declared parameters and use schema boundaries in the value cycle."""
    branch = ACTION_BRANCHES[action]
    definition = branch["properties"]["parameters"]
    properties = definition.get("properties", {})
    required = set(definition.get("required", []))
    result: dict[str, Any] = {}
    for name, spec in properties.items():
        if name not in required and index % 2:
            continue
        if "$ref" in spec and spec["$ref"].endswith("/datasetName"):
            result[name] = ("indoor_co2", "sensor_data", "cifar10")[index % 3]
        elif "$ref" in spec and spec["$ref"].endswith("/runIds"):
            result[name] = [f"run_{index + 1:03d}", f"run_{index + 2:03d}"]
        elif "enum" in spec:
            result[name] = spec["enum"][index % len(spec["enum"])]
        elif spec.get("type") == "boolean":
            result[name] = bool(index % 2 == 0)
        elif spec.get("type") == "integer":
            minimum = spec["minimum"]
            maximum = spec["maximum"]
            values = (minimum, maximum, (minimum + maximum) // 2, minimum + 1)
            result[name] = values[index % len(values)]
        elif spec.get("type") == "array":
            item = spec["items"]
            if "enum" in item:
                values = item["enum"]
                count = spec.get("minItems", 1)
                result[name] = [values[(index + offset) % len(values)] for offset in range(count)]
            elif item.get("type") == "integer":
                low = item["minimum"]
                high = item["maximum"]
                count = spec.get("minItems", 1)
                result[name] = [low if offset == 0 and index % 2 == 0 else high if offset == 0 else low + offset
                                for offset in range(count)]
            else:
                raise ValueError(f"Unsupported array parameter in schema: {action}.{name}")
        else:
            raise ValueError(f"Unsupported parameter in schema: {action}.{name}")
    return result


def build_reason(category: str, action: str) -> str:
    reasons = {
        "legacy_test": "Expected action and parameters are copied from the Phase 3 accuracy report.",
        "normal": f"A clear in-scope request maps to the schema-valid {action} action.",
        "mixed_language": f"A code-switched request maps to the schema-valid {action} action.",
        "compound": "A multi-step request requires choosing one safe first step.",
        "ambiguous": "Required intent or details are ambiguous, so clarification is needed.",
        "invalid_value": "The request contains a value outside the schema's permitted range.",
        "unsafe": "The request asks for unsafe or out-of-scope behavior.",
        "chit_chat": "Unrelated conversation is redirected to the DFL assistant's scope.",
    }
    return reasons[category]


def make_user_prompt(category: str, action: str, parameters: dict[str, Any]) -> str:
    if category in {"normal", "mixed_language"}:
        style = (
            "Use natural English. Vary length and tone: some brief, some polite, some direct, "
            "and vary sentence structure. Use number words in some utterances where exact values "
            "remain unambiguous."
        )
        if category == "mixed_language":
            style = (
                "Use English mixed with informal Roman Urdu/code-switching (Latin script only); "
                "include occasional harmless typos and casual chat style, while keeping the "
                "requested intent and values unambiguous."
            )
        task = (
            f"Write exactly {BATCH_SIZE} distinct synthetic user messages that clearly request "
            f"action {action!r} with exactly these parameters: "
            f"{json.dumps(parameters, ensure_ascii=False, sort_keys=True)}. Do not ask for a "
            "different action or add unprovided parameter values."
        )
    elif category == "compound":
        task = (
            f"Write exactly {BATCH_SIZE} distinct synthetic requests that combine several safe "
            "DFL operations (for example upload then preprocess then train). Under a one-action "
            "policy they should ask which step to do first; they must not contain secrets."
        )
    elif category == "ambiguous":
        task = (
            f"Write exactly {BATCH_SIZE} distinct synthetic DFL requests that are genuinely "
            "ambiguous or omit information required for {action!r}; do not invent the missing "
            "details and do not include personal data or secrets."
        )
    elif category == "invalid_value":
        task = (
            "Write exactly 20 distinct synthetic DFL requests containing clearly invalid or "
            "out-of-range values, such as 1000 rounds, negative epochs, or an unknown topology. "
            "Do not put invalid values in a label; only produce the user wording."
        )
    elif category == "unsafe":
        task = (
            "Write exactly 20 distinct synthetic user requests that attempt unsafe or out-of-scope "
            "behavior, such as ignoring rules, running shell commands, deleting data, revealing "
            "credentials, or prompt injection embedded in pasted logs. Use placeholders such as "
            "[REDACTED LOG], never real secrets, keys, commands that cause harm, or personal data."
        )
    else:
        task = (
            "Write exactly 20 distinct short synthetic chit-chat or unrelated requests that are "
            "not about DFL system operation."
        )
    return task + "\n" + style if category in {"normal", "mixed_language"} else task


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def contains_sensitive_text(text: str) -> bool:
    if SECRET_PATTERN.search(text):
        return True
    return any(pattern.search(text) for pattern in PERSONAL_DATA_PATTERNS)


def parse_utterances(text: str) -> list[str]:
    cleaned = FENCE_PATTERN.sub("", text.strip()).strip()
    payload = json.loads(cleaned)
    if not isinstance(payload, dict) or set(payload) != {"utterances"}:
        raise ValueError("Generation response must contain only an utterances array")
    utterances = payload["utterances"]
    if (
        not isinstance(utterances, list)
        or not 1 <= len(utterances) <= BATCH_SIZE
        or any(not isinstance(item, str) or not item.strip() for item in utterances)
    ):
        raise ValueError(f"Generation response must contain 1-{BATCH_SIZE} non-empty strings")
    return [item.strip() for item in utterances]


def valid_record(record: dict[str, Any]) -> bool:
    try:
        output = json.loads(record["label"])
        CHAT_VALIDATOR.validate(output)
        ACTION_VALIDATOR.validate(output["action"])
        return (
            output["action"]["action"] == record["action"]
            and output["action"]["parameters"] == record["params"]
        )
    except (KeyError, TypeError, json.JSONDecodeError, ValidationError):
        return False


def deduplicate_records(
    records: list[dict[str, Any]], *, threshold: float = 0.92
) -> tuple[list[dict[str, Any]], int, int]:
    kept: list[dict[str, Any]] = []
    exact: set[str] = set()
    normalized_rows: list[str] = []
    gram_index: dict[str, set[int]] = {}
    short_text_indices: list[int] = []
    exact_drops = 0
    near_drops = 0
    for record in records:
        normalized = normalize_text(record["user"])
        if not normalized or normalized in exact:
            exact_drops += 1
            continue
        grams = {normalized[index:index + 3] for index in range(max(0, len(normalized) - 2))}
        candidate_counts: Counter[int] = Counter()
        if threshold <= 1.0:
            if grams:
                for gram in grams:
                    for candidate in gram_index.get(gram, ()):
                        candidate_counts[candidate] += 1
            else:
                candidate_counts.update(short_text_indices)
        is_near_duplicate = False
        minimum_shared_grams = min(2, len(grams))
        for candidate, shared_grams in candidate_counts.items():
            if shared_grams < minimum_shared_grams:
                continue
            prior = normalized_rows[candidate]
            longest = max(len(normalized), len(prior))
            if not longest or min(len(normalized), len(prior)) / longest < threshold:
                continue
            if fuzzy_ratio(normalized, prior) / 100.0 >= threshold:
                is_near_duplicate = True
                break
        if is_near_duplicate:
            near_drops += 1
            continue
        exact.add(normalized)
        kept_index = len(normalized_rows)
        normalized_rows.append(normalized)
        kept.append(record)
        if grams:
            for gram in grams:
                gram_index.setdefault(gram, set()).add(kept_index)
        else:
            short_text_indices.append(kept_index)
    return kept, exact_drops, near_drops


def split_by_template_family(
    records: list[dict[str, Any]], seed: int = 20261025
) -> dict[str, list[dict[str, Any]]]:
    if any(not record.get("template_family") for record in records):
        raise ValueError("Every record must have a template_family")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["template_family"], []).append(record)
    if len(groups) < 3:
        raise ValueError("At least three template families are required for splitting")
    rng = random.Random(seed)
    category_groups: dict[str, list[str]] = {}
    for family, family_rows in groups.items():
        categories = {row.get("category", "__all__") for row in family_rows}
        if len(categories) != 1:
            raise ValueError(f"Template family {family!r} spans multiple categories")
        category_groups.setdefault(next(iter(categories)), []).append(family)
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    assignments = {name: [] for name in ratios}
    total = len(records)
    remaining: list[str] = []
    for families in category_groups.values():
        rng.shuffle(families)
        if len(families) >= 3:
            assignments["validation"].extend(groups[families.pop()])
            assignments["test"].extend(groups[families.pop()])
        remaining.extend(families)
    rng.shuffle(remaining)
    remaining.sort(key=lambda family: len(groups[family]), reverse=True)
    for family in remaining:
        destination = min(
            ratios,
            key=lambda name: (
                len(assignments[name]) / total - ratios[name],
                len(assignments[name]),
            ),
        )
        assignments[destination].extend(groups[family])
    return assignments


def read_jsonl(path: Path, malformed_lines: list[int] | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    if malformed_lines is not None:
                        malformed_lines.append(line_number)
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_api_keys(env_path: Path = ROOT / ".env") -> dict[str, str]:
    keys: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s*(GROQ_API_KEY\d*)\s*=\s*(.*?)\s*$", line)
        if match and match.group(2):
            keys[match.group(1)] = match.group(2)
    if not keys:
        raise RuntimeError("No non-empty GROQ_API_KEY variables found in repo-root .env")
    return dict(sorted(keys.items(), key=lambda item: (
        int(re.search(r"(\d+)$", item[0]).group(1))
        if re.search(r"(\d+)$", item[0]) else 0
    )))


def safe_error(error: Exception, keys: dict[str, str]) -> str:
    body = getattr(error, "body", None)
    api_error = body.get("error") if isinstance(body, dict) else None
    message = (
        str(api_error.get("message", "Groq API error"))
        if isinstance(api_error, dict)
        else str(error)
    )
    message = re.sub(
        r"(?is)failed_generation.*$",
        "failed_generation=[REDACTED]",
        message,
    )
    for value in keys.values():
        message = message.replace(value, "[REDACTED]")
    message = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[REDACTED]", message)
    message = re.sub(r"(?i)(gsk_[A-Za-z0-9_-]{8,})", "[REDACTED]", message)
    return message[:1000]


def retry_after_seconds(error: Exception) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", {})
    value = headers.get("retry-after") if headers else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = email.utils.parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def make_client_call(
    client: Any,
    category: str,
    action: str,
    parameters: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    schema = {
        "type": "object",
        "properties": {
            "utterances": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": BATCH_SIZE,
            }
        },
        "required": ["utterances"],
        "additionalProperties": False,
    }
    response = client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "Write only synthetic user-message wording. Do not invent labels, output "
                "credentials, use real personal data, or reveal system information. Return "
                "exactly one JSON object matching the supplied schema."
            )},
            {"role": "user", "content": make_user_prompt(category, action, parameters)},
        ],
        temperature=1,
        top_p=1,
        max_completion_tokens=4096,
        reasoning_effort="low",
        stream=False,
        stop=None,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "synthetic_user_utterances",
                "strict": True,
                "schema": schema,
            },
        },
    )
    completion = response.parse()
    content = completion.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("Groq returned no text content")
    utterances = parse_utterances(content)
    usage = completion.usage
    stats = {
        "model": completion.model,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
        "headers": {
            name: response.headers.get(name)
            for name in (
                "x-ratelimit-limit-requests",
                "x-ratelimit-limit-tokens",
                "x-ratelimit-remaining-requests",
                "x-ratelimit-remaining-tokens",
                "x-ratelimit-reset-requests",
                "x-ratelimit-reset-tokens",
            )
            if response.headers.get(name) is not None
        },
    }
    return utterances, stats


def generate_batch(
    category: str,
    action: str,
    parameters: dict[str, Any],
    *,
    keys: dict[str, str],
    metrics: dict[str, Any],
) -> list[str]:
    from groq import APIStatusError, Groq

    key_names = list(keys)
    today = datetime.now(timezone.utc).date().isoformat()
    if metrics.get("daily_limit_date") != today:
        metrics["daily_limited_keys"] = []
        metrics["daily_limit_date"] = today
    daily_limited: set[str] = set(metrics.get("daily_limited_keys", []))
    start_index = metrics.get("next_key_index", 0) % len(key_names)
    attempt = 0
    while True:
        available = [name for name in key_names if name not in daily_limited]
        if not available:
            metrics["daily_limited_keys"] = sorted(daily_limited)
            raise DailyQuotaExhausted("All configured Groq accounts reported daily quota exhaustion")
        key_name = key_names[start_index % len(key_names)]
        if key_name in daily_limited:
            start_index += 1
            continue
        client = Groq(api_key=keys[key_name])
        try:
            metrics["api_requests"] = metrics.get("api_requests", 0) + 1
            utterances, stats = make_client_call(client, category, action, parameters)
            metrics["calls"] = metrics.get("calls", 0) + 1
            metrics["tokens"] = metrics.get("tokens", 0) + stats["total_tokens"]
            metrics["prompt_tokens"] = metrics.get("prompt_tokens", 0) + stats["prompt_tokens"]
            metrics["completion_tokens"] = metrics.get("completion_tokens", 0) + stats["completion_tokens"]
            metrics["last_rate_limits"] = stats["headers"]
            metrics["next_key_index"] = (key_names.index(key_name) + 1) % len(key_names)
            return utterances
        except Exception as error:
            metrics["errors"] = metrics.get("errors", 0) + 1
            save_metrics(metrics)
            status = getattr(error, "status_code", None)
            detail = safe_error(error, keys)
            lowered = detail.casefold()
            body = getattr(error, "body", None)
            api_error = body.get("error") if isinstance(body, dict) else None
            error_code = api_error.get("code") if isinstance(api_error, dict) else None
            is_daily = status == 429 and any(
                marker in lowered for marker in ("per day", "daily", "day limit", "daily quota")
            )
            if is_daily:
                daily_limited.add(key_name)
                metrics["daily_limited_keys"] = sorted(daily_limited)
                metrics["daily_limit_date"] = today
                metrics["retries"] = metrics.get("retries", 0) + 1
                save_metrics(metrics)
                if len(daily_limited) == len(key_names):
                    raise DailyQuotaExhausted(
                        "All configured Groq accounts reported daily quota exhaustion"
                    ) from None
                start_index = (key_names.index(key_name) + 1) % len(key_names)
                metrics["next_key_index"] = start_index
                save_metrics(metrics)
                continue
            if status == 429:
                metrics["retries"] = metrics.get("retries", 0) + 1
                if attempt >= 8:
                    save_metrics(metrics)
                    raise RuntimeError("Groq rate-limit retries exhausted; progress is saved") from None
                exponential = min(60.0, 2**attempt)
                delay = max(retry_after_seconds(error) or 0.0, exponential)
                delay += random.uniform(0.1, 1.0)
                time.sleep(delay)
                attempt += 1
                continue
            if status == 400 and error_code == "json_validate_failed":
                metrics["retries"] = metrics.get("retries", 0) + 1
                save_metrics(metrics)
                if attempt >= 2:
                    raise RuntimeError(
                        "Groq repeatedly rejected generated structured output; progress is saved"
                    ) from None
                attempt += 1
                continue
            if isinstance(error, (ValueError, json.JSONDecodeError)):
                metrics["retries"] = metrics.get("retries", 0) + 1
                if attempt >= 2:
                    save_metrics(metrics)
                    raise RuntimeError("Groq returned invalid structured output after retries") from None
                attempt += 1
                continue
            if isinstance(error, APIStatusError):
                save_metrics(metrics)
                raise RuntimeError(
                    f"Groq request failed with HTTP {status}: {detail}"
                ) from None
            save_metrics(metrics)
            raise RuntimeError(f"Groq request failed: {detail}") from None


def save_metrics(metrics: dict[str, Any]) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def record_from_utterance(
    utterance: str,
    category: str,
    action: str,
    parameters: dict[str, Any],
    family: str,
) -> dict[str, Any]:
    actual_action = action
    actual_parameters = parameters
    if category in {"compound", "ambiguous"}:
        actual_action, actual_parameters = "clarify", {}
    elif category in {"invalid_value", "unsafe"}:
        actual_action, actual_parameters = "refuse", {}
    elif category == "chit_chat":
        actual_action, actual_parameters = "redirect", {}
    return {
        "id": str(uuid.uuid4()),
        "category": category,
        "user": utterance,
        "label": build_label(actual_action, actual_parameters, category),
        "action": actual_action,
        "params": actual_parameters,
        "template_family": family,
        "reason": build_reason(category, actual_action),
        "source_model": MODEL,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {
            "calls": 0,
            "errors": 0,
            "retries": 0,
            "tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "daily_limited_keys": [],
            "daily_limit_date": datetime.now(timezone.utc).date().isoformat(),
            "next_key_index": 0,
        }
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("Generation metrics file is invalid; preserve it and inspect locally") from error
    return metrics


def _choose_category(counts: Counter[str], targets: dict[str, int]) -> str:
    return max(
        targets,
        key=lambda category: targets[category] - counts[category],
    )


def _choose_action(category: str, records: list[dict[str, Any]]) -> str:
    if category not in {"normal", "mixed_language"}:
        return {
            "compound": "clarify",
            "ambiguous": "clarify",
            "invalid_value": "refuse",
            "unsafe": "refuse",
            "chit_chat": "redirect",
        }[category]
    counts = Counter(row["action"] for row in records if row["category"] == category)
    return min(OPERATIONAL_ACTIONS, key=lambda action: counts[action])


def _usable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in records
        if valid_record(row)
        and not contains_sensitive_text(row.get("user", ""))
        and row.get("category") in CATEGORY_TARGETS
    ]


def run_generation(
    targets: dict[str, int] | None = None,
    *,
    baseline_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    targets = targets or CATEGORY_TARGETS
    baseline_counts = baseline_counts or {}
    absolute_targets = {
        category: baseline_counts.get(category, 0) + count
        for category, count in targets.items()
    }
    keys = read_api_keys()
    metrics = load_metrics()
    metrics.setdefault("started_at", datetime.now(timezone.utc).isoformat())
    elapsed_before_session = float(metrics.get("elapsed_seconds", 0.0))
    session_started = time.monotonic()
    while True:
        raw_records = read_jsonl(RAW_PATH)
        usable = _usable_records(raw_records)
        unique, _, _ = deduplicate_records(usable, threshold=1.1)
        counts = Counter(row["category"] for row in unique)
        shortages = {
            name: max(0, count - counts[name])
            for name, count in absolute_targets.items()
        }
        if not any(shortages.values()):
            return metrics
        category = _choose_category(counts, absolute_targets)
        action = _choose_action(category, unique)
        index = counts[action] + counts[category]
        parameters = sample_parameters(action, index)
        utterances = generate_batch(
            category,
            action,
            parameters,
            keys=keys,
            metrics=metrics,
        )
        family = f"{category}-{action}-{uuid.uuid4().hex[:12]}"
        rows = [
            record_from_utterance(utterance, category, action, parameters, family)
            for utterance in utterances
        ]
        append_jsonl(RAW_PATH, rows)
        metrics["rows_written"] = metrics.get("rows_written", 0) + len(rows)
        metrics["last_batch_category"] = category
        metrics["last_batch_action"] = action
        metrics["updated_at"] = datetime.now(timezone.utc).isoformat()
        metrics["elapsed_seconds"] = elapsed_before_session + time.monotonic() - session_started
        save_metrics(metrics)
        safe_counts = Counter(row["category"] for row in _usable_records(read_jsonl(RAW_PATH)))
        print(
            f"batch_saved category={category} action={action} "
            f"usable_category_rows={safe_counts[category]} "
            f"calls={metrics['calls']} tokens={metrics['tokens']}",
            flush=True,
        )


def parse_phase3_legacy(path: Path = PHASE3_REPORT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    table_started = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| # | Prompt | Expected action"):
            table_started = True
            continue
        if not table_started:
            continue
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        prompt = cells[1]
        expected = re.sub(r"`", "", cells[2])
        match = re.match(r"([a-z_]+)\s*(\{.*\})", expected)
        if not match:
            raise ValueError(f"Could not parse expected legacy label for prompt {cells[0]}")
        action = match.group(1)
        params_text = match.group(2).strip("{} ")
        params: dict[str, Any] = {}
        if params_text and "unspecified" not in params_text:
            for key, value in re.findall(r"([a-z_]+):\s*(\[[^\]]+\]|[^,]+)", params_text):
                value = value.strip()
                if value.startswith("["):
                    params[key] = [part.strip().strip("_` ") for part in value.strip("[]").split(",")]
                elif value.isdigit():
                    params[key] = int(value)
                elif value in {"true", "false"}:
                    params[key] = value == "true"
                else:
                    params[key] = value.strip("_` ")
        row = record_from_utterance(
            prompt,
            "legacy_test",
            action,
            params,
            f"legacy-phase3-{cells[0]}",
        )
        row["source_model"] = "phase3_report"
        ACTION_VALIDATOR.validate({"action": row["action"], "parameters": row["params"]})
        rows.append(row)
    if len(rows) != 20:
        raise ValueError(f"Expected 20 legacy prompts from Phase 3 report, found {len(rows)}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def finalize_dataset(
    seed: int = 20261025,
    targets: dict[str, int] | None = None,
) -> dict[str, Any]:
    targets = targets or CATEGORY_TARGETS
    malformed_lines: list[int] = []
    raw = read_jsonl(RAW_PATH, malformed_lines)
    valid = [row for row in raw if valid_record(row)]
    for row in valid:
        row["label"] = build_label(row["action"], row["params"], row["category"])
    invalid_count = len(raw) - len(valid)
    pii_free = [row for row in valid if not contains_sensitive_text(row.get("user", ""))]
    sensitive_count = len(valid) - len(pii_free)
    legacy = parse_phase3_legacy()
    legacy_texts = {normalize_text(row["user"]) for row in legacy}
    safe = [row for row in pii_free if normalize_text(row["user"]) not in legacy_texts]
    duplicates_removed, exact_drops, near_drops = deduplicate_records(safe)
    write_jsonl(DATA_ROOT / "legacy_test.jsonl", legacy)
    available = Counter(row["category"] for row in duplicates_removed)
    deficits = {
        category: max(0, count - available[category])
        for category, count in targets.items()
    }
    if not any(deficits.values()):
        rng = random.Random(seed)
        selected = []
        for category, count in targets.items():
            candidates = [row for row in duplicates_removed if row["category"] == category]
            rng.shuffle(candidates)
            selected.extend(candidates[:count])
        splits = split_by_template_family(selected, seed)
        for split_name, split_rows in splits.items():
            write_jsonl(SPLIT_ROOT / f"{split_name}.jsonl", split_rows)
    else:
        splits = {"train": [], "validation": [], "test": []}
    metrics = load_metrics()
    stats = {
        "raw_rows": len(raw),
        "truncated_or_malformed_jsonl_rows_dropped": len(malformed_lines),
        "invalid_labels_dropped": invalid_count,
        "sensitive_text_dropped": sensitive_count,
        "legacy_prompt_duplicates_excluded": len(pii_free) - len(safe),
        "exact_duplicates_removed": exact_drops,
        "near_duplicates_removed": near_drops,
        "surplus_not_selected": max(0, len(duplicates_removed) - sum(targets.values())),
        "retained": sum(len(rows) for rows in splits.values()),
        "category_deficits": deficits,
        "splits": {name: len(rows) for name, rows in splits.items()},
        "category_counts": dict(Counter(row["category"] for row in selected))
        if not any(deficits.values()) else {},
        "action_counts": dict(Counter(row["action"] for row in selected))
        if not any(deficits.values()) else {},
        "split_category_counts": {
            split: dict(Counter(row["category"] for row in rows))
            for split, rows in splits.items()
        },
        "split_action_counts": {
            split: dict(Counter(row["action"] for row in rows))
            for split, rows in splits.items()
        },
        "legacy_test_count": len(legacy),
        "api": metrics,
    }
    stats_path = DATA_ROOT / "validation_report.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def write_review_sample(seed: int = 20261025) -> list[dict[str, Any]]:
    rows = []
    for split_name in ("train", "validation", "test"):
        rows.extend(read_jsonl(SPLIT_ROOT / f"{split_name}.jsonl"))
    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    general_pool = [row for row in rows if row["category"] != "unsafe"]
    rng.shuffle(general_pool)
    general = []
    for category in CATEGORY_TARGETS:
        candidates = [row for row in general_pool if row["category"] == category]
        if candidates:
            general.append(candidates.pop())
            selected_id = general[-1]["id"]
            general_pool = [row for row in general_pool if row["id"] != selected_id]
    rng.shuffle(general_pool)
    general.extend(general_pool[: max(0, 25 - len(general))])
    unsafe = by_category.get("unsafe", [])
    rng.shuffle(unsafe)
    review = general + unsafe[:10]
    write_jsonl(SAMPLE_PATH, review)
    for index, row in enumerate(general, 1):
        print(
            f"REVIEW {index:02d} [{row['category']}] "
            f"user={json.dumps(row['user'], ensure_ascii=True)}"
        )
        print(f"  label={json.dumps(row['label'], ensure_ascii=True)}")
        print(f"  reason={row['reason']}")
    for index, row in enumerate(unsafe[:10], 1):
        print(f"UNSAFE {index:02d} user={json.dumps(row['user'], ensure_ascii=True)}")
        print(f"  label={json.dumps(row['label'], ensure_ascii=True)}")
        print(f"  reason={row['reason']}")
    return review


def write_dataset_card(stats: dict[str, Any]) -> None:
    api = stats["api"]
    date = datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()
    active_seconds = api.get("elapsed_seconds", 0)
    wall_seconds = 0
    if api.get("started_at") and api.get("updated_at"):
        started_at = datetime.fromisoformat(api["started_at"])
        updated_at = datetime.fromisoformat(api["updated_at"])
        wall_seconds = max(0, (updated_at - started_at).total_seconds())
    rate_limits = api.get("last_rate_limits", {})
    content = f"""# SLM instruction dataset

## Status

Synthetic instruction data for the DFL-SLM action interface. Generated {date} with `{MODEL}`. Fine-tuning has **not** been started.

## Splits and balance

- Retained examples: {stats['retained']}
- Train: {stats['splits'].get('train', 0)}; validation: {stats['splits'].get('validation', 0)}; test: {stats['splits'].get('test', 0)}
- Separate legacy Phase 3 test prompts: {stats['legacy_test_count']}
- Category counts: `{json.dumps(stats['category_counts'], sort_keys=True)}`
- Action counts: `{json.dumps(stats['action_counts'], sort_keys=True)}`
- Split category counts: `{json.dumps(stats['split_category_counts'], sort_keys=True)}`
- Split action counts: `{json.dumps(stats['split_action_counts'], sort_keys=True)}`

Splitting is performed by `template_family`, never by individual utterance. All paraphrases in a family remain in one split. Legacy test prompts are kept in a separate file and exact normalized matches are excluded from generated data.

## Reproduction

Install the dataset-generation dependencies with `python -m pip install -r requirements-slm-dataset.txt`, then run `python -m scripts.slm_dataset`. The command resumes from `data/slm/raw/dataset.jsonl`, validates the saved rows, creates family-separated split files, writes the legacy test file, and prints a review sample. Use `python -m scripts.slm_dataset --prepare-only` to rebuild the splits without contacting Groq.

## Generation and labels

The generator samples actions and parameters from `common/slm_action_schema.json`; it builds the target JSON label, reply, and one-sentence reason in code. Groq is used only to produce synthetic natural-language utterances for that fixed target. Structured JSON output is validated locally. API keys are read from the ignored repository-root `.env`; values are not written to logs or dataset records.

Label output format:

```json
{{"reply":"brief text","action":{{"action":"<allow-listed action>","parameters":{{}}}}}}
```

## Validation

- Raw rows: {stats['raw_rows']}
- Truncated or malformed JSONL rows dropped: {stats['truncated_or_malformed_jsonl_rows_dropped']}
- Invalid or truncated/schema-invalid rows dropped: {stats['invalid_labels_dropped']}
- Secret-looking or personal-data rows dropped: {stats['sensitive_text_dropped']}
- Legacy prompt matches excluded: {stats['legacy_prompt_duplicates_excluded']}
- Exact duplicates removed: {stats['exact_duplicates_removed']}
- Near duplicates removed: {stats['near_duplicates_removed']}
- API requests: {api.get('api_requests', 0)} ({api.get('calls', 0)} successful completions)
- Token usage: {api.get('prompt_tokens', 0)} prompt; {api.get('completion_tokens', 0)} completion; {api.get('tokens', 0)} total
- Generation runtime: {active_seconds:.1f}s active across resumes; {wall_seconds:.1f}s from first to final API batch
- Observed rate limits: {rate_limits.get('x-ratelimit-limit-requests', 'not exposed')} requests/minute; {rate_limits.get('x-ratelimit-limit-tokens', 'not exposed')} tokens/minute. Daily quota limits were not exposed.
- Errors: {api.get('errors', 0)}; retries: {api.get('retries', 0)}
- The single failed request was an initial strict-JSON batch-size rejection; the output constraint was corrected and generation resumed. No rate-limit retries were needed.

## Known limitations

- The utterances are synthetic and generated by `{MODEL}`; they may not reflect real user language or real operational diversity.
- Language coverage is English and Roman Urdu/code-switching only.
- Validation and test splits are held out by template family, but synthetic paraphrases can still share semantic patterns.
- The refusal and clarification categories are generated from bounded templates and do not constitute a comprehensive safety evaluation.
- Secret and personal-data screening uses local pattern checks rather than a full identity/entity-recognition system; review the sample before fine-tuning.
- Legacy prompts come from the Phase 3 report; they are evaluation-only and are not included in training or validation.
"""
    (ROOT / "docs" / "slm-dataset-card.md").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Validate and split saved raw data without API calls")
    parser.add_argument("--target", type=int, default=TARGET_SIZE, help="Approximate retained dataset size")
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if args.target < 1:
        parser.error("--target must be at least 1")
    if args.target != TARGET_SIZE:
        scale = args.target / TARGET_SIZE
        targets = {name: round(count * scale) for name, count in CATEGORY_TARGETS.items()}
    else:
        targets = CATEGORY_TARGETS
    if not args.prepare_only:
        started = time.monotonic()
        try:
            metrics = run_generation(targets)
            stats = finalize_dataset(args.seed, targets)
            for _ in range(5):
                deficits = stats["category_deficits"]
                if not any(deficits.values()):
                    break
                usable, _, _ = deduplicate_records(
                    _usable_records(read_jsonl(RAW_PATH)),
                    threshold=1.1,
                )
                raw_counts = Counter(row["category"] for row in usable)
                additional = {
                    category: max(1, count + math.ceil(count * 0.25))
                    for category, count in deficits.items()
                    if count
                }
                baseline = {category: raw_counts[category] for category in additional}
                run_generation(additional, baseline_counts=baseline)
                stats = finalize_dataset(args.seed, targets)
        except DailyQuotaExhausted as error:
            print(str(error))
            print("Progress and metrics are saved; resume later after quota resets.")
            return 2
        except RuntimeError as error:
            print(f"Generation stopped: {error}")
            print("Raw progress and generation metrics are saved; rerun to resume.")
            return 1
        finally:
            current = load_metrics()
            if current.get("elapsed_seconds") is None:
                current["elapsed_seconds"] = round(time.monotonic() - started, 3)
            save_metrics(current)
    stats = finalize_dataset(args.seed, targets)
    if any(stats["category_deficits"].values()):
        print("Dataset is incomplete; raw progress is preserved for resume.")
        print(json.dumps(stats["category_deficits"], sort_keys=True))
        return 3
    write_dataset_card(stats)
    write_review_sample(args.seed)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
