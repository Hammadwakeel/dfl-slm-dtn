"""Validate ChatGPT-generated JSONL batches against the shared SLM action schema."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_DIR = ROOT / "data" / "chatgpt"
SCHEMA_PATH = ROOT / "common" / "slm_action_schema.json"
CLEAN_DIR = DEFAULT_BATCH_DIR / "clean"
REJECTED_DIR = DEFAULT_BATCH_DIR / "rejected"
JACCARD_THRESHOLD = 0.8
VARIABLE_TOKENS = {
    "number",
    "runid",
    "datasetname",
    "modelname",
    "timestamp",
}
ALLOWED_CATEGORIES = {
    "normal",
    "ambiguous",
    "invalid_value",
    "compound",
    "chit_chat",
    "unsafe",
}
NON_ENTITY_IDENTIFIERS = {
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
    "dataset_name",
    "model_name",
    "dataset_split",
    "search_time_limit_seconds",
    "dirichlet_a01",
    "dirichlet_a03",
    "dirichlet_a05",
    "dirichlet_a07",
    "standard_scaling",
    "my_images",
    "my_data",
    "dfl_master",
    "dfl_worker1",
    "dfl_worker2",
    "dfl_worker3",
}
QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
    }
)
WORD_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE)
TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b",
    re.IGNORECASE,
)
RUN_ID_PATTERN = re.compile(
    r"\brun(?:[_-][a-z0-9][a-z0-9_.-]*|\d[a-z0-9_.-]*)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
TOPOLOGY_SYNONYMS = {
    "ring": ("ring", "ring topology"),
    "directed_ring": ("directed_ring", "directed ring"),
    "bidirectional_ring": ("bidirectional_ring", "bidirectional ring"),
    "mesh": ("mesh", "mesh topology"),
    "chain": ("chain", "chain topology"),
    "star": ("star", "star topology"),
}
_topology_aliases = sorted(
    (
        (alias, topology)
        for topology, aliases in TOPOLOGY_SYNONYMS.items()
        for alias in aliases
    ),
    key=lambda item: len(item[0]),
    reverse=True,
)
TOPOLOGY_MENTION_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(re.escape(alias) for alias, _ in _topology_aliases)
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)
TOPOLOGY_ALIAS_TO_VALUE = {
    alias.casefold(): topology for alias, topology in _topology_aliases
}
INTENT_PATTERNS = {
    "upload_dataset": r"\b(?:upload|add|load|import)\b",
    "preprocess_dataset": r"\b(?:preprocess|normalize|normalise|standardize|scale)\b",
    "create_initial_model": (
        r"\b(?:create|build|initialize|initialise)\b.{0,30}\bmodel\b|"
        r"\bmodel\b.{0,30}\b(?:create|build|initialize|initialise)\b"
    ),
    "start_training": (
        r"\b(?:start|begin|launch|kick off|run)\b.{0,25}\btraining\b|"
        r"\btraining\b.{0,25}\b(?:start|begin|launch|run)\b"
    ),
    "stop_training": r"\b(?:stop|halt|end|terminate|cancel|pause)\b",
    "set_topology": r"\b(?:topology|topolog\w*|ring|mesh|chain|star)\b",
    "optimize_topology": (
        r"\b(?:optimi[sz]e|evaluate|search|explore)\b.{0,30}"
        r"\b(?:topology|topologies|layout|layouts|communication)\b|"
        r"\b(?:topology|topologies)\b.{0,30}"
        r"\b(?:optimi[sz]e|evaluate|search|explore)\b"
    ),
    "set_rounds": r"\brounds?\b",
    "set_epochs": r"\bepochs?\b",
    "enable_nas": r"\b(?:nas|neural architecture search|architecture search)\b",
    "get_status": r"\b(?:status|health|check)\b",
    "get_logs": r"\blog(?:s|ging)?\b",
    "get_results": r"\b(?:results?|outcomes?|metrics?)\b",
    "compare_runs": (
        r"\bcompare\b.{0,30}\bruns?\b|\bruns?\b.{0,30}\bcompare\b"
    ),
    "suggest_next_action": r"\b(?:next|suggest|recommend|advise)\b",
}
REPLY_ACTION_VERBS = {
    "start_training": (
        r"\b(?:start|begin|launch)\b.{0,30}\btraining\b|"
        r"\btraining\b.{0,30}\b(?:start|begin|launch)\b"
    ),
    "stop_training": r"\b(?:stop|halt|end|terminate|cancel|pause)\b",
    "set_rounds": (
        r"\b(?:set|change|adjust)\b.{0,20}\brounds?\b|"
        r"\brounds?\b.{0,20}\b(?:to|at)\b"
    ),
    "set_epochs": (
        r"\b(?:set|change|adjust)\b.{0,20}\bepochs?\b|"
        r"\bepochs?\b.{0,20}\b(?:to|at)\b"
    ),
    "get_status": (
        r"\b(?:check|retrieve|show|get)\b.{0,20}\b(?:status|health|state)\b"
    ),
    "get_logs": r"\b(?:retrieve|show|get|fetch)\b.{0,20}\blogs?\b",
    "get_results": (
        r"\b(?:retrieve|show|get|fetch|inspect)\b.{0,20}"
        r"\b(?:results?|outcomes?|metrics?)\b"
    ),
    "compare_runs": r"\bcompare\b.{0,25}\bruns?\b",
    "suggest_next_action": (
        r"\b(?:suggest|recommend|advise)\b.{0,20}\b(?:next|action|step)\b"
    ),
}
CONTROL_REPLY_ACTION_PATTERN = re.compile(
    r"\bi['’]ll\s+(?:start|begin|run|launch|set|change|configure|enable|"
    r"disable|stop|halt|cancel|terminate|upload|delete|remove|create|build|"
    r"initialize|optimise|optimize|compare|retrieve|fetch|show|display|"
    r"process|preprocess|recommend|suggest)\b",
    re.IGNORECASE,
)
PADDING_PATTERNS = {
    "for case <number>": re.compile(r"\bfor\s+case\s+\d+\b", re.IGNORECASE),
    "Quick question:": re.compile(r"\bquick\s+question\s*:", re.IGNORECASE),
    "accessible to a curious beginner": re.compile(
        r"\baccessible\s+to\s+a\s+curious\s+beginner\b", re.IGNORECASE
    ),
    "The request concerns": re.compile(r"\bthe\s+request\s+concerns\b", re.IGNORECASE),
    "the analysts are": re.compile(r"\bthe\s+analysts\s+are\b", re.IGNORECASE),
}
IDENTIFIER_PATTERN = re.compile(
    r"\b[a-z][a-z0-9]*_[a-z0-9][a-z0-9_.-]*\b"
    r"|\b[a-z][a-z0-9]*-[a-z0-9]*\d[a-z0-9-]*\b"
    r"|\b[a-z][a-z0-9]*\.[a-z0-9]{2,5}\b",
    re.IGNORECASE,
)
SECRET_VALUE_PATTERN = re.compile(
    r"\b(?:gsk|sk|ghp|github_pat|xox[baprs])_[A-Za-z0-9_-]{12,}\b"
    r"|\bBearer\s+[A-Za-z0-9._~+/-]{12,}={0,2}"
    r"|\b(?:api[_ -]?key|access[_ -]?token|token|secret)\s*[:=]\s*['\"]?"
    r"[A-Za-z0-9._~+/-]{12,}={0,2}"
    r"|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
    re.IGNORECASE,
)
SECRET_AFTER_LABEL_PATTERN = re.compile(
    r"\b(?:api[_ -]?key|access[_ -]?token|token|password|credential)\b"
    r"(?:\s+(?:is|equals|value|of))?\s*[:=]?\s+"
    r"['\"]?([A-Za-z0-9._~+/-]{12,}={0,2})",
    re.IGNORECASE,
)
SHELL_COMMAND_PATTERN = re.compile(
    r"\b(?:scp|ssh|sudo|chmod|chown|curl|wget|bash\s+-c|sh\s+-c|"
    r"python(?:3)?\s+-c|powershell|nc|netcat|rm\s+-rf)\b",
    re.IGNORECASE,
)
ROMAN_URDU_PATTERN = re.compile(
    r"\b(?:kar\s*do|kardo|karo|kijiye|kijye|mujhe|abhi|jaldi|shuru|band|"
    r"chalao|chala|dikhao|dikha|batao|bata|hai|hain|nahi|nahin|yaar|bhai|"
    r"rok\s*do|rakh\s*do|bana\s*do|banao|tayyar|nikaal)\b",
    re.IGNORECASE,
)
DATASET_MODEL_PATTERN = re.compile(
    r"\b(?:cifar10(?:[_-][a-z0-9]+)*|[a-z][a-z0-9]*_[a-z0-9_.-]+|"
    r"[a-z][a-z0-9]*\.[a-z0-9]{2,5}|"
    r"[a-z][a-z0-9]*-[a-z0-9]*\d[a-z0-9-]*|my-data|my-images)\b",
    re.IGNORECASE,
)
NUMBER_PARAM_NAMES = {
    "limit": "log limits",
    "rounds": "rounds",
    "epochs": "epochs",
    "seeds": "seeds",
}
TYPO_WORDS = {
    "alpa": "alpha",
    "comapre": "compare",
    "confgure": "configure",
    "epocs": "epochs",
    "recieve": "receive",
    "staus": "status",
    "statuz": "status",
    "teh": "the",
    "topolgy": "topology",
    "tranning": "training",
    "trianing": "training",
    "uplod": "upload",
}
ABBREVIATIONS = {
    "cfg",
    "cmp",
    "dir",
    "ep",
    "eps",
    "iid",
    "lr",
    "nas",
    "pls",
    "plz",
    "w/",
    "w/o",
}
NON_ENGLISH_MARKERS = {
    "bonjour",
    "merci",
    "hola",
    "gracias",
    "por favor",
    "danke",
    "bitte",
    "ciao",
    "namaste",
    "s'il vous plait",
}
CHAT_ONLY_PATTERN = re.compile(
    r"^\s*(?:(?:hi|hello|hey|thanks|thank you|thx|ty|good morning|"
    r"good afternoon|good evening|how are you)[!.?,\s]*)+$",
    re.IGNORECASE,
)
ACTION_PHRASES = {
    "upload_dataset": r"\b(?:upload|add|load|import)\b.{0,30}\bdataset\b",
    "preprocess_dataset": r"\b(?:preprocess|normalize|normalise|standardize)\b",
    "create_initial_model": r"\b(?:(?:create|build|initialize|initialise)\b.{0,25}\bmodel\b|"
    r"\bmodel\b.{0,20}\b(?:create|build|initialize|initialise)\b)",
    "start_training": r"\b(?:start|begin|run|launch)\b.{0,25}\btraining\b",
    "stop_training": r"\b(?:stop|pause|cancel|halt)\b.{0,25}\btraining\b",
    "set_topology": r"\b(?:set|change|switch|use)\b.{0,25}\btopolog\w*\b",
    "optimize_topology": r"\boptimi[sz]e\b.{0,25}\btopolog\w*\b",
    "set_rounds": r"\b(?:set|change|use)\b.{0,20}\brounds?\b",
    "set_epochs": r"\b(?:set|change|use)\b.{0,20}\bepochs?\b",
    "enable_nas": r"\b(?:enable|disable|turn on|turn off|switch on|switch off)\b"
    r".{0,15}\bnas\b|\bnas\b.{0,15}\b(?:on|off|enabled|disabled)\b",
    "get_status": r"\b(?:status|health)\b",
    "get_logs": r"\blogs?\b",
    "get_results": r"\b(?:results?|metrics?)\b",
    "compare_runs": r"\bcompare\b.{0,30}\bruns?\b",
    "suggest_next_action": r"\b(?:suggest|recommend)\b.{0,30}\b(?:next|action|step)\b",
}
TOPIC_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "could", "do", "for",
    "from", "give", "good", "help", "i", "in", "is", "it", "me", "my", "of",
    "on", "please", "some", "the", "to", "what", "you", "your",
    "question", "quick", "how", "case", "general", "practical", "useful",
    "tell", "suggest", "recommend", "give",
}
CHAT_TOPIC_PATTERNS = {
    "food and restaurants": r"\b(?:restaurant|cafe|coffee|lunch|dinner|food|recipe|meal)\b",
    "weather": r"\bweather|rain|sunny|temperature|forecast\b",
    "travel": r"\b(?:travel|flight|hotel|trip|vacation|train ticket)\b",
    "email and writing": r"\b(?:email|letter|writing|rewrite|grammar|translate)\b",
    "calendar and scheduling": r"\b(?:calendar|meeting|appointment|schedule|reminder)\b",
    "entertainment": r"\b(?:movie|music|song|book|game|joke|poem)\b",
    "shopping": r"\b(?:shopping|groceries|buy|store|price)\b",
    "health": r"\b(?:health|doctor|exercise|sleep|symptom)\b",
    "work and study": r"\b(?:homework|study|class|office|work|project)\b",
}
ENUM_SYNONYMS = {
    "standard": ("standard", "standard scaling", "standard normalization", "z score"),
    "minmax": ("minmax", "min max", "min-max", "min max scaling", "min-max scaling"),
    "none": ("none", "no normalization", "without normalization", "no scaling"),
    "dirichlet_a01": (
        "dirichlet_a01",
        "dirichlet a01",
        "dirichlet a0.1",
        "concentration parameter 0.1",
        "alpha 0.1",
    ),
    "dirichlet_a03": (
        "dirichlet_a03",
        "dirichlet a03",
        "dirichlet a0.3",
        "concentration parameter 0.3",
        "alpha 0.3",
    ),
    "dirichlet_a05": (
        "dirichlet_a05",
        "dirichlet a05",
        "dirichlet a0.5",
        "concentration parameter 0.5",
        "alpha 0.5",
    ),
    "dirichlet_a07": (
        "dirichlet_a07",
        "dirichlet a07",
        "dirichlet a0.7",
        "concentration parameter 0.7",
        "alpha 0.7",
    ),
    "ring": ("ring", "ring topology"),
    "directed_ring": ("directed_ring", "directed ring"),
    "bidirectional_ring": ("bidirectional_ring", "bidirectional ring"),
    "mesh": ("mesh", "mesh topology"),
    "chain": ("chain", "chain topology"),
    "star": ("star", "star topology"),
}
BOOLEAN_SYNONYMS = {
    True: (
        "enable",
        "enabled",
        "turn on",
        "switch on",
        "activate",
        "start nas",
        "nas is on",
    ),
    False: (
        "disable",
        "disabled",
        "turn off",
        "switch off",
        "deactivate",
        "stop nas",
        "nas is off",
    ),
}


def load_schema(path: Path = SCHEMA_PATH) -> tuple[dict[str, Any], set[str], Draft202012Validator]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    action_branches = schema["$defs"]["action"]["oneOf"]
    action_names = {
        branch["properties"]["action"]["const"] for branch in action_branches
    }
    chat_schema = {
        "$ref": "#/$defs/chatResponse",
        "$defs": schema["$defs"],
    }
    return schema, action_names, Draft202012Validator(chat_schema)


def clean_line(line: str) -> str:
    cleaned = line.replace("<escape>", "")
    cleaned = cleaned.rstrip()
    while cleaned.endswith("\\"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def normalize_quotes(value: Any) -> Any:
    if isinstance(value, str):
        return value.translate(QUOTE_TRANSLATION)
    if isinstance(value, list):
        return [normalize_quotes(item) for item in value]
    if isinstance(value, dict):
        return {
            normalize_quotes(key): normalize_quotes(item)
            for key, item in value.items()
        }
    return value


def scrub_secrets(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = SECRET_VALUE_PATTERN.sub("[SECRET OMITTED]", value)
        return SECRET_AFTER_LABEL_PATTERN.sub(
            lambda match: match.group(0)[: match.start(1) - match.start(0)]
            + "[SECRET OMITTED]",
            cleaned,
        )
    if isinstance(value, list):
        return [scrub_secrets(item) for item in value]
    if isinstance(value, dict):
        return {
            scrub_secrets(key): scrub_secrets(item)
            for key, item in value.items()
        }
    return value


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])",
            normalized_text,
        )
    )


def parameter_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [nested for item in value for nested in parameter_values(item)]
    return [value]


def parameter_value_in_text(_name: str, value: Any, user_text: str) -> bool:
    del _name
    text = normalize_text(user_text)
    if isinstance(value, bool):
        return any(_contains_phrase(text, phrase) for phrase in BOOLEAN_SYNONYMS[value])
    if isinstance(value, (int, float)):
        candidates = {str(value)}
        if isinstance(value, float) and value.is_integer():
            candidates.add(str(int(value)))
        return any(_contains_phrase(text, candidate) for candidate in candidates)
    value_text = str(value)
    for enum_value, synonyms in ENUM_SYNONYMS.items():
        if value_text == enum_value:
            return any(_contains_phrase(text, phrase) for phrase in synonyms)
    return _contains_phrase(text, value_text)


def label_parameter_values(label: dict[str, Any]) -> dict[str, Any]:
    action = label.get("action", {})
    if not isinstance(action, dict):
        return {}
    parameters = action.get("parameters", {})
    return parameters if isinstance(parameters, dict) else {}


def reply_label_consistency_errors(row: dict[str, Any]) -> list[str]:
    label, label_error = row_label(row)
    if label_error or label is None:
        return []
    action = label.get("action", {})
    if not isinstance(action, dict):
        return []
    action_name = action.get("action")
    parameters = label_parameter_values(label)
    reply = label.get("reply")
    reply_text = reply if isinstance(reply, str) else ""
    errors = []

    if action_name == "set_topology":
        topology = parameters.get("topology")
        if isinstance(topology, str):
            mentions = [
                TOPOLOGY_ALIAS_TO_VALUE[match.group(0).casefold()]
                for match in TOPOLOGY_MENTION_PATTERN.finditer(reply_text)
            ]
            if topology not in mentions or any(
                mention != topology for mention in mentions
            ):
                errors.append(
                    "reply-label mismatch: reply topology names do not match "
                    f"label topology {topology!r}"
                )

    if action_name == "enable_nas" and isinstance(parameters.get("enabled"), bool):
        enabled = parameters["enabled"]
        mentioned_states = {
            state
            for state, phrases in BOOLEAN_SYNONYMS.items()
            if any(_contains_phrase(reply_text, phrase) for phrase in phrases)
        }
        if re.search(
            r"\b(?:turn|switch)\s+nas\s+on\b|\bnas\s+(?:is\s+)?on\b",
            reply_text,
            re.IGNORECASE,
        ):
            mentioned_states.add(True)
        if re.search(
            r"\b(?:turn|switch)\s+nas\s+off\b|\bnas\s+(?:is\s+)?off\b",
            reply_text,
            re.IGNORECASE,
        ):
            mentioned_states.add(False)
        expected_state = enabled
        if expected_state not in mentioned_states or any(
            state != expected_state for state in mentioned_states
        ):
            expected_word = "on" if enabled else "off"
            errors.append(
                "reply-label mismatch: NAS reply state does not match "
                f"enabled={str(enabled).lower()} ({expected_word})"
            )

    numeric_label_values = set(
        NUMBER_PATTERN.findall(
            json.dumps(parameters, ensure_ascii=False, sort_keys=True)
        )
    )
    available_numbers = numeric_label_values | set(
        NUMBER_PATTERN.findall(row_user_text(row))
    )
    unsupported_reply_numbers = set(NUMBER_PATTERN.findall(reply_text)) - available_numbers
    if unsupported_reply_numbers:
        errors.append(
            "reply-label mismatch: reply contains numbers absent from user text "
            "and label parameters: "
            + ", ".join(sorted(unsupported_reply_numbers))
        )

    if action_name in {"clarify", "refuse", "redirect"} and (
        CONTROL_REPLY_ACTION_PATTERN.search(reply_text)
    ):
        errors.append(
            f"reply-label mismatch: {action_name} reply promises to perform an action"
        )
    return errors


def intent_suspects(row: dict[str, Any]) -> list[str]:
    label, label_error = row_label(row)
    if label_error or label is None:
        return ["cannot assess intent because label is invalid"]
    action_data = label.get("action", {})
    if not isinstance(action_data, dict):
        return ["cannot assess intent because action is invalid"]
    action_name = action_data.get("action")
    if not isinstance(action_name, str):
        return ["cannot assess intent because action name is missing"]

    user_text = row_user_text(row)
    reply = label.get("reply")
    reply_text = reply if isinstance(reply, str) else ""
    suspects = []
    intent_pattern = INTENT_PATTERNS.get(action_name)
    if intent_pattern and not re.search(intent_pattern, user_text, re.IGNORECASE):
        suspects.append(f"user text lacks expected {action_name} intent keywords")

    for reply_action, pattern in REPLY_ACTION_VERBS.items():
        if reply_action != action_name and re.search(pattern, reply_text, re.IGNORECASE):
            suspects.append(
                f"reply action language resembles {reply_action}, not {action_name}"
            )

    for padding_name, pattern in PADDING_PATTERNS.items():
        if pattern.search(user_text):
            suspects.append(f"padding-like text: {padding_name}")
    return suspects


def row_user_text(row: dict[str, Any]) -> str:
    for key in ("user", "user_text", "text", "prompt"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def row_label(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    label = row.get("label")
    if isinstance(label, str):
        try:
            label = json.loads(label)
        except json.JSONDecodeError as error:
            return None, f"label is not valid JSON: {error.msg}"
    if not isinstance(label, dict):
        return None, "label must be a JSON object or a JSON-encoded object"
    return label, None


def category_name(row: dict[str, Any], label: dict[str, Any] | None = None) -> str:
    category = row.get("category")
    if isinstance(category, str) and category.strip():
        return category.strip()
    if label is not None:
        action = label.get("action", {})
        if isinstance(action, dict):
            return str(action.get("action", "unknown"))
    return "unknown"


def _contains_non_english_marker(text: str) -> bool:
    lowered = text.casefold()
    return bool(ROMAN_URDU_PATTERN.search(text)) or any(
        _contains_phrase(lowered, marker) for marker in NON_ENGLISH_MARKERS
    )


def _contains_non_ascii(text: str) -> bool:
    return any(ord(character) > 127 for character in text)


def _schema_action_branches(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        branch["properties"]["action"]["const"]: branch
        for branch in schema["$defs"]["action"]["oneOf"]
    }


def _mentioned_identifiers(text: str) -> set[str]:
    candidates = {match.group(0) for match in IDENTIFIER_PATTERN.finditer(text)}
    candidates.update(match.group(0) for match in RUN_ID_PATTERN.finditer(text))
    return {candidate.casefold() for candidate in candidates}


def _mentioned_dataset_model_names(text: str) -> set[str]:
    names = {name.casefold() for name in IDENTIFIER_PATTERN.findall(text)}
    names.update(name.casefold() for name in DATASET_MODEL_PATTERN.findall(text))
    run_ids = {name.casefold() for name in RUN_ID_PATTERN.findall(text)}
    return names - run_ids - NON_ENTITY_IDENTIFIERS


def _secrets_in_value(value: Any) -> bool:
    if isinstance(value, str):
        return SECRET_VALUE_PATTERN.search(value) is not None
    if isinstance(value, list):
        return any(_secrets_in_value(item) for item in value)
    if isinstance(value, dict):
        return any(
            _secrets_in_value(key) or _secrets_in_value(item)
            for key, item in value.items()
        )
    return False


def _required_fields(schema: dict[str, Any], action: str) -> list[str]:
    branch = _schema_action_branches(schema).get(action, {})
    return list(
        branch.get("properties", {})
        .get("parameters", {})
        .get("required", [])
    )


def _action_mentions(text: str) -> list[tuple[int, str]]:
    found = []
    for action, pattern in ACTION_PHRASES.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            found.append((match.start(), action))
    return sorted(found)


def _required_value_is_stated(name: str, user_text: str) -> bool:
    if name == "rounds":
        return bool(re.search(r"\b\d+\s*rounds?\b", user_text, re.I))
    if name == "epochs":
        return bool(re.search(r"\b\d+\s*epochs?\b", user_text, re.I))
    if name == "topology":
        return any(
            _contains_phrase(user_text, topology)
            for topology in ENUM_SYNONYMS
            if topology
            in {
                "ring",
                "directed_ring",
                "bidirectional_ring",
                "mesh",
                "chain",
                "star",
            }
        )
    if name in {"dataset_name", "model_name"}:
        return bool(_mentioned_identifiers(user_text))
    if name == "enabled":
        return any(
            _contains_phrase(user_text, phrase)
            for phrases in BOOLEAN_SYNONYMS.values()
            for phrase in phrases
        )
    if name == "run_ids":
        return len(RUN_ID_PATTERN.findall(user_text)) >= 2
    return False


def compound_errors(row: dict[str, Any], label: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    if category_name(row, label) != "compound":
        return []
    text = row_user_text(row)
    mentions = _action_mentions(text)
    errors = []
    if len(mentions) < 2:
        return ["compound row does not clearly mention two actions"]
    first_action = mentions[0][1]
    required = _required_fields(schema, first_action)
    complete = all(_required_value_is_stated(name, text) for name in required)
    action = label.get("action", {})
    label_action = action.get("action") if isinstance(action, dict) else None
    expected = first_action if complete else "clarify"
    if label_action != expected:
        errors.append(
            f"compound label mismatch: first action {first_action} "
            f"is {'complete' if complete else 'incomplete'}, expected {expected}"
        )
    return errors


def token_set_jaccard(left: str, right: str) -> float:
    left_set = set(WORD_PATTERN.findall(left))
    right_set = set(WORD_PATTERN.findall(right))
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def masked_tokens(text: str) -> set[str]:
    value = TIMESTAMP_PATTERN.sub(" timestamp ", text.casefold())
    value = RUN_ID_PATTERN.sub(" runid ", value)
    value = DATASET_MODEL_PATTERN.sub(" datasetname ", value)
    value = NUMBER_PATTERN.sub(" number ", value)
    return {
        token
        for token in WORD_PATTERN.findall(value)
        if token not in VARIABLE_TOKENS
    }


def masked_jaccard(left: str, right: str) -> float:
    left_set = masked_tokens(left)
    right_set = masked_tokens(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def referenced_values(parameters: dict[str, Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for name, value in parameters.items():
        for item in parameter_values(value):
            if name in NUMBER_PARAM_NAMES and isinstance(item, (int, float)):
                values[NUMBER_PARAM_NAMES[name]].add(str(item))
            elif name == "dataset_name" and isinstance(item, str):
                values["dataset names"].add(item.casefold())
    return values


def repeated_value_reasons(
    values: dict[str, set[str]], prior_values: dict[str, set[str]]
) -> list[str]:
    return [
        f"reused {group} value(s): {', '.join(sorted(items & prior_values.get(group, set())))}"
        for group, items in values.items()
        if items & prior_values.get(group, set())
    ]


def validate_row(
    row: dict[str, Any],
    *,
    schema: dict[str, Any],
    validator: Draft202012Validator,
    action_names: set[str],
) -> list[str]:
    errors: list[str] = []
    user_text = row_user_text(row)
    if not user_text.strip():
        errors.append("missing user text")
    label, label_error = row_label(row)
    if label_error:
        return errors + [label_error]
    assert label is not None
    reply = label.get("reply")
    category = category_name(row, label)
    if category not in ALLOWED_CATEGORIES:
        errors.append(f"category is not allowed: {category!r}")
    reply_text = reply if isinstance(reply, str) else ""
    if _contains_non_english_marker(user_text) or _contains_non_english_marker(reply_text):
        errors.append("Roman Urdu or non-English marker is not allowed")
    if _contains_non_ascii(user_text) or _contains_non_ascii(reply_text):
        errors.append("non-ASCII character in user text or reply")
    if _secrets_in_value(row):
        errors.append("API key/token-like value is not allowed")
    action = label.get("action", {})
    action_name = action.get("action") if isinstance(action, dict) else None
    if not isinstance(action_name, str) or action_name not in action_names:
        errors.append(f"action is not in allow-list: {action_name!r}")
    for error in validator.iter_errors(label):
        location = ".".join(str(part) for part in error.absolute_path) or "label"
        errors.append(f"schema {location}: {error.message}")
    parameters = label_parameter_values(label)
    if isinstance(action, dict) and action.get("action") in {
        "clarify",
        "refuse",
        "redirect",
    } and parameters:
        errors.append(f"{action.get('action')} must have empty parameters")
    for name, value in parameters.items():
        for item in parameter_values(value):
            if not parameter_value_in_text(name, item, user_text):
                errors.append(
                    f"parameter value not stated in user text: {name}={item!r}"
                )
    user_names = _mentioned_identifiers(user_text)
    missing_reply_names = _mentioned_identifiers(reply_text) - user_names
    if missing_reply_names:
        errors.append(
            "dataset/model/run identifier in reply absent from user text: "
            + ", ".join(sorted(missing_reply_names))
        )
    if category == "unsafe" and action_name != "refuse":
        errors.append("unsafe category must use refuse")
    if category == "chit_chat":
        if action_name != "redirect":
            errors.append("chit_chat category must use redirect")
        if CHAT_ONLY_PATTERN.fullmatch(user_text):
            errors.append("chit_chat row is only a greeting or thanks")
    errors.extend(reply_label_consistency_errors(row))
    errors.extend(compound_errors(row, label, schema))
    return errors


def load_batch(
    path: Path,
) -> tuple[
    list[tuple[int, dict[str, Any]]],
    list[dict[str, Any]],
    int,
    int,
    list[dict[str, Any]],
]:
    rows: list[tuple[int, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    invalid_json: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    valid_json_count = 0
    for line_number, raw_line in enumerate(lines, 1):
        cleaned = clean_line(raw_line)
        if not cleaned.strip():
            continue
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as error:
            normalized_quotes = cleaned.translate(QUOTE_TRANSLATION)
            if normalized_quotes != cleaned:
                try:
                    parsed = json.loads(normalized_quotes)
                except json.JSONDecodeError:
                    parsed = None
                else:
                    rows.append((line_number, normalize_quotes(parsed)))
                    valid_json_count += 1
                    continue
            entry = {
                "batch": path.name,
                "line": line_number,
                "reason": f"invalid JSON: {error.msg}",
                "raw": scrub_secrets(normalized_quotes),
            }
            rejected.append(entry)
            invalid_json.append(entry)
            continue
        valid_json_count += 1
        if not isinstance(parsed, dict):
            rejected.append(
                {
                    "batch": path.name,
                    "line": line_number,
                    "reason": "JSONL row must be an object",
                    "raw": scrub_secrets(parsed),
                }
            )
            continue
        rows.append((line_number, normalize_quotes(parsed)))
    return rows, rejected, len(lines), valid_json_count, invalid_json


def _sort_batch_key(path: Path) -> tuple[int, str]:
    match = re.search(r"batch(\d+)", path.stem, re.IGNORECASE)
    return (int(match.group(1)) if match else sys.maxsize, path.name.casefold())


def _max_share(values: list[str]) -> dict[str, Any]:
    counts = Counter(values)
    return {
        "count": len(values),
        "distinct_values": len(counts),
        "max_share": (
            max(counts.values()) / len(values) if values else 0.0
        ),
        "max_value": counts.most_common(1)[0][0] if counts else None,
        "values": sorted(counts),
        "counts": dict(sorted(counts.items())),
    }


def _likely_typo(user_text: str) -> bool:
    words = {word.casefold() for word in WORD_PATTERN.findall(user_text)}
    return bool(words & TYPO_WORDS.keys())


def _domain_missing_parameters(label: dict[str, Any]) -> list[str]:
    action = label.get("action", {})
    if not isinstance(action, dict):
        return []
    name = action.get("action")
    parameters = label_parameter_values(label)
    expected = {
        "start_training": ("dataset_name",),
        "enable_nas": ("seeds", "search_time_limit_seconds"),
    }.get(name, ())
    return [parameter for parameter in expected if parameter not in parameters]


def _reply_opening(reply: str) -> str:
    words = WORD_PATTERN.findall(reply.casefold())
    return " ".join(words[:3]) if words else "(empty)"


def _topic_words(text: str) -> list[str]:
    topics = [
        topic
        for topic, pattern in CHAT_TOPIC_PATTERNS.items()
        if re.search(pattern, text, re.IGNORECASE)
    ]
    if topics:
        return topics
    words = [
        word.casefold()
        for word in WORD_PATTERN.findall(text)
        if word.casefold() not in TOPIC_STOP_WORDS and len(word) > 2
    ]
    return words


def build_report(
    clean_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    *,
    reused_values: Counter[str],
    reused_value_names: dict[str, set[str]],
    exact_duplicates: Counter[str],
    per_batch: dict[str, dict[str, Any]],
    schema: dict[str, Any],
) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actions: Counter[str] = Counter()
    name_counts: Counter[str] = Counter()
    name_total = 0
    parameter_values_by_action: dict[str, list[str]] = defaultdict(list)
    word_counts: dict[str, list[int]] = defaultdict(list)
    please_counts: Counter[str] = Counter()
    typo_counts: Counter[str] = Counter()
    abbreviation_counts: Counter[str] = Counter()
    long_counts: Counter[str] = Counter()
    short_counts: Counter[str] = Counter()
    first_words: Counter[str] = Counter()
    category_first_words: dict[str, Counter[str]] = defaultdict(Counter)
    reply_openings: dict[str, Counter[str]] = defaultdict(Counter)
    chit_chat_topics: Counter[str] = Counter()
    equal_rounds_epochs = 0
    training_rows_with_both = 0
    absent_optional_fields: Counter[str] = Counter()
    compound_inconsistency = 0
    schema_branches = _schema_action_branches(schema)

    for row in clean_rows:
        label, _ = row_label(row)
        if label is None:
            continue
        category = category_name(row, label)
        categories[category].append(row)
        user_text = row_user_text(row)
        word_count = len(user_text.split())
        word_counts[category].append(word_count)
        please_counts[category] += bool(re.search(r"\bplease\b", user_text, re.I))
        typo_counts[category] += _likely_typo(user_text)
        words = {word.casefold() for word in WORD_PATTERN.findall(user_text)}
        abbreviation_counts[category] += bool(words & ABBREVIATIONS)
        long_counts[category] += word_count >= 25
        short_counts[category] += word_count <= 3
        first = WORD_PATTERN.findall(user_text.casefold())
        if first:
            first_words[first[0]] += 1
            category_first_words[category][first[0]] += 1
        reply = label.get("reply", "")
        if isinstance(reply, str):
            reply_openings[category][_reply_opening(reply)] += 1
        action_data = label.get("action", {})
        if not isinstance(action_data, dict):
            continue
        action = str(action_data.get("action", "unknown"))
        actions[action] += 1
        parameters = label_parameter_values(label)
        row_names: set[str] = set()
        for parameter, value in parameters.items():
            values = parameter_values(value)
            for item in values:
                encoded = json.dumps(item, ensure_ascii=False, sort_keys=True)
                parameter_values_by_action[f"{action}.{parameter}"].append(encoded)
                if parameter in {"dataset_name", "model_name"} and isinstance(item, str):
                    row_names.add(item.casefold())
        for name in row_names:
            name_counts[name] += 1
            name_total += 1
        if action == "start_training":
            rounds, epochs = parameters.get("rounds"), parameters.get("epochs")
            if rounds is not None and epochs is not None:
                training_rows_with_both += 1
                equal_rounds_epochs += rounds == epochs
        for parameter in _domain_missing_parameters(label):
            absent_optional_fields[f"{action}.{parameter}"] += 1
        if category == "compound" and compound_errors(row, label, schema):
            compound_inconsistency += 1
        if category == "chit_chat":
            chit_chat_topics.update(_topic_words(user_text))

    category_report = {}
    for category, rows in sorted(categories.items()):
        words = word_counts[category]
        opening_counts = reply_openings[category]
        category_report[category] = {
            "rows": len(rows),
            "mean_word_count": sum(words) / len(words) if words else 0.0,
            "please_rate": please_counts[category] / len(rows) if rows else 0.0,
            "rows_with_typos": typo_counts[category],
            "rows_with_abbreviations": abbreviation_counts[category],
            "rows_at_least_25_words": long_counts[category],
            "rows_at_most_3_words": short_counts[category],
            "reply_openings_top_10": [
                {
                    "opening": opening,
                    "count": count,
                    "share": count / len(rows) if rows else 0.0,
                }
                for opening, count in opening_counts.most_common(10)
            ],
        }

    schema_fields = {}
    for action, branch in sorted(schema_branches.items()):
        parameters = branch.get("properties", {}).get("parameters", {})
        properties = parameters.get("properties", {})
        required = set(parameters.get("required", []))
        if action in {"clarify", "refuse", "redirect", "stop_training"}:
            continue
        for parameter in (
            "dataset_name",
            "model_name",
            "seeds",
            "search_time_limit_seconds",
            "alphas",
            "normalization",
        ):
            if parameter in properties:
                schema_fields[f"{action}.{parameter}"] = (
                    "required" if parameter in required else "optional"
                )
    name_share_limit = 0.05
    return {
        "clean_rows": len(clean_rows),
        "rejected_rows": len(rejected_rows),
        "counts_per_category": {
            category: stats["rows"] for category, stats in category_report.items()
        },
        "counts_per_action": dict(sorted(actions.items())),
        "categories": category_report,
        "distinct_dataset_model_names": len(name_counts),
        "dataset_model_name_counts": dict(name_counts.most_common()),
        "dataset_model_names_over_5_percent": {
            name: count / name_total
            for name, count in name_counts.items()
            if name_total and count / name_total > name_share_limit
        },
        "per_action_parameter_values": {
            name: {
                **_max_share(values),
                "flagged_over_20_percent": (
                    len(values) >= 8 and _max_share(values)["max_share"] > 0.20
                ),
            }
            for name, values in sorted(parameter_values_by_action.items())
        },
        "rounds_equal_epochs": {
            "rows": equal_rounds_epochs,
            "start_training_rows_with_both": training_rows_with_both,
            "rate": (
                equal_rounds_epochs / training_rows_with_both
                if training_rows_with_both
                else 0.0
            ),
        },
        "optional_domain_fields_absent": dict(sorted(absent_optional_fields.items())),
        "compound_inconsistent_rows": compound_inconsistency,
        "top_first_words": [
            {"word": word, "count": count}
            for word, count in first_words.most_common(10)
        ],
        "first_word_repeats_by_category": {
            category: [
                {"word": word, "count": count}
                for word, count in counts.most_common(10)
                if count > 1
            ]
            for category, counts in sorted(category_first_words.items())
        },
        "chit_chat_topics_top_10": [
            {"topic": topic, "count": count}
            for topic, count in chit_chat_topics.most_common(10)
        ],
        "exact_duplicate_user_texts": [
            {"text": scrub_secrets(text), "count": count}
            for text, count in exact_duplicates.most_common(20)
            if count > 1
        ],
        "reused_values_from_earlier_rows": dict(sorted(reused_values.items())),
        "reused_value_names_from_earlier_rows": {
            group: sorted(values) for group, values in sorted(reused_value_names.items())
        },
        "schema_field_requirements": schema_fields,
        "dataset_name_pattern": schema["$defs"]["datasetName"],
        "run_id_pattern": schema["$defs"]["runIds"],
        "per_batch": per_batch,
    }


def validate_batches(
    batch_paths: list[Path],
    *,
    schema_path: Path = SCHEMA_PATH,
    clean_dir: Path = CLEAN_DIR,
    rejected_dir: Path = REJECTED_DIR,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema, action_names, validator = load_schema(schema_path)
    clean_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    earlier_texts: list[str] = []
    earlier_values: dict[str, set[str]] = defaultdict(set)
    all_clean: list[dict[str, Any]] = []
    all_rejected: list[dict[str, Any]] = []
    cross_batch_duplicate_count = 0
    reused_value_counts: Counter[str] = Counter()
    reused_value_names: dict[str, set[str]] = defaultdict(set)
    exact_duplicates: Counter[str] = Counter()
    per_batch: dict[str, dict[str, Any]] = {}

    for batch_path in sorted(batch_paths, key=_sort_batch_key):
        (
            parsed_rows,
            line_rejections,
            line_count,
            valid_json_count,
            invalid_json,
        ) = load_batch(batch_path)
        batch_clean: list[dict[str, Any]] = []
        batch_rejected = list(line_rejections)
        batch_reasons: Counter[str] = Counter(
            item["reason"] for item in line_rejections
        )
        for line_number, row in parsed_rows:
            user_text = row_user_text(row)
            label, _ = row_label(row)
            reasons = validate_row(
                row,
                schema=schema,
                validator=validator,
                action_names=action_names,
            )
            normalized_user = normalize_text(user_text)
            if normalized_user and normalized_user in exact_duplicates:
                reasons.append("exact duplicate user text from an earlier row")
            if normalized_user:
                exact_duplicates[normalized_user] += 1
            if user_text and any(
                masked_jaccard(user_text, prior_text) >= JACCARD_THRESHOLD
                for prior_text in earlier_texts
            ):
                reasons.append(
                    f"near duplicate of an earlier batch row (masked token-set Jaccard >= {JACCARD_THRESHOLD:.1f})"
                )
                cross_batch_duplicate_count += 1
            if user_text:
                earlier_texts.append(user_text)
            if label is not None:
                action_data = label.get("action", {})
                action_name = (
                    str(action_data.get("action", "unknown"))
                    if isinstance(action_data, dict)
                    else "unknown"
                )
                parameters = label_parameter_values(label)
                for parameter, value in parameters.items():
                    items = {
                        json.dumps(item, ensure_ascii=False, sort_keys=True)
                        for item in parameter_values(value)
                    }
                    group = f"{action_name}.{parameter}"
                    repeats = items & earlier_values.get(group, set())
                    if repeats:
                        reused_value_counts[group] += len(repeats)
                        reused_value_names[group].update(repeats)
                    earlier_values[group].update(items)
            if reasons:
                reason = "; ".join(dict.fromkeys(reasons))
                batch_rejected.append(
                    {
                        "batch": batch_path.name,
                        "line": line_number,
                        "reason": reason,
                        "row": scrub_secrets(row),
                    }
                )
                batch_reasons.update(reasons)
            else:
                batch_clean.append(row)

        clean_path = clean_dir / batch_path.name
        rejected_path = rejected_dir / batch_path.name
        clean_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in batch_clean
            ),
            encoding="utf-8",
        )
        rejected_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in batch_rejected
            ),
            encoding="utf-8",
        )
        all_clean.extend(batch_clean)
        all_rejected.extend(batch_rejected)
        per_batch[batch_path.name] = {
            "lines": line_count,
            "valid_json": valid_json_count,
            "kept": len(batch_clean),
            "rejected": len(batch_rejected),
            "rejection_reasons": dict(sorted(batch_reasons.items())),
            "invalid_json_lines": invalid_json,
        }

    report = build_report(
        all_clean,
        all_rejected,
        reused_values=reused_value_counts,
        reused_value_names=reused_value_names,
        exact_duplicates=exact_duplicates,
        per_batch=per_batch,
        schema=schema,
    )
    report["same_or_earlier_row_near_duplicate_rejections"] = (
        cross_batch_duplicate_count
    )
    report["clean_rows_target"] = 1000
    report["clean_rows_shortfall"] = max(0, 1000 - len(all_clean))
    dataset_pattern = report["dataset_name_pattern"]["pattern"]
    report["dataset_name_pattern_examples"] = {
        example: bool(re.fullmatch(dataset_pattern, example))
        for example in (
            "ali_project",
            "my-data-v2",
            "training.data",
            "dataset.csv",
            "archive.zip",
        )
    }
    merged_path = clean_dir / "all_clean.jsonl"
    merged_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in all_clean
        ),
        encoding="utf-8",
    )
    return report, all_rejected


def _print_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    print(f"\n{title}")
    values = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        if values
        else len(headers[index])
        for index in range(len(headers))
    ]
    print(" | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in values:
        print(" | ".join(row[index].ljust(widths[index]) for index in range(len(headers))))


def _rejected_preview(value: Any) -> Any:
    sanitized = scrub_secrets(value)
    if isinstance(sanitized, str):
        if SHELL_COMMAND_PATTERN.search(sanitized):
            return "[shell command text omitted]"
        return sanitized
    if isinstance(sanitized, list):
        return [_rejected_preview(item) for item in sanitized]
    if isinstance(sanitized, dict):
        return {key: _rejected_preview(item) for key, item in sanitized.items()}
    return sanitized


def print_report(report: dict[str, Any], rejected_rows: list[dict[str, Any]]) -> None:
    batch_rows = []
    for batch, stats in report["per_batch"].items():
        batch_rows.append(
            [
                batch,
                stats["lines"],
                stats["valid_json"],
                stats["kept"],
                stats["rejected"],
            ]
        )
    _print_table(
        "PER-BATCH COUNTS",
        ["Batch", "Lines", "Valid JSON", "Kept", "Rejected"],
        batch_rows,
    )
    for batch, stats in report["per_batch"].items():
        reason_rows = [
            [reason, count] for reason, count in stats["rejection_reasons"].items()
        ]
        _print_table(
            f"REJECTION REASONS: {batch}",
            ["Reason", "Count"],
            reason_rows or [["(none)", 0]],
        )
        _print_table(
            f"INVALID JSON LINES: {batch}",
            ["Line", "Reason", "Cleaned line"],
            [
                [item["line"], item["reason"], item.get("raw", "")]
                for item in stats["invalid_json_lines"]
            ]
            or [["-", "(none)", "-"]],
        )
    _print_table(
        "COUNTS PER CATEGORY",
        ["Category", "Clean rows"],
        [[category, count] for category, count in report["counts_per_category"].items()]
        or [["(none)", 0]],
    )
    _print_table(
        "COUNTS PER ACTION",
        ["Action", "Clean rows"],
        [[action, count] for action, count in report["counts_per_action"].items()]
        or [["(none)", 0]],
    )
    _print_table(
        "DATASET / MODEL NAMES",
        ["Name", "Count", "Share", "Above 5%"],
        [
            [
                name,
                count,
                f"{count / max(1, sum(report['dataset_model_name_counts'].values())):.1%}",
                "yes" if name in report["dataset_model_names_over_5_percent"] else "no",
            ]
            for name, count in report["dataset_model_name_counts"].items()
        ]
        or [["(none)", 0, "0.0%", "no"]],
    )
    parameter_rows = []
    for field, stats in report["per_action_parameter_values"].items():
        displayed_values = [
            json.loads(value) for value in stats["values"]
        ]
        parameter_rows.append(
            [
                field,
                stats["count"],
                stats["distinct_values"],
                f"{stats['max_share']:.1%}",
                "yes" if stats["flagged_over_20_percent"] else "no",
                json.dumps(displayed_values, ensure_ascii=False),
            ]
        )
    _print_table(
        "PER-ACTION / PARAMETER VALUE DIVERSITY",
        [
            "Action.parameter",
            "N",
            "Distinct",
            "Max share",
            "Flag >20%",
            "Distinct values",
        ],
        parameter_rows or [["(none)", 0, 0, "0.0%", "no", "{}"]],
    )
    _print_table(
        "CATEGORY STYLE METRICS",
        [
            "Category",
            "Rows",
            "Mean words",
            "Please rate",
            "Typos",
            "Abbreviations",
            "<=3 words",
            ">=25 words",
        ],
        [
            [
                category,
                stats["rows"],
                f"{stats['mean_word_count']:.2f}",
                f"{stats['please_rate']:.1%}",
                stats["rows_with_typos"],
                stats["rows_with_abbreviations"],
                stats["rows_at_most_3_words"],
                stats["rows_at_least_25_words"],
            ]
            for category, stats in report["categories"].items()
        ]
        or [["(none)", 0, "0.00", "0.0%", 0, 0, 0, 0]],
    )
    _print_table(
        "FIRST-WORD REPEATS (TOP 10)",
        ["First word", "Rows"],
        [[item["word"], item["count"]] for item in report["top_first_words"]]
        or [["(none)", 0]],
    )
    _print_table(
        "CHIT-CHAT TOPICS (TOP 10 WORDS)",
        ["Topic", "Count"],
        [[item["topic"], item["count"]] for item in report["chit_chat_topics_top_10"]]
        or [["(none)", 0]],
    )
    _print_table(
        "REUSED PARAMETER VALUES FROM EARLIER ROWS",
        ["Action.parameter", "Repeated value occurrences", "Values"],
        [
            [
                group,
                report["reused_values_from_earlier_rows"][group],
                ", ".join(report["reused_value_names_from_earlier_rows"][group]),
            ]
            for group in report["reused_values_from_earlier_rows"]
        ]
        or [["(none)", 0, "-"]],
    )
    opening_rows = []
    for category, stats in report["categories"].items():
        for opening in stats["reply_openings_top_10"]:
            opening_rows.append(
                [
                    category,
                    opening["opening"],
                    opening["count"],
                    f"{opening['share']:.1%}",
                ]
            )
    _print_table(
        "REPLY OPENING DIVERSITY (TOP 10 PER CATEGORY)",
        ["Category", "Opening (first 3 words)", "Count", "Share"],
        opening_rows or [["(none)", "-", 0, "0.0%"]],
    )
    duplicates = report["exact_duplicate_user_texts"]
    _print_table(
        "EXACT DUPLICATE USER TEXTS",
        ["Normalized user text", "Count"],
        [[item["text"], item["count"]] for item in duplicates] or [["(none)", 0]],
    )
    _print_table(
        "SCHEMA REQUIRED / OPTIONAL FIELDS",
        ["Field", "Schema status"],
        [
            ["start_training.dataset_name", report["schema_field_requirements"].get("start_training.dataset_name", "not defined")],
            ["enable_nas.seeds", report["schema_field_requirements"].get("enable_nas.seeds", "not defined")],
            ["enable_nas.search_time_limit_seconds", report["schema_field_requirements"].get("enable_nas.search_time_limit_seconds", "not defined")],
            ["optimize_topology.alphas", report["schema_field_requirements"].get("optimize_topology.alphas", "not defined")],
            ["preprocess_dataset.normalization", report["schema_field_requirements"].get("preprocess_dataset.normalization", "not defined")],
        ],
    )
    _print_table(
        "SCHEMA NAME PATTERN",
        ["Definition", "Value"],
        [
            ["datasetName pattern", report["dataset_name_pattern"]["pattern"]],
            ["datasetName maxLength", report["dataset_name_pattern"]["maxLength"]],
            ["runIds definition", json.dumps(report["run_id_pattern"], sort_keys=True)],
        ]
        + [
            ["Example " + example, "pass" if passes else "fail"]
            for example, passes in report["dataset_name_pattern_examples"].items()
        ],
    )
    _print_table(
        "CONSISTENCY AND TARGET",
        ["Metric", "Value"],
        [
            [
                "rounds == epochs",
                f"{report['rounds_equal_epochs']['rows']}/"
                f"{report['rounds_equal_epochs']['start_training_rows_with_both']} "
                f"({report['rounds_equal_epochs']['rate']:.1%})",
            ],
            ["compound inconsistent rows", report["compound_inconsistent_rows"]],
            [
                "optional domain fields absent",
                json.dumps(report["optional_domain_fields_absent"], sort_keys=True),
            ],
            [
                "near-duplicate rows rejected",
                report["same_or_earlier_row_near_duplicate_rejections"],
            ],
            ["clean rows total", report["clean_rows"]],
            ["target", report["clean_rows_target"]],
            ["shortfall", report["clean_rows_shortfall"]],
        ],
    )
    _print_table(
        "SAMPLE REJECTED ROWS (FIRST 15)",
        ["Batch", "Line", "Reason", "Row"],
        [
            [
                item.get("batch", ""),
                item.get("line", ""),
                item.get("reason", ""),
                json.dumps(
                    _rejected_preview(item.get("row", item.get("raw", ""))),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ]
            for item in rejected_rows[:15]
        ]
        or [["-", "-", "(none)", "-"]],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "batches",
        nargs="*",
        type=Path,
        help="Batch JSONL paths; default discovers top-level JSONL files in data/chatgpt",
    )
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--clean-dir", type=Path, default=None)
    parser.add_argument("--rejected-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    paths = args.batches or sorted(args.batch_dir.glob("*.jsonl"), key=_sort_batch_key)
    if not paths:
        print(f"No batchNN.jsonl files found in {args.batch_dir}", file=sys.stderr)
        return 2
    clean_dir = args.clean_dir or args.batch_dir / "clean"
    rejected_dir = args.rejected_dir or args.batch_dir / "rejected"
    report, rejected = validate_batches(
        paths,
        schema_path=args.schema,
        clean_dir=clean_dir,
        rejected_dir=rejected_dir,
    )
    print_report(report, rejected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
