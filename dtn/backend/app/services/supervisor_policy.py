"""Pure decision contract for the DFL supervisor agent.

The supervisor asks the on-device SLM to choose only the *next high-level step* from a
small, stage-filtered tool set. A measured benchmark on the master Jetson showed that
offering all 14 coordinator tools made Qwen3.5-4B pick a read-only status tool every
time, while a compact set of progress steps produced a useful choice every time
(docs/evidence/agent-toolcall-benchmark-2026-11-17.json). Everything the model does not
need to decide (architecture JSON, dataset split, seeds, retries) is filled in by the
coordinator, so a decision is always small and schema-checkable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from common.pruning import PASS_THROUGH_LAYERS, PRUNABLE_LAYERS

DATASETS = ("mnist", "cifar10")
# Native image size of each supported dataset (preprocessing must not resize them).
DATASET_IMAGE_SIZES = {"mnist": 28, "cifar10": 32}
DATASET_SPLITS = ("stratified", "dirichlet_a01", "dirichlet_a03", "dirichlet_a05", "dirichlet_a07")
TOPOLOGIES = ("directed_ring", "bidirectional_ring", "mesh", "chain", "star", "learned")
AGGREGATIONS = ("fedavg", "uniform", "metropolis", "trimmed_mean")
SYNC_MODES = ("sync", "semi_sync")
FINISH_OUTCOMES = ("objective_achieved", "no_worthwhile_action", "blocked")

# Steps that start a measured experiment on the cluster, in the order the
# deterministic fallback policy tries them.
SEARCH_STEPS = ("run_nas", "run_hpo", "learn_topology", "optimize_topology")
EXPERIMENT_STEPS = (
    "baseline_training",
    "run_nas",
    "run_hpo",
    "learn_topology",
    "optimize_topology",
    "train_final",
    "prune_model",
)
# Steps that need a measured, trained model (and a custom layer architecture).
MODEL_STEPS = ("train_final", "prune_model")
# Measured on MNIST (8 workers): 50% pruning gave 3.9x fewer parameters but lost ~5
# points after a 5-round fine-tune; 30% keeps the model lighter with a smaller loss.
PRUNE_RATIO = 0.3
MAX_STEP_FAILURES = 2

STEP_TOOLS: dict[str, dict[str, Any]] = {
    "run_nas": {
        "description": (
            "Search a better small CNN on every worker's own data and validation split. "
            "Useful when the baseline model may be too small or too large."
        ),
        # The search budget comes from the operator's goal: a model-chosen budget was
        # too small for even two candidates and made every search time out.
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "run_hpo": {
        "description": (
            "Tune the learning rate and batch size of the best architecture so far on every "
            "worker's own data. Useful after NAS, or when training looks unstable or slow."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "prune_model": {
        "description": (
            "Remove the least important 30% of channels of the best trained model and "
            "fine-tune it on all workers. Makes the model smaller and faster on the Jetsons; "
            "useful once accuracy is good enough."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "learn_topology": {
        "description": (
            "Learn a new peer-to-peer topology with APG from measured peer-model losses and "
            "waiting times on the real workers, then train on it. Useful on non-IID data "
            "when peers learn slowly from each other; works for every dataset."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "optimize_topology": {
        "description": (
            "Measure candidate peer-to-peer topologies and select the best one by reward. "
            "Useful when peers learn slowly from each other on non-IID data (MNIST only)."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "train_final": {
        "description": (
            "Retrain with the best architecture and topology found so far. Useful after a "
            "search found something better than the baseline configuration."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rounds": {"type": "integer", "minimum": 1, "maximum": 20},
                "local_epochs": {"type": "integer", "minimum": 1, "maximum": 20},
                "topology": {"enum": list(TOPOLOGIES)},
                "aggregation": {
                    "enum": list(AGGREGATIONS),
                    "description": (
                        "fedavg weights peers by data size; uniform equally; metropolis "
                        "uses consensus weights; trimmed_mean ignores one outlying peer."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
    "finish_workflow": {
        "description": "End the workflow when the target is met or no listed step is worthwhile.",
        "parameters": {
            "type": "object",
            "properties": {
                "outcome": {"enum": list(FINISH_OUTCOMES)},
                "summary": {"type": "string", "minLength": 1, "maxLength": 400},
            },
            "required": ["outcome", "summary"],
            "additionalProperties": False,
        },
    },
}
# Every step accepts an optional one-sentence reason so each decision is explainable even
# when the model returns a tool call without assistant text.
REASON_PROPERTY = {"type": "string", "maxLength": 300}
for _tool in STEP_TOOLS.values():
    _tool["parameters"]["properties"].setdefault("reason", REASON_PROPERTY)
_VALIDATORS = {
    name: Draft202012Validator(tool["parameters"]) for name, tool in STEP_TOOLS.items()
}

SYSTEM_PROMPT = (
    "You are the supervisor agent of a decentralized federated learning (DFL) cluster of "
    "Jetson devices. Choose the single next step by calling exactly one provided tool. "
    "Use only the state JSON as evidence. Prefer the step most likely to raise the mean "
    "worker validation accuracy toward the target within the remaining budget. Call "
    "finish_workflow when no listed step is worthwhile. Put one short sentence explaining "
    "your choice in the tool's reason argument."
)


@dataclass(frozen=True)
class Decision:
    step: str
    arguments: dict[str, Any]
    rationale: str
    source: str  # "slm", "slm_repaired", "slm_text", "rule", or "fallback"


class DecisionError(ValueError):
    """The model's reply could not be turned into an allowed, schema-valid step."""


def default_goal(**overrides: Any) -> dict[str, Any]:
    goal = {
        "dataset": "mnist",
        "target_accuracy": 0.95,
        "max_experiments": 6,
        "time_budget_minutes": 480,
        "dataset_split": "dirichlet_a03",
        "rounds": 10,
        "local_epochs": 5,
        "topology": "directed_ring",
        "nas_budget_seconds": 600,
        "seed": 2026,
        "aggregation": "fedavg",
        "sync_mode": "semi_sync",
        "sync_deadline_seconds": 300,
    }
    goal.update(overrides)
    return goal


def new_state() -> dict[str, Any]:
    return {
        "completed_steps": [],
        "experiments": [],
        "best": None,
        "nas_architecture": None,
        "baseline_architecture": None,
        "last_trained_architecture": None,
        "pruning": None,
        "selected_topology": None,
        "trained_configs": [],
        "failures": {},
        "pending_step": None,
        "active": None,
        "counters": {"slm_decisions": 0, "slm_repairs": 0, "fallback_decisions": 0, "rule_decisions": 0},
    }


def experiments_used(state: dict[str, Any]) -> int:
    return sum(1 for item in state.get("experiments", []) if item.get("step") in EXPERIMENT_STEPS)


def config_key(architecture: dict[str, Any] | None, topology: str) -> str:
    architecture_id = "initial"
    if isinstance(architecture, dict):
        architecture_id = str(architecture.get("id") or architecture.get("model_id") or "searched")
    return f"{architecture_id}|{topology}"


def final_training_config(goal: dict[str, Any], state: dict[str, Any], topology: str | None = None) -> str:
    chosen_topology = topology or state.get("selected_topology") or goal["topology"]
    return config_key(state.get("nas_architecture"), chosen_topology)


def allowed_steps(goal: dict[str, Any], state: dict[str, Any]) -> list[str]:
    """Steps the SLM may choose now; finish_workflow is always allowed."""
    completed = set(state.get("completed_steps", []))
    failures = state.get("failures", {})
    steps: list[str] = []
    if experiments_used(state) < int(goal["max_experiments"]):
        if "run_nas" not in completed and failures.get("run_nas", 0) < MAX_STEP_FAILURES:
            steps.append("run_nas")
        if (
            goal["dataset"] == "mnist"
            and "optimize_topology" not in completed
            and failures.get("optimize_topology", 0) < MAX_STEP_FAILURES
        ):
            steps.append("optimize_topology")
        if "learn_topology" not in completed and failures.get("learn_topology", 0) < MAX_STEP_FAILURES:
            steps.append("learn_topology")
        if (
            "run_hpo" not in completed
            and failures.get("run_hpo", 0) < MAX_STEP_FAILURES
            and is_custom_architecture(tuning_architecture(state))
        ):
            steps.append("run_hpo")
        has_improvement = bool(state.get("nas_architecture") or state.get("selected_topology"))
        untried = final_training_config(goal, state) not in state.get("trained_configs", [])
        # One explicit final retrain is always worth offering (the SLM may ask for more
        # rounds), even when a search step already trained the same configuration.
        never_retrained = "train_final" not in completed
        if has_improvement and (untried or never_retrained) and failures.get("train_final", 0) < MAX_STEP_FAILURES:
            steps.append("train_final")
        # Compress only a good model: when the target is met, or nothing else can help.
        best = (state.get("best") or {}).get("validation_accuracy")
        target_met = isinstance(best, (int, float)) and best >= float(goal["target_accuracy"])
        if can_prune(state) and (target_met or not steps):
            steps.append("prune_model")
    steps.append("finish_workflow")
    return steps


def tuning_architecture(state: dict[str, Any]) -> dict[str, Any] | None:
    """The architecture HPO tunes: the NAS result if any, else the deployed baseline."""
    return state.get("nas_architecture") or state.get("baseline_architecture")


def is_custom_architecture(architecture: Any) -> bool:
    return isinstance(architecture, dict) and architecture.get("model_type") == "custom"


def can_prune(state: dict[str, Any]) -> bool:
    """Prune once, after a completed training, if the trained model is a prunable CNN."""
    completed = set(state.get("completed_steps", []))
    if "prune_model" in completed or state.get("failures", {}).get("prune_model", 0) >= MAX_STEP_FAILURES:
        return False
    trained = [item for item in state.get("experiments", []) if item.get("kind") == "decentralized" and item.get("status") == "completed"]
    if not trained:
        return False
    # Workers keep only their newest model, so pruning compresses the newest training;
    # only do that when it is also the best model of the run.
    best_group = (state.get("best") or {}).get("group_id")
    if best_group is not None and trained[-1].get("group_id") != best_group:
        return False
    # The newest training is what workers hold; only plain CNN/MLP layers can be pruned.
    architecture = state.get("last_trained_architecture")
    if not is_custom_architecture(architecture):
        return False
    layer_types = {layer.get("type") for layer in architecture.get("layers", []) if isinstance(layer, dict)}
    return bool(layer_types) and layer_types <= PRUNABLE_LAYERS | PASS_THROUGH_LAYERS


def tool_definitions(steps: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": STEP_TOOLS[name]["description"],
                "parameters": STEP_TOOLS[name]["parameters"],
            },
        }
        for name in steps
    ]


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as error:
            raise DecisionError("Tool arguments are not valid JSON") from error
    if not isinstance(raw, dict):
        raise DecisionError("Tool arguments must be a JSON object")
    coerced: dict[str, Any] = {}
    for key, value in raw.items():
        # Small models sometimes quote integers ("5"); accept exact integer strings only.
        if isinstance(value, str) and re.fullmatch(r"-?[0-9]{1,9}", value.strip()):
            coerced[key] = int(value)
        else:
            coerced[key] = value
    return coerced


def _clamp_integers(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Clamp out-of-range integers to the tool bounds (e.g. rounds 50 -> 20).

    A small model asking for "more rounds than allowed" means "as many as allowed";
    rejecting it wasted the repair attempt and ended runs early.
    """
    properties = STEP_TOOLS.get(name, {}).get("parameters", {}).get("properties", {})
    # Drop arguments the tool does not declare (e.g. a stale budget_seconds habit).
    clamped = {key: value for key, value in arguments.items() if key in properties or key == "reason"}
    if isinstance(clamped.get("reason"), str):
        # A long explanation is still a valid decision; keep it, shortened.
        clamped["reason"] = " ".join(clamped["reason"].split())[: REASON_PROPERTY["maxLength"]]
    for key, value in list(clamped.items()):
        schema = properties.get(key)
        if not isinstance(schema, dict) or schema.get("type") != "integer":
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if "minimum" in schema:
            value = max(int(schema["minimum"]), value)
        if "maximum" in schema:
            value = min(int(schema["maximum"]), value)
        clamped[key] = value
    return clamped


def validate_step(name: Any, arguments: Any, allowed: list[str]) -> dict[str, Any]:
    if not isinstance(name, str) or name not in allowed:
        raise DecisionError(f"Step {name!r} is not one of the allowed steps: {', '.join(allowed)}")
    parsed = _clamp_integers(name, _coerce_arguments(arguments))
    try:
        _VALIDATORS[name].validate(parsed)
    except ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path) or "arguments"
        raise DecisionError(f"Invalid {name} arguments at {location}: {error.message}") from error
    return parsed


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _text_tool_call(content: str) -> tuple[str, Any] | None:
    """Recover a tool call the model wrote as text instead of a native tool call."""
    match = _JSON_OBJECT.search(content)
    if match is None:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("function"), dict):
        value = value["function"]
    name = value.get("name")
    if not isinstance(name, str):
        return None
    return name, value.get("arguments", value.get("parameters", {}))


def _rationale(content: Any) -> str:
    if not isinstance(content, str):
        return ""
    text = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
    text = _JSON_OBJECT.sub("", text)
    return " ".join(text.split())[:500]


def _pop_reason(arguments: dict[str, Any], content: str) -> str:
    reason = arguments.pop("reason", None)
    if isinstance(reason, str) and reason.strip():
        return " ".join(reason.split())[:300]
    return _rationale(content)


def parse_decision(message: dict[str, Any], allowed: list[str]) -> Decision:
    """Turn one model message into an allowed, validated step or raise DecisionError."""
    if not isinstance(message, dict):
        raise DecisionError("The model returned no message")
    content = message.get("content") if isinstance(message.get("content"), str) else ""
    calls = message.get("tool_calls")
    if isinstance(calls, list) and calls:
        first = calls[0] if isinstance(calls[0], dict) else {}
        function = first.get("function") if isinstance(first.get("function"), dict) else {}
        arguments = validate_step(function.get("name"), function.get("arguments"), allowed)
        return Decision(function["name"], arguments, _pop_reason(arguments, content), "slm")
    recovered = _text_tool_call(content)
    if recovered is None:
        raise DecisionError("The model replied in prose without calling one of the allowed tools")
    arguments = validate_step(recovered[0], recovered[1], allowed)
    return Decision(recovered[0], arguments, _pop_reason(arguments, content), "slm_text")


def rule_decision(goal: dict[str, Any], state: dict[str, Any], minutes_left: float) -> Decision | None:
    """Deterministic stop rules that never need the language model."""
    best = state.get("best") or {}
    accuracy = best.get("validation_accuracy")
    target_met = isinstance(accuracy, (int, float)) and accuracy >= float(goal["target_accuracy"])
    if target_met and minutes_left > 0 and "prune_model" in allowed_steps(goal, state):
        return Decision(
            "prune_model",
            {},
            "The target is met; compress the model for the Jetsons before finishing.",
            "rule",
        )
    if target_met:
        return Decision(
            "finish_workflow",
            {
                "outcome": "objective_achieved",
                "summary": (
                    f"Mean worker validation accuracy {accuracy:.4f} meets the target "
                    f"{float(goal['target_accuracy']):.4f}."
                ),
            },
            "The measured result meets the goal.",
            "rule",
        )
    if minutes_left <= 0:
        return Decision(
            "finish_workflow",
            {"outcome": "blocked", "summary": "The workflow time budget is exhausted."},
            "No time budget remains.",
            "rule",
        )
    if allowed_steps(goal, state) == ["finish_workflow"]:
        return Decision(
            "finish_workflow",
            {
                "outcome": "no_worthwhile_action",
                "summary": "Every available improvement step is done, failed twice, or out of budget.",
            },
            "No further step is available.",
            "rule",
        )
    return None


def fallback_decision(goal: dict[str, Any], state: dict[str, Any]) -> Decision:
    """Used when the SLM is unreachable or keeps returning invalid steps."""
    allowed = allowed_steps(goal, state)
    for step in ("run_nas", "run_hpo", "learn_topology", "optimize_topology", "train_final", "prune_model"):
        if step in allowed:
            return Decision(
                step, {}, "Default improvement order: NAS, HPO, learned topology, topology sweep, retrain, prune.", "fallback"
            )
    return Decision(
        "finish_workflow",
        {"outcome": "no_worthwhile_action", "summary": "No improvement step remains."},
        "No improvement step remains.",
        "fallback",
    )


def prompt_state(goal: dict[str, Any], state: dict[str, Any], cluster: dict[str, Any], minutes_left: float) -> dict[str, Any]:
    """Compact evidence for the decision prompt (kept well under 1,500 characters)."""
    recent = [
        {
            key: item.get(key)
            for key in ("step", "status", "validation_accuracy", "macro_f1", "architecture", "topology", "aggregation", "partial_rounds", "error")
            if item.get(key) is not None
        }
        for item in state.get("experiments", [])[-3:]
    ]
    for item in recent:
        if isinstance(item.get("error"), str):
            item["error"] = item["error"][:160]
    return {
        "goal": {
            "dataset": goal["dataset"],
            "target_validation_accuracy": goal["target_accuracy"],
            "experiments_left": int(goal["max_experiments"]) - experiments_used(state),
            "minutes_left": int(max(0, minutes_left)),
        },
        "completed_steps": state.get("completed_steps", []),
        "best": state.get("best"),
        "recent_experiments": recent,
        "nas_architecture_found": bool(state.get("nas_architecture")),
        "pruned_model": (state.get("pruning") or {}).get("compression"),
        "selected_topology": state.get("selected_topology"),
        "cluster": cluster,
    }


def decision_messages(state_json: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
            + "\nState: "
            + json.dumps(state_json, ensure_ascii=False, separators=(",", ":"), default=str),
        },
        {"role": "user", "content": "Choose the next step."},
    ]


TRANSIENT_ERROR = re.compile(
    r"timed? ?out|timeout|memory|oom|resource exhausted|connection|unreachable|"
    r"reset by peer|refused|temporarily|busy|503|502",
    re.IGNORECASE,
)


def is_transient_failure(error: str | None) -> bool:
    return bool(error) and bool(TRANSIENT_ERROR.search(error or ""))


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def summarize_group(group: dict[str, Any]) -> dict[str, Any]:
    """Extract the measured outcome of a finished master experiment group."""
    summary: dict[str, Any] = {
        "group_id": group.get("group_id"),
        "kind": group.get("kind"),
        "status": group.get("status"),
    }
    if group.get("error"):
        summary["error"] = str(group["error"])[:500]
    kind = group.get("kind")
    if kind == "decentralized":
        runs = [run for run in group.get("runs", []) if isinstance(run, dict) and run.get("rounds")]
        if runs:
            final_round = runs[-1]["rounds"][-1]
            workers = final_round.get("workers", {}) if isinstance(final_round, dict) else {}
            accuracies = [
                float(metrics["validation_accuracy"])
                for metrics in workers.values()
                if isinstance(metrics, dict) and isinstance(metrics.get("validation_accuracy"), (int, float))
            ]
            f1_scores = [
                float(metrics["validation_macro_f1"])
                for metrics in workers.values()
                if isinstance(metrics, dict) and isinstance(metrics.get("validation_macro_f1"), (int, float))
            ]
            summary["validation_accuracy"] = _mean(accuracies)
            summary["macro_f1"] = _mean(f1_scores)
            summary["rounds_completed"] = len(runs[-1]["rounds"])
            summary["worker_count"] = len(workers)
        summary["topology"] = group.get("topology")
        summary["aggregation"] = group.get("aggregation", "fedavg")
        summary["sync_mode"] = group.get("sync_mode", "sync")
        partial_rounds = 0
        for run in runs:
            for round_row in run.get("rounds", []):
                workers = round_row.get("workers", {}) if isinstance(round_row, dict) else {}
                if any(isinstance(row, dict) and row.get("missing_peers") for row in workers.values()):
                    partial_rounds += 1
        summary["partial_rounds"] = partial_rounds
        architecture = group.get("architecture")
        if isinstance(architecture, dict):
            summary["architecture"] = architecture.get("id") or architecture.get("model_id")
            summary["parameter_count"] = architecture.get("parameter_count")
        learning = group.get("topology_learning")
        if isinstance(learning, dict):
            summary["topology_learning"] = {
                key: learning.get(key)
                for key in ("edges", "mesh_edges", "ring_added_for_connectivity", "iterations", "objective", "senders")
            }
    elif kind == "dataset_architecture_search":
        architecture = group.get("selected_architecture")
        if isinstance(architecture, dict):
            summary["selected_architecture"] = architecture
            summary["architecture"] = architecture.get("id") or architecture.get("model_id")
            summary["parameter_count"] = group.get("selected_parameter_count")
        selection = group.get("selection")
        candidate = selection.get("selected_candidate") if isinstance(selection, dict) else None
        if isinstance(candidate, dict):
            for key in ("mean_validation_accuracy", "validation_accuracy"):
                if isinstance(candidate.get(key), (int, float)):
                    summary["search_validation_accuracy"] = float(candidate[key])
                    break
    elif kind == "hyperparameter_search":
        architecture = group.get("selected_architecture")
        if isinstance(architecture, dict):
            summary["selected_architecture"] = architecture
            summary["architecture"] = architecture.get("id")
            summary["parameter_count"] = architecture.get("parameter_count")
        if isinstance(group.get("selected_hyperparameters"), dict):
            summary["hyperparameters"] = group["selected_hyperparameters"]
        selection = group.get("selection")
        candidate = selection.get("selected_candidate") if isinstance(selection, dict) else None
        if isinstance(candidate, dict) and isinstance(candidate.get("validation_accuracy"), (int, float)):
            summary["search_validation_accuracy"] = float(candidate["validation_accuracy"])
    elif kind == "topology_optimization":
        selected = group.get("selected_configuration")
        if isinstance(selected, dict):
            summary["selected_topology"] = selected.get("topology")
            summary["topology"] = selected.get("topology")
    return summary


def group_progress(group: dict[str, Any]) -> float | None:
    if group.get("kind") != "decentralized":
        return None
    rounds = group.get("rounds")
    runs = group.get("runs") or []
    if not isinstance(rounds, int) or rounds <= 0 or not runs:
        return None
    completed = len(runs[-1].get("rounds", [])) if isinstance(runs[-1], dict) else 0
    return round(min(1.0, completed / rounds), 3)
