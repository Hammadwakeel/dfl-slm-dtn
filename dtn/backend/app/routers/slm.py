"""Authenticated coordinator proxy for the SLM master service."""

import json
import logging
import math
import re
from datetime import datetime, timezone
from uuid import UUID, uuid4
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request as UrlRequest, build_opener

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from common.agent_tools import agent_tool_action
from app.core.config import Settings
from app.core.security import require_api_token
from app.db.models import Device, SlmConversation, SlmMessage
from app.routers.autonomy import get_autonomy
from app.services.autonomy import (
    ACTION_SCHEMA_DOCUMENT,
    ACTION_VALIDATOR,
    AutonomyController,
    AutonomyError,
    validate_action,
)
from app.services.datasets import DatasetError
from app.services.feedback import FeedbackSummaryError, build_training_feedback_summary
from app.services.langgraph_agent import run_agent

router = APIRouter(
    prefix="/api/slm",
    tags=["slm-assistant"],
    dependencies=[Depends(require_api_token)],
)
LOGGER = logging.getLogger(__name__)
AGENT_HISTORY_CHAR_LIMIT = 3000
AGENT_CONTEXT_CHAR_LIMIT = 4_000
FEEDBACK_ACTIONS = {
    "enable_nas",
    "set_epochs",
    "set_rounds",
    "set_topology",
    "start_training",
    "stop_training",
}
FEEDBACK_RETRIES = 2
ACTION_HISTORY_EXCLUDED = frozenset(
    {
        "get_status",
        "get_logs",
        "get_results",
        "compare_runs",
        "suggest_next_action",
        "clarify",
        "refuse",
        "redirect",
    }
)
NEXT_ACTION_PATTERNS = (
    r"\bwhat\s+(?:action|step)\s+should\s+i\s+(?:perform|take|do)\s+next\b",
    r"\bwhat\s+(?:is|['’]s)\s+(?:the\s+)?next\s+(?:action|step)\b",
    r"\bwhat\s+(?:should|can)\s+i\s+do\s+(?:next|now)\b",
    r"\bnow\s+what\s+(?:should|can)\s+i\s+do\b",
    r"\bwhat\s+(?:is|['’]s)\s+next\b",
    r"\bwhat\s+next\b",
    r"\bwhat\s+action\s+next\b",
    r"\b(?:suggest|recommend|advise)\s+(?:me\s+)?(?:the\s+)?next\s+(?:action|step)\b",
    r"\btraining\b.*\b(?:complete|completed|finished|done)\b.*\b(?:what\s+(?:should|can)\s+i\s+do|what\s+next|next\s+step)\b",
    r"\bwhat\s+(?:should|can)\s+i\s+do\b.*\bafter\s+(?:the\s+)?training\b",
)


def _is_next_action_request(message: str) -> bool:
    normalized = re.sub(r"\s+", " ", message.strip().lower())
    return any(re.search(pattern, normalized) for pattern in NEXT_ACTION_PATTERNS)


def _utc_isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: UUID | None = None
    workflow_dataset: str | None = Field(default=None, pattern=r"^(mnist|cifar10)$")
    workflow_id: UUID | None = None
    workflow_previous_group_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9._-]{1,100}$",
    )


class BaselineRequest(BaseModel):
    rounds: int | None = Field(default=None, ge=1, le=100)
    epochs: int | None = Field(default=None, ge=1, le=50)


class FeedbackRequest(BaseModel):
    group_id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,100}$")
    iteration: int = Field(ge=1, le=100)
    previous_accuracy: float | None = Field(default=None, ge=0, le=1)


def _save_user_message(
    request: Request,
    conversation_id: UUID | None,
    message: str,
    *,
    mode: str = "human",
    workflow_id: UUID | None = None,
) -> str:
    conversation_id_text = str(conversation_id) if conversation_id else None
    timestamp = datetime.now(timezone.utc)
    with request.app.state.session_factory.begin() as session:
        conversation = (
            session.get(SlmConversation, conversation_id_text)
            if conversation_id_text
            else None
        )
        if conversation_id_text and conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conversation is not None and conversation.mode != mode:
            raise HTTPException(
                status_code=409,
                detail="A conversation cannot switch between human and agent modes",
            )
        workflow_id_text = str(workflow_id) if workflow_id else None
        if (
            conversation is not None
            and workflow_id_text
            and conversation.workflow_id
            and conversation.workflow_id != workflow_id_text
        ):
            raise HTTPException(
                status_code=409,
                detail="A conversation cannot be moved to another workflow run",
            )
        if conversation is None:
            first_line = (message.strip().splitlines() or ["New conversation"])[0]
            title = first_line[:117].strip()
            if len(first_line) > 117:
                title += "..."
            conversation = SlmConversation(
                id=str(uuid4()),
                title=title or "New conversation",
                mode=mode,
                workflow_id=workflow_id_text,
                created_at=timestamp,
                updated_at=timestamp,
            )
            session.add(conversation)
        elif workflow_id_text and conversation.workflow_id is None:
            conversation.workflow_id = workflow_id_text
        conversation.updated_at = timestamp
        session.add(
            SlmMessage(
                conversation_id=conversation.id,
                role="user",
                content=message,
                created_at=timestamp,
            )
        )
        resolved_id = conversation.id
    return resolved_id


def _save_assistant_message(
    request: Request,
    conversation_id: str,
    content: str,
    response_data: dict[str, object],
) -> None:
    timestamp = datetime.now(timezone.utc)
    with request.app.state.session_factory.begin() as session:
        conversation = session.get(SlmConversation, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation.updated_at = timestamp
        session.add(
            SlmMessage(
                conversation_id=conversation_id,
                role="assistant",
                content=content,
                response_data={
                    **response_data,
                    "conversation_id": conversation_id,
                },
                created_at=timestamp,
            )
        )


def _compact_agent_dataset(metadata: dict[str, object]) -> dict[str, object]:
    fields = (
        "dataset_name",
        "classes",
        "class_counts",
        "image_count",
        "source",
    )
    return {key: metadata[key] for key in fields if key in metadata}


def _compact_agent_preprocessing_report(
    report: dict[str, object],
    *,
    workflow_bound: bool = False,
) -> dict[str, object]:
    fields = (
        "dataset_name",
        "classes",
        "image_count",
        "image_shape",
        "split_strategy",
        "official_test_set_preserved",
        "split_sizes",
        "split_indices_disjoint",
        "worker_shards_disjoint",
        "seed",
    )
    if not workflow_bound:
        fields = (*fields, "normalization", "created_at")
    summary = {key: report[key] for key in fields if key in report}
    workers = report.get("workers")
    if isinstance(workers, dict) and not workflow_bound:
        summary["workers"] = {
            worker_id: {
                key: worker[key]
                for key in ("rows",)
                if key in worker
            }
            for worker_id, worker in workers.items()
            if isinstance(worker, dict)
        }
    uploaded_workers = report.get("uploaded_workers")
    if isinstance(uploaded_workers, dict) and not workflow_bound:
        summary["uploaded_workers"] = sorted(uploaded_workers)
    return summary


def _compact_agent_action_history(
    actions: list[dict[str, object]],
) -> list[dict[str, object]]:
    compacted = []
    for action in actions:
        details = action.get("details")
        details = details if isinstance(details, dict) else {}
        summary: dict[str, object] = {
            key: value
            for key, value in {
                "action": details.get("action", action.get("action")),
                "mode": details.get("mode", action.get("mode")),
                "result_status": details.get(
                    "result_status",
                    action.get("result_status"),
                ),
                "dataset_name": details.get(
                    "dataset_name",
                    action.get("dataset_name"),
                ),
                "run_id": details.get("run_id"),
                "timestamp": action.get("timestamp", action.get("created_at")),
            }.items()
            if value is not None
        }
        reason = details.get("reason")
        if isinstance(reason, str) and reason:
            summary["reason"] = reason[:200]
        parameters = details.get("parameters")
        if isinstance(parameters, dict):
            relevant_parameters = {
                key: parameters[key]
                for key in (
                    "dataset_name",
                    "model_name",
                    "rounds",
                    "epochs",
                    "topology",
                    "enabled",
                    "dataset_split",
                    "search_time_limit_seconds",
                )
                if key in parameters
            }
            if relevant_parameters:
                summary["parameters"] = relevant_parameters
        result = details.get("result")
        if isinstance(result, dict):
            summary["result"] = _compact_agent_tool_value(result)
        compacted.append(summary)
    return compacted


def _bound_agent_context(context: dict[str, object]) -> dict[str, object]:
    bounded = dict(context)
    actions = bounded.get("previous_actions")
    retained_actions = list(actions) if isinstance(actions, list) else []
    while True:
        bounded["previous_actions"] = retained_actions
        size = len(
            json.dumps(
                bounded,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )
        if size <= AGENT_CONTEXT_CHAR_LIMIT:
            return bounded
        if not retained_actions:
            raise HTTPException(
                status_code=413,
                detail="The compact Agent system snapshot exceeds its safe size limit.",
            )
        retained_actions.pop()


def _compact_agent_tool_value(value: object, depth: int = 0) -> object:
    if isinstance(value, dict):
        if depth >= 3:
            return {
                key: item
                for key, item in value.items()
                if isinstance(item, (str, int, float, bool)) or item is None
            }
        compacted: dict[str, object] = {}
        for key, item in value.items():
            if key in {
                "evidence",
                "local_model_path",
                "architecture_sha256",
                "weights_sha256",
                "worker_payload",
                "confusion_matrix",
                "confusion_matrix_counts",
                "confusion_matrix_row_normalized_percent",
                "topologies",
                "candidates",
            }:
                continue
            if key == "groups" and isinstance(item, list):
                compacted[key] = [
                    _compact_agent_experiment_group(group)
                    for group in item[-5:]
                    if isinstance(group, dict)
                ]
                compacted["group_count"] = len(item)
                continue
            if key == "configuration" and isinstance(item, dict):
                compacted[key] = {
                    name: item[name]
                    for name in ("rounds", "epochs", "topology", "dataset_split")
                    if name in item
                }
                continue
            if key == "workers" and isinstance(item, dict):
                compacted[key] = sorted(item)
                continue
            if key == "architecture" and isinstance(item, dict):
                compacted[key] = {
                    name: item[name]
                    for name in (
                        "model_type",
                        "id",
                        "design",
                        "parameter_count",
                        "input_shape",
                        "num_classes",
                    )
                    if name in item
                }
                continue
            compacted[key] = _compact_agent_tool_value(item, depth + 1)
            if len(compacted) >= 24:
                break
        return compacted
    if isinstance(value, list):
        return [_compact_agent_tool_value(item, depth + 1) for item in value[:8]]
    if isinstance(value, str):
        return value[:160]
    return value


def _compact_agent_experiment_group(group: dict[str, object]) -> dict[str, object]:
    is_topology_optimization = group.get("kind") == "topology_optimization"
    fields = (
        "group_id",
        "kind",
        "dataset_name",
        "status",
        "rounds",
        "epochs",
        "topology",
        "dataset_split",
        "error",
        "architecture",
        "selected_architecture",
        "selected_configuration",
        "nas_beats_fixed",
        "comparison",
        "search_elapsed_seconds",
        "search_peak_rss_mb",
    )
    summary = {
        key: (
            _compact_agent_tool_value(group[key])
            if key in {
                "architecture",
                "comparison",
                "selected_architecture",
                "selected_configuration",
            }
            else group[key]
        )
        for key in fields
        if key in group
    }
    runs = group.get("runs")
    if isinstance(runs, list) and not is_topology_optimization:
        run_summaries: list[dict[str, object]] = []
        for run in runs[-2:]:
            if not isinstance(run, dict):
                continue
            run_summary: dict[str, object] = {
                key: run[key]
                for key in ("run_id", "seed", "status", "error", "dataset_split")
                if key in run
            }
            rounds = run.get("rounds")
            if isinstance(rounds, list) and rounds:
                latest = rounds[-1]
                if isinstance(latest, dict):
                    workers = latest.get("workers")
                    if isinstance(workers, dict):
                        run_summary["latest_round"] = latest.get("round")
                        run_summary["workers"] = {
                            str(node_id): {
                                key: metrics[key]
                                for key in (
                                    "status",
                                    "training_loss",
                                    "validation_accuracy",
                                    "validation_loss",
                                )
                                if key in metrics
                            }
                            for node_id, metrics in workers.items()
                            if isinstance(metrics, dict)
                        }
            run_summaries.append(run_summary)
        summary["runs"] = run_summaries
    for key in ("candidates", "nas_candidates"):
        candidates = group.get(key)
        if isinstance(candidates, list):
            if is_topology_optimization and key == "candidates":
                summary[key] = [
                    {
                        field: candidate[field]
                        for field in (
                            "topology",
                            "alpha",
                            "completed_runs",
                            "mean_final_accuracy",
                            "mean_reward",
                            "mean_components",
                        )
                        if field in candidate
                    }
                    for candidate in candidates
                    if isinstance(candidate, dict)
                ]
                selected = group.get("selected_configuration")
                if isinstance(selected, dict):
                    selected_accuracy = next(
                        (
                            candidate.get("mean_final_accuracy")
                            for candidate in candidates
                            if isinstance(candidate, dict)
                            and candidate.get("topology") == selected.get("topology")
                            and candidate.get("alpha") == selected.get("alpha")
                        ),
                        None,
                    )
                    if isinstance(selected_accuracy, (int, float)):
                        summary["selected_validation_accuracy"] = selected_accuracy
                continue
            summary[key] = [
                _compact_agent_tool_value(candidate)
                for candidate in candidates[:3]
                if isinstance(candidate, dict)
            ]
    return summary


def _repeats_completed_training(
    parameters: dict[str, object],
    previous_group: dict[str, object],
    configuration: dict[str, object],
) -> bool:
    if previous_group.get("kind") != "decentralized":
        return False
    previous_architecture = previous_group.get("architecture")
    current_architecture = parameters.get(
        "architecture",
        configuration.get("architecture"),
    )
    if (
        not isinstance(previous_architecture, dict)
        or not isinstance(current_architecture, dict)
        or previous_architecture.get("id") != current_architecture.get("id")
    ):
        return False
    return (
        parameters.get("rounds") == previous_group.get("rounds")
        and parameters.get("epochs") == previous_group.get("epochs")
        and parameters.get(
            "dataset_split",
            configuration.get("dataset_split"),
        )
        == previous_group.get("dataset_split")
        and configuration.get("topology") == previous_group.get("topology")
    )


def _bound_agent_chat_history(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    bounded: list[dict[str, str]] = []
    remaining = AGENT_HISTORY_CHAR_LIMIT
    for message in reversed(messages):
        if remaining <= 0:
            break
        content = message["content"]
        content = content[-remaining:]
        if content:
            bounded.append({"role": message["role"], "content": content})
            remaining -= len(content)
    return list(reversed(bounded))


def _filter_agent_action_history(
    actions: list[dict[str, object]],
    *,
    workflow_dataset: str | None,
    workflow_id: str | None,
) -> list[dict[str, object]]:
    filtered = actions
    if workflow_dataset:
        filtered = [
            action
            for action in filtered
            if isinstance(action.get("details"), dict)
            and (
                action["details"].get("dataset_name")
                or (
                    isinstance(action["details"].get("parameters"), dict)
                    and action["details"]["parameters"].get("dataset_name")
                )
            )
            == workflow_dataset
        ]
    if workflow_id:
        filtered = [
            action
            for action in filtered
            if isinstance(action.get("details"), dict)
            and isinstance(action["details"].get("evidence"), dict)
            and action["details"]["evidence"].get("workflow_id") == workflow_id
        ]
    return filtered


def _agent_context(
    request: Request,
    settings: Settings,
    controller: AutonomyController,
    *,
    workflow_dataset: str | None = None,
    workflow_id: str | None = None,
) -> dict[str, object]:
    with request.app.state.session_factory() as session:
        devices = session.scalars(
            select(Device).order_by(Device.role, Device.node_id)
        ).all()
        node_data: list[dict[str, object]] = []
        for device in devices:
            latest_status = (
                device.latest_status if isinstance(device.latest_status, dict) else {}
            )
            memory = latest_status.get("memory")
            memory = memory if isinstance(memory, dict) else {}
            temperatures = latest_status.get("temperatures_celsius")
            temperatures = temperatures if isinstance(temperatures, dict) else {}
            numeric_temperatures = [
                value
                for value in temperatures.values()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            ]
            memory_summary = {
                key: round(float(memory[source]) / (1024 * 1024), 1)
                for key, source in (
                    ("available_mb", "available_bytes"),
                    ("total_mb", "total_bytes"),
                )
                if isinstance(memory.get(source), (int, float))
                and math.isfinite(float(memory[source]))
                and memory[source] >= 0
            }
            compact_status = {
                key: latest_status[key]
                for key in ("cpu_utilization_percent", "gpu_utilization_percent")
                if key in latest_status
            }
            compact_status.update(memory_summary)
            if numeric_temperatures:
                compact_status["max_temperature_celsius"] = round(
                    max(float(value) for value in numeric_temperatures),
                    1,
                )
            node_summary: dict[str, object] = {
                "node_id": device.node_id,
                "role": device.role,
                "online": device.online,
                "training_ready": device.training_ready,
            }
            if device.last_error:
                node_summary["last_error"] = device.last_error[:200]
            if workflow_dataset:
                available_memory = memory_summary.get("available_mb")
                if available_memory is not None:
                    node_summary["available_memory_mb"] = available_memory
            else:
                node_summary["latest_status"] = compact_status
            node_data.append(node_summary)

    datasets = [
        _compact_agent_dataset(metadata)
        for metadata in request.app.state.datasets.list_metadata()
        if isinstance(metadata, dict)
        and (
            workflow_dataset is None
            or metadata.get("dataset_name") == workflow_dataset
        )
    ]
    preprocessing: dict[str, object] = {}
    for metadata in datasets:
        name = metadata.get("dataset_name")
        if not isinstance(name, str):
            continue
        try:
            preprocessing[name] = _compact_agent_preprocessing_report(
                request.app.state.datasets.get_latest_preprocessing_report(name),
                workflow_bound=workflow_dataset is not None,
            )
        except DatasetError as error:
            if error.status_code != 404:
                raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    results = _forward(settings, "/results")
    groups = results.get("groups")
    if isinstance(groups, list) and workflow_dataset:
        groups = [
            group
            for group in groups
            if isinstance(group, dict)
            and group.get("dataset_name") == workflow_dataset
        ]
    if isinstance(groups, list) and workflow_dataset:
        recent_groups = groups[-1:]
        previous_completed_training = next(
            (
                group
                for group in reversed(groups[:-1])
                if isinstance(group, dict)
                and group.get("kind") == "decentralized"
                and group.get("status") == "completed"
            ),
            None,
        )
        if previous_completed_training is not None:
            recent_groups = [previous_completed_training, *recent_groups]
    elif isinstance(groups, list):
        recent_groups = groups[-2:]
    else:
        recent_groups = []
    experiment_state = {
        "configuration": results.get("configuration"),
        "active_group_id": results.get("active_group_id"),
        "groups": [
            _compact_agent_experiment_group(group)
            for group in recent_groups
            if isinstance(group, dict)
        ],
    }
    action_history = _filter_agent_action_history(
        controller.history(),
        workflow_dataset=workflow_dataset,
        workflow_id=workflow_id,
    )
    history_limit = (
        min(settings.agent_action_history_limit, 5)
        if workflow_dataset
        else settings.agent_action_history_limit
    )
    recent_actions = _compact_agent_action_history(
        action_history[:history_limit]
    )
    return _bound_agent_context(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "nodes": node_data,
            "datasets": datasets,
            "preprocessing_reports": preprocessing,
            "experiments": experiment_state,
            "global_stop_requested": controller.state()["stop_requested"],
            "previous_actions": recent_actions,
        }
    )


def _agent_chat_history(
    request: Request,
    conversation_id: str,
    limit: int,
) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    with request.app.state.session_factory() as session:
        messages = session.scalars(
            select(SlmMessage)
            .where(SlmMessage.conversation_id == conversation_id)
            .order_by(SlmMessage.id.desc())
            .limit(limit + 1)
        ).all()
    ordered = list(reversed(messages))
    if ordered:
        ordered.pop()
    history: list[dict[str, str]] = []
    for item in ordered:
        if item.role not in {"user", "assistant"}:
            continue
        content = item.content
        if item.role == "assistant" and isinstance(item.response_data, dict):
            trace = item.response_data.get("trace")
            if isinstance(trace, list) and trace:
                content += "\nObserved tool decisions and results: " + json.dumps(
                    _compact_agent_tool_value(trace),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
        history.append({"role": item.role, "content": content})
    return _bound_agent_chat_history(
        history
    )


@router.get("/conversations")
def list_conversations(
    request: Request,
) -> list[dict[str, object]]:
    with request.app.state.session_factory() as session:
        conversations = session.scalars(
            select(SlmConversation).order_by(
                SlmConversation.updated_at.desc(),
                SlmConversation.id.desc(),
            )
        ).all()
        counts = dict(
            session.execute(
                select(SlmMessage.conversation_id, func.count(SlmMessage.id))
                .group_by(SlmMessage.conversation_id)
            ).all()
        )
        return [
            {
                "id": conversation.id,
                "title": conversation.title,
                "mode": conversation.mode,
                "workflow_id": conversation.workflow_id,
                "updated_at": _utc_isoformat(conversation.updated_at),
                "message_count": counts.get(conversation.id, 0),
            }
            for conversation in conversations
        ]


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: UUID,
    request: Request,
) -> dict[str, object]:
    with request.app.state.session_factory() as session:
        conversation = session.get(SlmConversation, str(conversation_id))
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        messages = session.scalars(
            select(SlmMessage)
            .where(SlmMessage.conversation_id == conversation.id)
            .order_by(SlmMessage.id)
        ).all()
        return {
            "id": conversation.id,
            "title": conversation.title,
            "mode": conversation.mode,
            "workflow_id": conversation.workflow_id,
            "created_at": _utc_isoformat(conversation.created_at),
            "updated_at": _utc_isoformat(conversation.updated_at),
            "messages": [
                {
                    "id": message.id,
                    "role": message.role,
                    "content": message.content,
                    "response_data": message.response_data,
                    "created_at": _utc_isoformat(message.created_at),
                }
                for message in messages
            ],
        }


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: UUID,
    request: Request,
) -> None:
    with request.app.state.session_factory.begin() as session:
        conversation = session.get(SlmConversation, str(conversation_id))
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        session.delete(conversation)


def _feedback_action_schema(name: str) -> dict[str, object]:
    actions = ACTION_SCHEMA_DOCUMENT["$defs"]["action"]["oneOf"]
    for action in actions:
        if action.get("properties", {}).get("action", {}).get("const") == name:
            return action
    raise RuntimeError(f"Action schema is missing {name}")


def _numeric_range(action_name: str, field_name: str) -> str:
    action = _feedback_action_schema(action_name)
    parameters = action["properties"]["parameters"]
    field = parameters["properties"][field_name]
    return f"{field['minimum']}..{field['maximum']}"


def _feedback_prompt(
    summary: dict[str, object],
    *,
    retry_reason: str | None = None,
) -> str:
    dataset_name = summary["dataset_name"]
    topology_schema = _feedback_action_schema("set_topology")
    topologies = topology_schema["properties"]["parameters"]["properties"]["topology"]["enum"]
    nas_schema = _feedback_action_schema("enable_nas")
    search_time = (
        nas_schema["properties"]["parameters"]["properties"][
            "search_time_limit_seconds"
        ]
    )
    retry = (
        f" Previous action failed validation: {retry_reason}. Correct it."
        if retry_reason
        else ""
    )
    return (
        "Choose one allowed action. Return one JSON object with action and parameters; "
        "no prose or extra fields. start_training requires rounds "
        f"{_numeric_range('start_training', 'rounds')}, epochs "
        f"{_numeric_range('start_training', 'epochs')}, and dataset_name exactly "
        f"{dataset_name!r}. set_rounds accepts "
        f"{_numeric_range('set_rounds', 'rounds')}; set_epochs accepts "
        f"{_numeric_range('set_epochs', 'epochs')}. set_topology accepts "
        f"{'|'.join(topologies)}. enable_nas requires enabled=true, exact dataset_name, "
        f"and search_time_limit_seconds {search_time['minimum']}..{search_time['maximum']}. "
        "stop_training takes empty parameters. Example: "
        + json.dumps(
            {
                "action": "start_training",
                "parameters": {
                    "rounds": 3,
                    "epochs": 2,
                    "dataset_name": dataset_name,
                },
            },
            separators=(",", ":"),
        )
        + retry
        + " Metrics: "
        + json.dumps(summary, sort_keys=True, separators=(",", ":"))
    )


def _validate_feedback_suggestion(
    interpreted: object,
    summary: dict[str, object],
) -> tuple[dict[str, object] | None, str | None, str | None]:
    if not isinstance(interpreted, dict):
        return None, None, "invalid response: expected a JSON object"
    proposal = interpreted.get("action")
    reply = interpreted.get("reply")
    if not isinstance(proposal, dict):
        return None, None, "wrong or missing action: expected an action object"
    schema_errors = sorted(
        ACTION_VALIDATOR.iter_errors(proposal),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            error.message,
        ),
    )
    if schema_errors:
        error = schema_errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "<action>"
        return None, None, f"schema validation failed at {path}: {error.message}"
    if not isinstance(reply, str) or not reply.strip():
        return None, None, "missing or invalid reply: an evidence-based reason is required"
    try:
        action = validate_action(proposal)
        name = action["action"]
        parameters = action["parameters"]
        if name not in FEEDBACK_ACTIONS:
            raise AutonomyError(f"wrong action: {name} is not allowed in feedback loops", 422)
        if name in {"start_training", "enable_nas"}:
            if parameters.get("dataset_name") != summary["dataset_name"]:
                raise AutonomyError(
                    "parameters.dataset_name must equal the summarized dataset "
                    f"{summary['dataset_name']!r}",
                    422,
                )
        if name == "enable_nas":
            parameters["search_time_limit_seconds"] = 600
            action = validate_action(action)
    except (AutonomyError, KeyError, TypeError) as error:
        return None, None, str(error)
    return action, reply, None


def _rule_fallback_action(summary: dict[str, object]) -> tuple[dict[str, object], str]:
    accuracy_delta = summary.get("accuracy_delta")
    spread = summary["worker_accuracy_spread"]
    if isinstance(accuracy_delta, (int, float)) and accuracy_delta < 0.01:
        return (
            {
                "action": "enable_nas",
                "parameters": {
                    "enabled": True,
                    "dataset_name": summary["dataset_name"],
                    "search_time_limit_seconds": 600,
                },
            },
            "accuracy improvement was below 1%; compare dataset-specific NAS",
        )
    if isinstance(spread, (int, float)) and spread > 0.1:
        return (
            {"action": "set_topology", "parameters": {"topology": "mesh"}},
            "worker validation accuracy spread exceeded 0.1",
        )
    rounds_range = _numeric_range("set_rounds", "rounds")
    minimum, maximum = (int(value) for value in rounds_range.split(".."))
    current_rounds = summary["rounds"]
    if isinstance(current_rounds, int) and current_rounds < maximum:
        return (
            {
                "action": "set_rounds",
                "parameters": {"rounds": max(minimum, current_rounds + 1)},
            },
            "increase rounds by one within the shared action schema range",
        )
    epochs_range = _numeric_range("set_epochs", "epochs")
    epoch_minimum, epoch_maximum = (int(value) for value in epochs_range.split(".."))
    current_epochs = summary["epochs"]
    if isinstance(current_epochs, int) and current_epochs < epoch_maximum:
        return (
            {
                "action": "set_epochs",
                "parameters": {"epochs": max(epoch_minimum, current_epochs + 1)},
            },
            "increase epochs by one within the shared action schema range",
        )
    return (
        {"action": "stop_training", "parameters": {}},
        "rounds and epochs are already at their schema maxima",
    )


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def _forward(
    settings: Settings,
    path: str,
    *,
    payload: dict[str, object] | None = None,
    query: str = "",
    method: str | None = None,
) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    url = f"{settings.master_slm_url}{path}{query}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {settings.master_slm_token}",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    upstream = UrlRequest(
        url,
        data=body,
        headers=headers,
        method=method or ("POST" if body else "GET"),
    )
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(upstream, timeout=settings.slm_timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if 400 <= error.code < 500:
            upstream_detail = ""
            try:
                body = json.loads(error.read(4096).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = None
            if isinstance(body, dict):
                message = body.get("error", body.get("detail"))
                if isinstance(message, str):
                    upstream_detail = message.strip()[:300]
            raise HTTPException(
                status_code=error.code,
                detail=(
                    f"The master request failed (HTTP {error.code})"
                    + (f": {upstream_detail}" if upstream_detail else "")
                ),
            ) from error
        raise HTTPException(status_code=502, detail="The SLM service returned an error") from error
    except (URLError, TimeoutError, OSError) as error:
        raise HTTPException(
            status_code=502,
            detail="The SLM master is currently unreachable",
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=502,
            detail="The SLM master returned an invalid response",
        ) from error
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="The SLM master returned an invalid response")
    return result


def _record_completed_master_action(
    settings: Settings,
    action: dict[str, object],
    dispatch_result: dict[str, object],
) -> str | None:
    if action.get("action") in ACTION_HISTORY_EXCLUDED:
        return None
    if dispatch_result.get("status") != "executed":
        return None
    result = dispatch_result.get("result")
    if not isinstance(result, dict) or result.get("available") is not True:
        return None
    try:
        recorded = _forward(
            settings,
            "/action-history",
            payload={"action": action, "result": result},
        )
    except HTTPException as error:
        warning = (
            "The action completed, but the master did not save its history record "
            f"(HTTP {error.status_code}). Do not repeat the action; update the master service."
        )
        LOGGER.warning(
            "Completed action=%s but master history sync failed with HTTP %s",
            action.get("action"),
            error.status_code,
        )
        return warning
    if recorded.get("recorded") is not True:
        warning = (
            "The action completed, but the master did not save its history record. "
            "Do not repeat the action; check the SLM service."
        )
        LOGGER.warning(
            "Completed action=%s but master did not confirm history sync",
            action.get("action"),
        )
        return warning
    return None


def _mnist_results_reply(results: dict[str, object]) -> str:
    groups = results.get("groups")
    if not isinstance(groups, list):
        raise HTTPException(status_code=502, detail="The master returned invalid experiment results")
    groups = [
        group for group in groups
        if isinstance(group, dict) and group.get("dataset_name") == "mnist"
    ]
    if not groups:
        return "There are no stored MNIST experiment results yet."

    group = groups[-1]
    group_id = group.get("group_id", "unknown")
    status = group.get("status", "unknown")
    configured_rounds = group.get("rounds")
    runs = group.get("runs")
    latest_run = runs[-1] if isinstance(runs, list) and runs else None
    round_rows = latest_run.get("rounds") if isinstance(latest_run, dict) else None
    final_round = round_rows[-1] if isinstance(round_rows, list) and round_rows else None
    completed_rounds = len(round_rows) if isinstance(round_rows, list) else 0
    total_rounds = configured_rounds if isinstance(configured_rounds, int) else completed_rounds
    reply = (
        f"Stored MNIST experiment {group_id} is {status}; "
        f"{completed_rounds} of {total_rounds} rounds are recorded."
    )
    workers = final_round.get("workers") if isinstance(final_round, dict) else None
    if not isinstance(workers, dict) or not workers:
        return reply

    worker_summaries: list[str] = []
    sent_bytes = 0
    received_bytes = 0
    peer_wait_seconds = 0.0
    for worker_id, metrics in sorted(workers.items()):
        if not isinstance(metrics, dict):
            continue
        scores: list[str] = []
        for label, keys in (
            ("accuracy", ("validation_accuracy", "accuracy")),
            ("precision", ("validation_macro_precision", "macro_precision")),
            ("recall", ("validation_macro_recall", "macro_recall")),
            ("F1", ("validation_macro_f1", "macro_f1")),
        ):
            value = next((metrics[key] for key in keys if key in metrics), None)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                scores.append(f"{label} {float(value) * 100:.1f}%")
        if scores:
            worker_summaries.append(f"{worker_id}: {', '.join(scores)}")
        sent = metrics.get("bytes_sent", 0)
        received = metrics.get("bytes_received", 0)
        waiting = metrics.get("peer_wait_seconds", 0.0)
        if isinstance(sent, int) and sent >= 0:
            sent_bytes += sent
        if isinstance(received, int) and received >= 0:
            received_bytes += received
        if isinstance(waiting, (int, float)) and math.isfinite(float(waiting)) and waiting >= 0:
            peer_wait_seconds += float(waiting)
    if worker_summaries:
        reply += " Final-round validation scores: " + "; ".join(worker_summaries) + "."
    reply += (
        f" Recorded peer transfer: {sent_bytes + received_bytes} bytes; "
        f"peer waiting: {peer_wait_seconds:.3f} seconds."
    )
    return reply


def _next_action_reply(
    results: dict[str, object],
    preprocessing_report: dict[str, object] | None = None,
) -> str:
    groups = results.get("groups")
    if not isinstance(groups, list):
        raise HTTPException(status_code=502, detail="The master returned invalid experiment results")
    mnist_groups = [
        group for group in groups
        if isinstance(group, dict) and group.get("dataset_name") == "mnist"
    ]
    if not mnist_groups:
        workers = preprocessing_report.get("workers") if preprocessing_report else None
        uploaded_workers = (
            preprocessing_report.get("uploaded_workers") if preprocessing_report else None
        )
        recommendation = (
            "Next, create and distribute an initial model in Model Studio before starting "
            "distributed training."
        )
        if isinstance(workers, dict) and isinstance(uploaded_workers, dict) and workers:
            worker_ids = set(workers)
            installed_ids = worker_ids.intersection(uploaded_workers)
            if installed_ids == worker_ids:
                shipment_status = (
                    "The MNIST preprocessing report confirms shards are installed and "
                    f"verified on all {len(worker_ids)} workers. "
                )
            else:
                shipment_status = (
                    "The MNIST preprocessing report confirms shards are installed and "
                    f"verified on {len(installed_ids)} of {len(worker_ids)} workers. "
                )
                recommendation = (
                    "Complete shard installation in Datasets before creating a model, "
                    "then create and distribute an initial model in Model Studio before "
                    "starting distributed training."
                )
        else:
            shipment_status = (
                "If preprocessing and worker shard shipping are complete, verify the "
                "shards in Datasets. If they are not verified yet, finish that first. "
            )
        return (
            "There are no saved MNIST experiment results yet. "
            f"{shipment_status}{recommendation}"
        )

    group = mnist_groups[-1]
    group_id = group.get("group_id", "unknown")
    status = group.get("status", "unknown")
    if status == "running":
        runs = group.get("runs")
        runs = runs if isinstance(runs, list) else []
        completed_runs = sum(
            isinstance(run, dict) and run.get("status") == "completed"
            for run in runs
        )
        current_run = next(
            (
                run for run in reversed(runs)
                if isinstance(run, dict) and run.get("status") == "running"
            ),
            None,
        )
        progress_detail = ""
        if current_run is not None:
            rounds = current_run.get("rounds")
            completed_rounds = len(rounds) if isinstance(rounds, list) else 0
            progress = current_run.get("progress")
            progress_round = progress.get("round") if isinstance(progress, dict) else None
            if isinstance(progress_round, int):
                completed_rounds = max(completed_rounds, progress_round)
            configured_rounds = group.get("rounds")
            if isinstance(configured_rounds, int):
                progress_detail = (
                    f" The current {current_run.get('topology', 'training')} run is at "
                    f"round {completed_rounds}/{configured_rounds}."
                )
        total_runs = None
        candidates = group.get("candidates")
        seeds = group.get("seeds")
        if isinstance(candidates, list) and isinstance(seeds, list):
            total_runs = len(candidates) * len(seeds)
        run_detail = (
            f" {completed_runs}/{total_runs} candidate runs are complete."
            if total_runs is not None
            else ""
        )
        return (
            f"MNIST experiment {group_id} is still running.{run_detail}{progress_detail} "
            "Next, let it finish; then review the saved validation metrics and confusion "
            "matrix. Starting another run now would not improve this result."
        )

    if status in {"failed", "stopped"}:
        return (
            f"The latest MNIST experiment {group_id} is {status}. Next, inspect its recorded "
            "error and worker status in Experiments before deciding whether to retry."
        )

    summary = _mnist_results_reply(results)
    if status == "completed":
        selected = group.get("selected_configuration")
        if isinstance(selected, dict):
            topology = selected.get("topology")
            alpha = selected.get("alpha")
            if isinstance(topology, str) and isinstance(alpha, (int, float)):
                return (
                    f"{summary} The topology sweep selected {topology} with Dirichlet alpha "
                    f"{alpha:g}. Next, use that configuration for a standard training run "
                    "and review its final held-out evaluation."
                )
        return (
            f"{summary} Next, review the saved confusion matrix and validation precision, "
            "recall, and F1, then compare them with a baseline before changing the model."
        )
    return f"{summary} Check the Experiments page to confirm its state before taking another action."


def _experiment_results_from_action_result(result: object) -> dict[str, object] | None:
    pending = [result]
    visited: set[int] = set()
    while pending:
        value = pending.pop()
        if not isinstance(value, dict) or id(value) in visited:
            continue
        visited.add(id(value))
        if isinstance(value.get("groups"), list):
            return value
        for key in ("data", "result"):
            nested = value.get(key)
            if isinstance(nested, dict):
                pending.append(nested)
    return None


@router.post("/chat")
def chat(
    body: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    controller: AutonomyController = Depends(get_autonomy),
) -> dict[str, object]:
    conversation_id = _save_user_message(
        request,
        body.conversation_id,
        body.message,
    )
    next_action_requested = _is_next_action_request(body.message)
    interpreted = _forward(settings, "/interpret", payload={"message": body.message})
    action = interpreted.get("action")
    reply = interpreted.get("reply")
    if not isinstance(action, dict) or not isinstance(reply, str):
        raise HTTPException(status_code=502, detail="The SLM master returned an invalid interpretation")
    try:
        action = validate_action(action)
    except AutonomyError as error:
        raise HTTPException(status_code=502, detail="The SLM master returned an invalid action") from error
    if action.get("action") == "create_initial_model":
        dataset_name = action["parameters"]["model_name"]
        try:
            request.app.state.datasets.get_metadata(dataset_name)
        except DatasetError:
            response_data: dict[str, object] = {
                "reply": (
                    "Prepare the official MNIST dataset and its worker shards before "
                    "creating the model."
                ),
                "model_reply": reply,
                "action": {"action": "clarify", "parameters": {}},
                "result": {
                    "status": "clarify",
                    "reason": "Official MNIST is not prepared",
                },
                "metrics": interpreted.get("metrics", {}),
            }
            _save_assistant_message(
                request,
                conversation_id,
                str(response_data["reply"]),
                response_data,
            )
            return {**response_data, "conversation_id": conversation_id}
    if next_action_requested:
        results_data = _forward(settings, "/results")
        result = {
            "status": "executed",
            "result": {"available": True, "data": results_data},
        }
    else:
        try:
            result = controller.dispatch(
                action,
                reason=reply,
                evidence={"user_message": body.message},
            )
        except AutonomyError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error
        if action["action"] in {"preprocess_dataset", "create_initial_model"}:
            history_warning = _record_completed_master_action(settings, action, result)
            if history_warning:
                result["history_warning"] = history_warning
    results_summary: str | None = None
    if action["action"] == "get_results":
        results_summary = _mnist_results_reply(_forward(settings, "/results"))
    next_action_summary: str | None = None
    if action["action"] == "suggest_next_action":
        results_data = _experiment_results_from_action_result(result)
        if results_data is None:
            raise HTTPException(
                status_code=502,
                detail="The master did not return saved experiment results for the recommendation",
            )
        preprocessing_report = None
        if not any(
            isinstance(group, dict) and group.get("dataset_name") == "mnist"
            for group in results_data.get("groups", [])
        ):
            try:
                preprocessing_report = request.app.state.datasets.get_latest_preprocessing_report(
                    "mnist"
                )
            except DatasetError:
                pass
        next_action_summary = _next_action_reply(results_data, preprocessing_report)
    action_reply = reply
    context: dict[str, object] = {
        "action": action["action"],
        "action_parameters": action["parameters"],
        "action_reply": action_reply,
        "execution_status": result.get("status"),
    }
    if results_summary:
        context["stored_mnist_results"] = results_summary
    reply_style: dict[str, object] = {
        "available": False,
        "message": "Fine-tuned reply model unavailable; showing the action model's response.",
    }
    final_reply = action_reply
    if results_summary:
        final_reply += " " + results_summary
    if next_action_summary is not None:
        final_reply = f"{action_reply} {next_action_summary}"
    else:
        try:
            refined = _forward(
                settings,
                "/reply",
                payload={"message": body.message, "context": context},
            )
        except HTTPException as error:
            LOGGER.warning(
                "Fine-tuned reply model failed after action %s completed (HTTP %s)",
                action["action"],
                error.status_code,
            )
        else:
            refined_text = refined.get("reply")
            model_name = refined.get("model")
            latency_ms = refined.get("latency_ms")
            if (
                isinstance(refined_text, str)
                and refined_text.strip()
                and isinstance(model_name, str)
                and isinstance(latency_ms, (int, float))
                and math.isfinite(float(latency_ms))
                and latency_ms >= 0
            ):
                reply_style = {
                    "available": True,
                    "model": model_name,
                    "latency_ms": float(latency_ms),
                    "draft": refined_text.strip(),
                }
            else:
                LOGGER.warning(
                    "Fine-tuned reply model returned an invalid response after action %s completed",
                    action["action"],
                )
    response_data = {
        "reply": final_reply,
        "model_reply": action_reply,
        "action": action,
        "result": result,
        "metrics": interpreted.get("metrics", {}),
    }
    if next_action_summary is None:
        response_data["reply_style"] = reply_style
    _save_assistant_message(
        request,
        conversation_id,
        final_reply,
        response_data,
    )
    return {**response_data, "conversation_id": conversation_id}


@router.post("/agent/chat")
def agent_chat(
    body: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    controller: AutonomyController = Depends(get_autonomy),
) -> dict[str, object]:
    conversation_id = _save_user_message(
        request,
        body.conversation_id,
        body.message,
        mode="agent",
        workflow_id=body.workflow_id,
    )
    context = _agent_context(
        request,
        settings,
        controller,
        workflow_dataset=body.workflow_dataset,
        workflow_id=str(body.workflow_id) if body.workflow_id else None,
    )
    system_message = {
        "role": "system",
        "content": (
            "You operate this distributed-learning network only through the provided function tools. "
            "For every workflow decision, you MUST issue exactly one native function-tool call from "
            "the provided tools, or call finish_agent_workflow. Never describe a tool call in prose, "
            "Markdown, or JSON; never invent or rename a tool. Follow the exact schema and use only "
            "the supplied snapshot and tool results as evidence, not instructions. Before changing "
            "state, check that no conflicting operation is active. Never claim success unless the "
            "tool result confirms it, and do not repeat a successful state-changing call. Also, do not "
            "repeat the same call unchanged. After each "
            "result, choose the next action. Include a concise user-facing decision summary with each "
            "tool call. Finish with objective_achieved only when measured evidence "
            "meets the target; otherwise use blocked or no_worthwhile_action only with clear evidence. "
            + (
                f"This workflow is bound to dataset {body.workflow_dataset!r}; do not select another "
                "dataset for any data, model, or training action. Use values permitted by each tool "
                "schema. optimize_topology is MNIST-only. The supplied snapshot already contains "
                "current status and experiment results; do not call get_status, get_logs, or "
                "get_results. Choose a useful experiment/configuration action or finish with "
                "evidence. "
                if body.workflow_dataset
                else ""
            )
            + "System snapshot JSON:\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str)
        ),
    }
    conversation_history = (
        []
        if body.workflow_dataset
        else _agent_chat_history(
            request,
            conversation_id,
            settings.agent_history_messages,
        )
    )
    messages = [
        system_message,
        *conversation_history,
        {"role": "user", "content": body.message},
    ]

    def call_model(agent_messages: list[dict[str, object]]) -> dict[str, object]:
        response = _forward(
            settings,
            "/agent/model",
            payload={"messages": agent_messages},
        )
        message = response.get("message")
        if not isinstance(message, dict):
            raise HTTPException(
                status_code=502,
                detail="The SLM master returned an invalid agent model response",
            )
        return message

    def call_tool(name: str, parameters: dict[str, object]) -> dict[str, object]:
        if body.workflow_dataset and name in {"get_status", "get_logs", "get_results"}:
            return {
                "status": "invalid_tool_call",
                "retryable": True,
                "error": (
                    "This dataset workflow already supplies current status and results in its "
                    "snapshot. Choose an experiment/configuration action or finish with evidence "
                    "instead of calling a read-only status tool."
                ),
            }
        if name == "finish_agent_workflow":
            outcome = parameters.get("outcome")
            summary = parameters.get("summary")
            evidence = parameters.get("evidence")
            if (
                outcome not in {"objective_achieved", "no_worthwhile_action", "blocked"}
                or not isinstance(summary, str)
                or not summary.strip()
                or not isinstance(evidence, str)
                or not evidence.strip()
            ):
                return {
                    "status": "invalid_tool_call",
                    "retryable": True,
                    "error": "Workflow termination requires a supported outcome, summary, and evidence.",
                }
            return {
                "status": "workflow_complete",
                "outcome": outcome,
                "summary": summary.strip(),
                "evidence": evidence.strip(),
            }
        try:
            action = agent_tool_action(name, parameters)
            if body.workflow_dataset:
                action_name = action["action"]
                action_parameters = action["parameters"]
                experiment_context = context.get("experiments")
                available_groups = (
                    experiment_context.get("groups", [])
                    if isinstance(experiment_context, dict)
                    else []
                )
                previous_group = next(
                    (
                        group
                        for group in available_groups
                        if isinstance(group, dict)
                        and group.get("group_id") == body.workflow_previous_group_id
                        and (
                            group.get("dataset_name") == body.workflow_dataset
                            or (
                                group.get("kind") == "topology_optimization"
                                and body.workflow_dataset == "mnist"
                            )
                        )
                        and group.get("status")
                        in {"completed", "failed", "stopped", "cancelled"}
                    ),
                    None,
                )
                if body.workflow_previous_group_id and previous_group is None:
                    return {
                        "status": "invalid_tool_call",
                        "error": (
                            "The previous workflow experiment is not a completed result for "
                            f"dataset {body.workflow_dataset}."
                        ),
                    }
                if action_name == "optimize_topology" and body.workflow_dataset != "mnist":
                    return {
                        "status": "invalid_tool_call",
                        "error": (
                            "optimize_topology currently runs MNIST only; use dataset-specific "
                            "NAS or set_topology for this workflow."
                        ),
                    }
                if action_name in {"preprocess_dataset", "start_training", "enable_nas"}:
                    requested_dataset = action_parameters.get("dataset_name")
                    if requested_dataset != body.workflow_dataset:
                        return {
                            "status": "invalid_tool_call",
                            "error": (
                                f"This workflow is bound to {body.workflow_dataset}; actions "
                                f"cannot use {requested_dataset}."
                            ),
                        }
                if action_name == "enable_nas":
                    if action_parameters.get("enabled") is not True:
                        return {
                            "status": "invalid_tool_call",
                            "error": "Dataset workflow NAS must be enabled to run a search.",
                        }
                    if action_parameters.get("smoke_test") is True:
                        return {
                            "status": "invalid_tool_call",
                            "error": "Dataset workflow NAS must run its full multi-fidelity search, not a smoke test.",
                        }
                if action_name == "start_training":
                    configuration = (
                        experiment_context.get("configuration")
                        if isinstance(experiment_context, dict)
                        else None
                    )
                    if (
                        isinstance(configuration, dict)
                        and any(
                            isinstance(group, dict)
                            and group.get("dataset_name") == body.workflow_dataset
                            and group.get("status") == "completed"
                            and _repeats_completed_training(
                                action_parameters,
                                group,
                                configuration,
                            )
                            for group in available_groups
                        )
                    ):
                        return {
                            "status": "invalid_tool_call",
                            "error": (
                                "This exact distributed-training configuration already "
                                "completed in the workflow. Choose a changed configuration "
                                "or another allowed action."
                            ),
                        }
                if (
                    action_name == "start_training"
                    and previous_group
                    and previous_group.get("status") == "completed"
                ):
                    if previous_group.get("kind") == "dataset_architecture_search":
                        selected_architecture = previous_group.get("selected_architecture")
                        if not isinstance(selected_architecture, dict):
                            return {
                                "status": "invalid_tool_call",
                                "error": "The completed NAS group has no selected architecture to retrain.",
                            }
                        if action_parameters.get("architecture") != selected_architecture:
                            return {
                                "status": "invalid_tool_call",
                                "error": (
                                    "The start_training architecture must exactly match the "
                                    "selected architecture in the referenced completed NAS result."
                                ),
                            }
                    if previous_group.get("kind") == "topology_optimization":
                        selected_configuration = previous_group.get("selected_configuration")
                        if isinstance(selected_configuration, dict):
                            selected_split = selected_configuration.get("dataset_split")
                            if (
                                isinstance(selected_split, str)
                                and action_parameters.get("dataset_split") != selected_split
                            ):
                                return {
                                    "status": "invalid_tool_call",
                                    "error": (
                                        "The training dataset_split must exactly match the "
                                        "selected split in the referenced topology result."
                                    ),
                                }
                if action_name == "create_initial_model":
                    requested_dataset = action_parameters.get("model_name")
                    if requested_dataset is not None and requested_dataset != body.workflow_dataset:
                        return {
                            "status": "invalid_tool_call",
                            "error": (
                                f"This workflow is bound to {body.workflow_dataset}; initial "
                                f"models cannot use {requested_dataset}."
                            ),
                        }
            action = validate_action(action)
        except AutonomyError as error:
            return {
                "status": "invalid_tool_call",
                "retryable": error.status_code == 422,
                "error": str(error),
            }
        try:
            dispatched = controller.dispatch(
                action,
                reason=body.message,
                evidence={
                    "source": "langgraph_agent",
                    "conversation_id": conversation_id,
                    **(
                        {"workflow_id": str(body.workflow_id)}
                        if body.workflow_id
                        else {}
                    ),
                    "snapshot_captured_at": context["captured_at"],
                },
                mode="AGENT",
            )
        except AutonomyError as error:
            return {"status": "failed", "error": str(error)}
        action_result = dispatched.get("result")
        if isinstance(action_result, dict):
            action_data = action_result.get("data")
            if (
                body.workflow_dataset
                and action["action"] == "create_initial_model"
                and isinstance(action_data, dict)
                and action_data.get("reused_existing_model") is True
            ):
                return {
                    "status": "invalid_tool_call",
                    "retryable": True,
                    "error": (
                        "The initial model already exists and was reused. Choose the next "
                        "experiment/configuration action instead of recreating it."
                    ),
                }
            if isinstance(action_data, dict) and action_data.get("accepted") is False:
                message = action_data.get("message")
                return {
                    "status": "invalid_tool_call",
                    "retryable": True,
                    "error": (
                        message
                        if isinstance(message, str) and message
                        else f"{action['action']} was not accepted"
                    ),
                }
        history_warning = None
        if isinstance(action_result, dict) and action_result.get("available") is True:
            history_warning = _record_completed_master_action(settings, action, dispatched)
        tool_result = {
            "status": dispatched.get("status", "executed"),
            "action": action["action"],
            "result": _compact_agent_tool_value(dispatched.get("result")),
        }
        if history_warning:
            tool_result["history_warning"] = history_warning
        return tool_result

    try:
        agent_result = run_agent(
            messages=messages,
            model_call=call_model,
            tool_call=call_tool,
            max_tool_calls=(
                1
                if body.workflow_id and body.workflow_dataset
                else settings.agent_max_tool_calls
            ),
        )
    except HTTPException:
        raise
    except RuntimeError as error:
        LOGGER.exception("LangGraph agent failed for conversation %s", conversation_id)
        raise HTTPException(status_code=502, detail="The SLM agent could not complete the request") from error

    if (
        body.workflow_dataset
        and not agent_result["terminated"]
        and agent_result["tool_calls"] == 0
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                "The SLM returned no native tool call for this Agent workflow. "
                "The workflow was halted instead of repeating a no-progress decision."
            ),
        )

    response_data: dict[str, object] = {
        "reply": agent_result["reply"],
        "mode": "agent",
        "tool_calls": agent_result["tool_calls"],
        "trace": agent_result["trace"],
        "terminated": agent_result["terminated"],
        **({"workflow_id": str(body.workflow_id)} if body.workflow_id else {}),
    }
    _save_assistant_message(
        request,
        conversation_id,
        str(agent_result["reply"]),
        response_data,
    )
    return {**response_data, "conversation_id": conversation_id}


@router.get("/reply-evaluation")
def reply_evaluation(
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    result = _forward(settings, "/reply-status")
    if result.get("available") is not True or not isinstance(
        result.get("evaluation"), dict
    ):
        raise HTTPException(
            status_code=502,
            detail="The fine-tuned reply model returned invalid evaluation data",
        )
    return result


@router.post("/suggest-next-action")
def suggest_next_action(
    body: FeedbackRequest,
    settings: Settings = Depends(get_settings),
    controller: AutonomyController = Depends(get_autonomy),
) -> dict[str, object]:
    result_payload = _forward(settings, "/results")
    groups = result_payload.get("groups")
    if not isinstance(groups, list):
        raise HTTPException(status_code=502, detail="The master returned invalid experiment results")
    group = next(
        (
            item
            for item in groups
            if isinstance(item, dict) and item.get("group_id") == body.group_id
        ),
        None,
    )
    if group is None:
        raise HTTPException(status_code=404, detail="Experiment group was not found")
    try:
        summary = build_training_feedback_summary(
            group,
            iteration=body.iteration,
            previous_accuracy=body.previous_accuracy,
        )
    except FeedbackSummaryError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    attempts: list[dict[str, object]] = []
    action: dict[str, object] | None = None
    reply: str | None = None
    validation_reason: str | None = None
    for attempt_number in range(1, FEEDBACK_RETRIES + 1):
        prompt = _feedback_prompt(
            summary,
            retry_reason=validation_reason if attempt_number > 1 else None,
        )
        interpreted = _forward(settings, "/interpret", payload={"message": prompt})
        action, reply, validation_reason = _validate_feedback_suggestion(
            interpreted,
            summary,
        )
        attempts.append(
            {
                "attempt": attempt_number,
                "raw_output": interpreted,
                "validator_result": validation_reason or "valid",
            }
        )
        if validation_reason is None:
            break
        LOGGER.warning(
            "Discarded invalid feedback suggestion attempt=%d reason=%s",
            attempt_number,
            validation_reason,
        )

    source = "slm"
    if validation_reason is not None:
        if not settings.feedback_rule_fallback_enabled:
            return {
                "status": "needs_clarification",
                "source": source,
                "reason": validation_reason,
                "attempts": attempts,
                "summary": summary,
            }
        action, fallback_reason = _rule_fallback_action(summary)
        try:
            validate_action(action)
        except AutonomyError as error:
            raise RuntimeError("Rule fallback generated an invalid schema action") from error
        source = "rule_fallback"
        reply = f"Deterministic fallback: {fallback_reason}."
        validation_reason = None
        LOGGER.warning(
            "Using rule_fallback after %d rejected feedback suggestions",
            len(attempts),
        )

    try:
        if action is None or reply is None:
            raise RuntimeError("Feedback validation completed without an action or reason")
        result = controller.dispatch(
            action,
            reason=reply,
            evidence={
                "source": source,
                "iteration": body.iteration,
                "summary": summary,
                **(
                    {"fallback_reason": reply}
                    if source == "rule_fallback"
                    else {}
                ),
            },
        )
    except AutonomyError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    return {
        "status": result.get("status", "executed"),
        "action": action,
        "reason": reply,
        "summary": summary,
        "result": result,
        "source": source,
        "attempts": attempts,
    }


@router.post("/command")
def command(
    body: dict[str, object],
    settings: Settings = Depends(get_settings),
    controller: AutonomyController = Depends(get_autonomy),
) -> dict[str, object]:
    if set(body) - {"action", "reason", "evidence"}:
        raise HTTPException(status_code=422, detail="Unexpected command fields")
    action = body.get("action")
    if not isinstance(action, dict):
        raise HTTPException(status_code=422, detail="An action object is required")
    reason = body.get("reason", "Authenticated coordinator command")
    evidence = body.get("evidence", {})
    if not isinstance(reason, str) or not isinstance(evidence, dict):
        raise HTTPException(status_code=422, detail="Invalid reason or evidence")
    try:
        result = controller.dispatch(action, reason=reason, evidence=evidence)
        history_warning = _record_completed_master_action(settings, action, result)
        if history_warning:
            result["history_warning"] = history_warning
        return result
    except AutonomyError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.get("/status")
def status(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    return _forward(settings, "/status")


@router.get("/logs")
def logs(
    limit: int = Query(default=50, ge=1, le=100),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    return _forward(settings, "/logs", query=f"?limit={limit}")


@router.get("/results")
def results(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    return _forward(settings, "/results")


@router.delete("/results/{group_id}")
def delete_result_group(
    group_id: str,
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    return _forward(
        settings,
        f"/results/{quote(group_id, safe='')}",
        method="DELETE",
    )


@router.post("/baseline")
def baseline(
    body: BaselineRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    return _forward(settings, "/baseline", payload=body.model_dump(exclude_none=True))
