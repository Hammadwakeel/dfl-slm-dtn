"""Supervisor agent: decision contract and end-to-end workflow with fake cluster/SLM."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import AutonomyState, Base, Device
from app.services.supervisor import Supervisor, SupervisorError
from app.services.supervisor_policy import (
    DecisionError,
    allowed_steps,
    default_goal,
    fallback_decision,
    new_state,
    parse_decision,
    rule_decision,
    summarize_group,
)

NAS_ARCHITECTURE = {"id": "b2_c16_k5", "model_type": "custom", "dataset_name": "mnist", "parameter_count": 7802}


# --------------------------------------------------------------------------- policy


def test_allowed_steps_offer_only_progress_steps_and_finish() -> None:
    goal = default_goal()
    state = new_state()
    state["completed_steps"] = ["prepare_data", "initial_model", "baseline_training"]

    assert allowed_steps(goal, state) == ["run_nas", "optimize_topology", "learn_topology", "finish_workflow"]

    state["completed_steps"].append("run_nas")
    state["nas_architecture"] = NAS_ARCHITECTURE
    assert allowed_steps(goal, state) == ["optimize_topology", "learn_topology", "run_hpo", "train_final", "finish_workflow"]

    cifar = default_goal(dataset="cifar10")
    assert "optimize_topology" not in allowed_steps(cifar, state)


def test_trained_configuration_is_not_offered_again() -> None:
    goal = default_goal()
    state = new_state()
    state["completed_steps"] = ["run_nas", "run_hpo", "learn_topology", "optimize_topology", "train_final"]
    state["nas_architecture"] = NAS_ARCHITECTURE
    state["trained_configs"] = ["b2_c16_k5|directed_ring"]

    assert allowed_steps(goal, state) == ["finish_workflow"]


def test_parse_decision_accepts_native_and_text_tool_calls() -> None:
    allowed = ["run_nas", "train_final", "finish_workflow"]
    native = parse_decision(
        {
            "content": "NAS can find a stronger model.",
            "tool_calls": [{"function": {"name": "train_final", "arguments": {"rounds": "50", "local_epochs": "3"}}}],
        },
        allowed,
    )
    # Out-of-range integers are clamped to the tool bounds instead of rejected.
    assert (native.step, native.arguments, native.source) == ("train_final", {"rounds": 20, "local_epochs": 3}, "slm")
    assert native.rationale == "NAS can find a stronger model."

    text = parse_decision(
        {"content": 'I choose NAS. {"name": "run_nas", "arguments": {}}'},
        allowed,
    )
    assert (text.step, text.source, text.rationale) == ("run_nas", "slm_text", "I choose NAS.")


@pytest.mark.parametrize(
    "message",
    [
        {"content": "Let me check the results first."},
        {"tool_calls": [{"function": {"name": "get_results", "arguments": {}}}]},
        {"tool_calls": [{"function": {"name": "train_final", "arguments": {"topology": "torus"}}}]},
        {"tool_calls": [{"function": {"name": "finish_workflow", "arguments": {"outcome": "done"}}}]},
    ],
)
def test_parse_decision_rejects_prose_unknown_tools_and_bad_arguments(message: dict[str, Any]) -> None:
    with pytest.raises(DecisionError):
        parse_decision(message, ["run_nas", "train_final", "finish_workflow"])


def test_rule_decision_finishes_when_target_met_and_fallback_order() -> None:
    goal = default_goal(target_accuracy=0.9)
    state = new_state()
    state["completed_steps"] = ["baseline_training"]
    state["best"] = {"validation_accuracy": 0.93}

    decision = rule_decision(goal, state, minutes_left=100)
    assert decision is not None and decision.arguments["outcome"] == "objective_achieved"

    state["best"] = {"validation_accuracy": 0.5}
    assert rule_decision(goal, state, minutes_left=100) is None
    assert rule_decision(goal, state, minutes_left=0).arguments["outcome"] == "blocked"
    fallback = fallback_decision(goal, state)
    assert (fallback.step, fallback.arguments, fallback.source) == ("run_nas", {}, "fallback")


def test_summarize_group_reads_final_round_worker_metrics() -> None:
    group = {
        "group_id": "g1",
        "kind": "decentralized",
        "status": "completed",
        "topology": "directed_ring",
        "architecture": {"id": "b1_c8_k3", "parameter_count": 251},
        "runs": [
            {
                "rounds": [
                    {"workers": {"worker1": {"validation_accuracy": 0.5}}},
                    {
                        "workers": {
                            "worker1": {"validation_accuracy": 0.9, "validation_macro_f1": 0.8},
                            "worker2": {"validation_accuracy": 0.8, "validation_macro_f1": 0.7},
                        }
                    },
                ]
            }
        ],
    }
    summary = summarize_group(group)
    assert summary["validation_accuracy"] == pytest.approx(0.85)
    assert summary["macro_f1"] == pytest.approx(0.75)
    assert summary["architecture"] == "b1_c8_k3"
    assert summary["rounds_completed"] == 2


# --------------------------------------------------------------------------- engine


class FakeCluster:
    """Fake coordinator: dispatches actions, runs experiments that finish on the next tick."""

    def __init__(self) -> None:
        self.actions: list[dict[str, Any]] = []
        self.groups: list[dict[str, Any]] = []
        self.accuracy = {"baseline_training": 0.90, "train_final": 0.96}
        self.fail_next: dict[str, str] = {}
        self.cleanups = 0
        self.counter = 0
        self.model_report: dict[str, Any] = {}
        self.nas_architecture: dict[str, Any] = NAS_ARCHITECTURE
        self.pruned = False

    def dispatch(self, action: dict[str, Any], reason: str, run_id: str) -> dict[str, Any]:
        self.actions.append(action)
        name = action["action"]
        if name in {"preprocess_dataset", "create_initial_model"}:
            return {"status": "executed", "result": {"available": True, "data": {"dataset_name": "mnist", **self.model_report}}}
        if name == "prune_model":
            self.pruned = True
            report = {"ratio": action["parameters"]["ratio"], "parameters_before": 1000, "parameters_after": 400, "compression": 2.5, "architecture_id": "nas-x-pruned50", "source_worker": "worker1"}
            return {"status": "executed", "result": {"available": True, "data": report}}
        if name in {"set_topology", "stop_training"}:
            for group in self.groups:
                if name == "stop_training" and group["status"] == "running":
                    group["status"] = "stopped"
            return {"status": "executed", "result": {"action": action, "result": {"available": True, "data": {}}}}
        self.counter += 1
        group_id = f"group-{self.counter}"
        kind = {
            "enable_nas": "dataset_architecture_search",
            "run_hpo": "hyperparameter_search",
            "optimize_topology": "topology_optimization",
            "start_training": "decentralized",
        }[name]
        self.groups.append(
            {
                "group_id": group_id,
                "kind": kind,
                "status": "running",
                "rounds": action["parameters"].get("rounds", 1),
                "architecture": action["parameters"].get("architecture", {"id": "b1_c8_k3"}),
                "runs": [],
                "_action": action,
                "_fine_tune": name == "start_training" and self.pruned,
            }
        )
        return {
            "status": "executed",
            "result": {"action": action, "result": {"available": True, "data": {"accepted": True, "group_id": group_id}}},
        }

    def results(self) -> dict[str, Any]:
        # Every running experiment finishes when the supervisor next looks at it.
        for group in self.groups:
            if group["status"] != "running" or group.get("_seen") is None:
                group["_seen"] = True
                continue
            action = group["_action"]
            if group["kind"] == "decentralized":
                step = "fine_tune" if group["_fine_tune"] else "train_final" if "architecture" in action["parameters"] else "baseline_training"
                if step in self.fail_next:
                    group.update(status="failed", error=self.fail_next.pop(step))
                    continue
                accuracy = self.accuracy[step]
                group["runs"] = [{"rounds": [{"workers": {"worker1": {"validation_accuracy": accuracy, "validation_macro_f1": accuracy - 0.01}}}]}]
                group["topology"] = "directed_ring"
            elif group["kind"] == "dataset_architecture_search":
                group["selected_architecture"] = self.nas_architecture
            elif group["kind"] == "hyperparameter_search":
                group["selected_architecture"] = {**action["parameters"]["architecture"], "id": "hpo-adam-lr3e-3-bs64"}
                group["selected_hyperparameters"] = {"optimizer": "adam", "learning_rate": 0.003, "batch_size": 64}
            elif group["kind"] == "topology_optimization":
                group["selected_configuration"] = {"topology": "bidirectional_ring"}
            group["status"] = "completed"
        return {"active_group_id": None, "groups": self.groups}

    def cleanup(self) -> dict[str, Any]:
        self.cleanups += 1
        return {"complete": True}


class FakeSLM:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        self.calls.append((messages, tools))
        if not self.replies:
            raise RuntimeError("SLM master is unreachable")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def tool_call(name: str, arguments: dict[str, Any] | None = None, content: str = "") -> dict[str, Any]:
    return {"role": "assistant", "content": content, "tool_calls": [{"function": {"name": name, "arguments": arguments or {}}}]}


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 17, 9, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def sessions():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory.begin() as session:
        session.add(AutonomyState(id=1, stop_requested=False, updated_at=datetime.now(timezone.utc)))
        for index in range(1, 4):
            session.add(
                Device(
                    node_id=f"worker{index}",
                    role="worker",
                    hostname=f"dfl-worker{index}",
                    ip=f"192.168.11.{index}",
                    port=5001,
                    online=True,
                    training_ready=True,
                    latest_status={"memory": {"available_bytes": 5_000 * 1048576}},
                )
            )
    return factory


def make_supervisor(sessions, cluster: FakeCluster, slm: FakeSLM, clock: Clock | None = None) -> Supervisor:
    return Supervisor(
        sessions,
        dispatch=cluster.dispatch,
        master_results=cluster.results,
        model_call=slm,
        memory_cleanup=cluster.cleanup,
        clock=clock or Clock(),
        preflight_wait_seconds=600,
    )


def run_until_done(supervisor: Supervisor, run_id: str, ticks: int = 40) -> dict[str, Any]:
    for _ in range(ticks):
        supervisor.tick()
        run = supervisor.get_run(run_id)
        if run["status"] not in {"running", "stopping", "queued"}:
            return run
    raise AssertionError(f"Run did not finish: {supervisor.get_run(run_id)['stage']}")


def test_supervisor_runs_baseline_nas_and_retrains_with_redistributed_architecture(sessions) -> None:
    cluster = FakeCluster()
    slm = FakeSLM([
        tool_call("run_nas", {}, "NAS should find a stronger model."),
        tool_call("train_final", {"rounds": 4, "local_epochs": 3, "aggregation": "metropolis"}, "Retrain with the searched model."),
    ])
    supervisor = make_supervisor(sessions, cluster, slm)
    run = supervisor.create_run(default_goal(target_accuracy=0.95))

    finished = run_until_done(supervisor, run["id"])

    assert finished["status"] == "completed"
    assert finished["result"]["outcome"] == "objective_achieved"
    assert finished["result"]["decided_by"] == "rule"
    assert finished["state"]["best"]["validation_accuracy"] == pytest.approx(0.96)
    names = [action["action"] for action in cluster.actions]
    assert names[:2] == ["preprocess_dataset", "create_initial_model"]
    # The searched architecture is redistributed before the retrain starts.
    retrain_index = len(names) - 1 - names[::-1].index("start_training")
    assert names[retrain_index - 1] == "create_initial_model"
    assert cluster.actions[retrain_index - 1]["parameters"]["architecture"] == NAS_ARCHITECTURE
    assert cluster.actions[retrain_index]["parameters"]["architecture"] == NAS_ARCHITECTURE
    assert cluster.actions[retrain_index]["parameters"]["rounds"] == 4
    # The SLM chose the aggregation rule; synchronization comes from the goal.
    assert cluster.actions[retrain_index]["parameters"]["aggregation"] == "metropolis"
    assert cluster.actions[retrain_index]["parameters"]["sync_mode"] == "semi_sync"
    assert cluster.actions[retrain_index]["parameters"]["sync_deadline_seconds"] == 300
    baseline = next(action for action in cluster.actions if action["action"] == "start_training")
    assert baseline["parameters"]["aggregation"] == "fedavg"
    # Every SLM decision received only the compact allowed tool set.
    offered = [{tool["function"]["name"] for tool in tools} for _messages, tools in slm.calls]
    assert offered[0] == {"run_nas", "optimize_topology", "learn_topology", "finish_workflow"}
    assert all("get_results" not in names for names in offered)
    assert finished["state"]["counters"]["slm_decisions"] == 2
    event_types = [event["event_type"] for event in finished["events"]]
    assert event_types.count("decision") == 3
    assert event_types[-1] == "finished"


def test_invalid_slm_reply_is_repaired_once_then_falls_back(sessions) -> None:
    cluster = FakeCluster()
    cluster.accuracy["baseline_training"] = 0.5
    slm = FakeSLM([
        {"role": "assistant", "content": "I will check the logs first."},
        {"role": "assistant", "content": "Still thinking."},
    ])
    supervisor = make_supervisor(sessions, cluster, slm)
    run = supervisor.create_run(default_goal(max_experiments=2))

    finished = run_until_done(supervisor, run["id"])

    counters = finished["state"]["counters"]
    assert counters["fallback_decisions"] >= 1
    assert "run_nas" in finished["state"]["completed_steps"]
    assert any(event["event_type"] == "invalid_decision" for event in finished["events"])
    # The repair prompt names the allowed tools explicitly.
    assert "Call exactly one of these tools" in slm.calls[1][0][-1]["content"]


def test_transient_experiment_failure_cleans_memory_and_retries_once(sessions) -> None:
    cluster = FakeCluster()
    cluster.fail_next["baseline_training"] = "TrainingError: Timed out waiting for peer weights from ['worker2']"
    cluster.accuracy["baseline_training"] = 0.97
    supervisor = make_supervisor(sessions, cluster, FakeSLM([]))
    run = supervisor.create_run(default_goal())

    finished = run_until_done(supervisor, run["id"])

    assert finished["status"] == "completed"
    assert cluster.cleanups == 1
    assert [a["action"] for a in cluster.actions].count("start_training") == 2
    assert any(event["event_type"] == "remediation" for event in finished["events"])


def test_low_memory_triggers_cleanup_before_admission(sessions) -> None:
    with sessions.begin() as session:
        device = session.get(Device, "worker2")
        device.latest_status = {"memory": {"available_bytes": 600 * 1048576}}
    cluster = FakeCluster()
    supervisor = make_supervisor(sessions, cluster, FakeSLM([]))
    run = supervisor.create_run(default_goal())

    supervisor.tick()

    assert cluster.cleanups == 1
    assert supervisor.get_run(run["id"])["stage"] == "preflight"
    assert cluster.actions == []


def test_stop_request_stops_active_experiment(sessions) -> None:
    cluster = FakeCluster()
    supervisor = make_supervisor(sessions, cluster, FakeSLM([]))
    run = supervisor.create_run(default_goal())
    for _ in range(4):
        supervisor.tick()
    assert supervisor.get_run(run["id"])["stage"] == "monitoring"

    supervisor.request_stop(run["id"], "Operator stop")
    supervisor.tick()

    stopped = supervisor.get_run(run["id"])
    assert stopped["status"] == "stopped"
    assert cluster.actions[-1]["action"] == "stop_training"
    with pytest.raises(SupervisorError):
        supervisor.request_stop(run["id"])


def test_only_one_active_run_and_state_survives_a_new_supervisor(sessions) -> None:
    cluster = FakeCluster()
    supervisor = make_supervisor(sessions, cluster, FakeSLM([]))
    run = supervisor.create_run(default_goal())
    with pytest.raises(SupervisorError):
        supervisor.create_run(default_goal())
    for _ in range(3):
        supervisor.tick()

    restarted = make_supervisor(sessions, cluster, FakeSLM([]))
    finished = run_until_done(restarted, run["id"])

    assert finished["status"] in {"completed", "failed"}
    assert [a["action"] for a in cluster.actions].count("preprocess_dataset") == 1


def test_global_stop_pauses_without_dispatching(sessions) -> None:
    with sessions.begin() as session:
        session.get(AutonomyState, 1).stop_requested = True
    cluster = FakeCluster()
    supervisor = make_supervisor(sessions, cluster, FakeSLM([]))
    run = supervisor.create_run(default_goal())

    supervisor.tick()
    supervisor.tick()

    assert cluster.actions == []
    events = supervisor.get_run(run["id"])["events"]
    assert sum(1 for event in events if event["event_type"] == "waiting") == 1


def test_unready_cluster_blocks_after_wait_budget(sessions) -> None:
    with sessions.begin() as session:
        session.get(Device, "worker3").online = False
    clock = Clock()
    cluster = FakeCluster()
    supervisor = make_supervisor(sessions, cluster, FakeSLM([]), clock)
    run = supervisor.create_run(default_goal())

    supervisor.tick()
    clock.now += timedelta(minutes=11)
    supervisor.tick()

    blocked = supervisor.get_run(run["id"])
    assert blocked["status"] == "failed"
    assert "worker3" in blocked["result"]["summary"]


def test_ask_answers_from_run_state_without_tools(sessions) -> None:
    cluster = FakeCluster()
    slm = FakeSLM([{"role": "assistant", "content": "The baseline is running."}])
    supervisor = make_supervisor(sessions, cluster, slm)
    run = supervisor.create_run(default_goal())

    answer = supervisor.ask(run["id"], "Kya ho raha hai?")

    assert answer["answer"] == "The baseline is running."
    messages, tools = slm.calls[0]
    assert tools is None
    assert messages[-1]["content"] == "Kya ho raha hai?"
    assert '"dataset":"mnist"' in messages[0]["content"]


def test_supervisor_api_requires_auth_and_validates_goal(client, auth_headers) -> None:
    assert client.get("/api/supervisor/runs").status_code == 401
    invalid = client.post("/api/supervisor/runs", json={"dataset": "imagenet"}, headers=auth_headers)
    assert invalid.status_code == 422
    listed = client.get("/api/supervisor/runs", headers=auth_headers)
    assert listed.status_code == 200
    assert listed.json() == []


def test_reason_argument_becomes_the_rationale_and_is_not_a_step_argument() -> None:
    decision = parse_decision(
        {"tool_calls": [{"function": {"name": "run_nas", "arguments": {"reason": "Baseline is 4 points below target."}}}]},
        ["run_nas", "finish_workflow"],
    )
    assert decision.arguments == {}
    stale = parse_decision(
        {"tool_calls": [{"function": {"name": "run_nas", "arguments": {"budget_seconds": 100}}}]},
        ["run_nas"],
    )
    assert stale.arguments == {}
    wordy = parse_decision(
        {"tool_calls": [{"function": {"name": "run_nas", "arguments": {"reason": "x" * 900}}}]},
        ["run_nas"],
    )
    assert wordy.step == "run_nas" and len(wordy.rationale) == 300
    assert decision.rationale == "Baseline is 4 points below target."


CNN_LAYERS = [
    {"type": "conv2d", "filters": 16, "kernel_size": 3, "padding": "same", "activation": "relu"},
    {"type": "batch_normalization"},
    {"type": "global_average_pooling2d"},
    {"type": "dense", "units": 10, "activation": "linear"},
]


def test_supervisor_tunes_hyperparameters_then_prunes_once_the_target_is_met(sessions) -> None:
    cluster = FakeCluster()
    cluster.nas_architecture = {**NAS_ARCHITECTURE, "id": "nas-cnn", "layers": CNN_LAYERS}
    cluster.accuracy["fine_tune"] = 0.955
    slm = FakeSLM([
        tool_call("run_nas", {}, "Search a better CNN."),
        tool_call("run_hpo", {}, "Tune the learning rate of the searched CNN."),
        tool_call("train_final", {}, "Retrain with the tuned CNN."),
    ])
    supervisor = make_supervisor(sessions, cluster, slm)
    run = supervisor.create_run(default_goal(target_accuracy=0.95))

    finished = run_until_done(supervisor, run["id"])

    assert finished["status"] == "completed"
    assert finished["result"]["outcome"] == "objective_achieved"
    names = [action["action"] for action in cluster.actions]
    hpo = cluster.actions[names.index("run_hpo")]
    assert hpo["parameters"]["architecture"]["id"] == "nas-cnn"
    retrain = [action for action in cluster.actions if action["action"] == "start_training" and "architecture" in action["parameters"]]
    assert retrain[-1]["parameters"]["architecture"]["id"] == "hpo-adam-lr3e-3-bs64"
    # Target met -> the rule prunes once, then a full-length fine-tune runs without a new architecture.
    prune_index = names.index("prune_model")
    assert cluster.actions[prune_index]["parameters"]["ratio"] == 0.3
    fine_tune = next(action for action in cluster.actions[prune_index:] if action["action"] == "start_training")
    assert "architecture" not in fine_tune["parameters"]
    assert fine_tune["parameters"]["rounds"] == 10
    pruning = finished["state"]["pruning"]
    assert pruning["compression"] == 2.5
    assert pruning["architecture_id"] == "hpo-adam-lr3e-3-bs64-pruned30"
    assert pruning["validation_accuracy"] == pytest.approx(0.955)
    assert pruning["accepted"] is True
    assert names.count("prune_model") == 1
    decisions = [event for event in finished["events"] if event["event_type"] == "decision"]
    assert [event["details"]["source"] for event in decisions][-2:] == ["rule", "rule"]


def test_baseline_architecture_enables_hpo_without_nas(sessions) -> None:
    cluster = FakeCluster()
    cluster.model_report = {"architecture": {"model_type": "custom", "id": "baseline", "layers": CNN_LAYERS}}
    slm = FakeSLM([tool_call("finish_workflow", {"outcome": "no_worthwhile_action", "summary": "stop"}, "stop")])
    supervisor = make_supervisor(sessions, cluster, slm)
    run = supervisor.create_run(default_goal(target_accuracy=0.99))

    run_until_done(supervisor, run["id"])

    offered = {tool["function"]["name"] for tool in slm.calls[0][1]}
    assert {"run_nas", "run_hpo", "optimize_topology"} <= offered
    # Below the target, compressing the model is not offered while other steps can help.
    assert "prune_model" not in offered


def test_learn_topology_trains_on_the_apg_graph_and_keeps_it_when_it_helps(sessions) -> None:
    cluster = FakeCluster()
    slm = FakeSLM([
        tool_call("learn_topology", {}, "Peers learn slowly on non-IID data."),
        tool_call("finish_workflow", {"outcome": "no_worthwhile_action", "summary": "done"}, "done"),
    ])
    supervisor = make_supervisor(sessions, cluster, slm)
    run = supervisor.create_run(default_goal(target_accuracy=0.99))

    finished = run_until_done(supervisor, run["id"])

    topologies = [action["parameters"]["topology"] for action in cluster.actions if action["action"] == "set_topology"]
    assert topologies == ["directed_ring", "learned"]
    assert finished["state"]["selected_topology"] == "learned"
    assert "learn_topology" in finished["state"]["completed_steps"]


def test_hpo_budget_fits_one_worker_search_request(sessions) -> None:
    cluster = FakeCluster()
    cluster.nas_architecture = {**NAS_ARCHITECTURE, "id": "nas-cnn", "layers": CNN_LAYERS}
    slm = FakeSLM([
        tool_call("run_nas", {}, "search"),
        tool_call("run_hpo", {}, "tune"),
        tool_call("finish_workflow", {"outcome": "no_worthwhile_action", "summary": "done"}, "done"),
    ])
    supervisor = make_supervisor(sessions, cluster, slm)
    run_until_done(supervisor, supervisor.create_run(default_goal(target_accuracy=0.99))["id"])
    hpo = next(action for action in cluster.actions if action["action"] == "run_hpo")
    assert hpo["parameters"]["search_time_limit_seconds"] <= 300


def test_prepare_data_keeps_each_dataset_at_its_native_size(sessions) -> None:
    for dataset, size in (("mnist", 28), ("cifar10", 32)):
        cluster = FakeCluster()
        slm = FakeSLM([tool_call("finish_workflow", {"outcome": "no_worthwhile_action", "summary": "x"}, "x")])
        supervisor = make_supervisor(sessions, cluster, slm)
        run_until_done(supervisor, supervisor.create_run(default_goal(dataset=dataset, target_accuracy=0.99))["id"])
        prepare = next(action for action in cluster.actions if action["action"] == "preprocess_dataset")
        assert prepare["parameters"]["target_size"] == {"height": size, "width": size}


def test_pruning_is_not_offered_when_the_newest_model_is_not_the_best() -> None:
    from app.services.supervisor_policy import can_prune

    state = new_state()
    state["completed_steps"] = ["baseline_training", "train_final"]
    state["last_trained_architecture"] = {"model_type": "custom", "layers": CNN_LAYERS}
    state["experiments"] = [
        {"step": "learn_topology", "kind": "decentralized", "status": "completed", "group_id": "g1", "validation_accuracy": 0.56},
        {"step": "train_final", "kind": "decentralized", "status": "completed", "group_id": "g2", "validation_accuracy": 0.48},
    ]
    state["best"] = {"validation_accuracy": 0.56, "group_id": "g1"}
    assert not can_prune(state)
    state["best"] = {"validation_accuracy": 0.48, "group_id": "g2"}
    assert can_prune(state)
