"""Supervisor agent: decision contract and end-to-end workflow with fake cluster/SLM."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import AutonomyState, Base, Device
from app.services.supervisor import Supervisor, SupervisorError
from app.services.supervisor_policy import (
    DecisionError,
    allowed_steps,
    default_goal,
    nas_parameters,
    run_augmentation,
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

    # The master sweeps topologies on either dataset, so CIFAR-10 is offered it too.
    cifar = default_goal(dataset="cifar10")
    assert "optimize_topology" in allowed_steps(cifar, state)


def test_cifar_trainings_use_flip_shift_and_mnist_trains_on_raw_images() -> None:
    assert run_augmentation(default_goal(dataset="cifar10")) == "flip_shift"
    assert run_augmentation(default_goal(dataset="mnist")) == "none"
    assert run_augmentation(default_goal(dataset="cifar10", augmentation="none")) == "none"


def test_nas_v2_searches_on_the_shared_split_with_the_baseline_as_seed() -> None:
    state = new_state()
    state["baseline_architecture"] = {"model_type": "custom", "layers": CNN_LAYERS}
    cifar = nas_parameters(default_goal(dataset="cifar10", seed=17), state)
    assert cifar["objective"] == "federated"
    assert cifar["search_space"] == "extended"
    assert cifar["augmentation"] == "flip_shift"
    assert cifar["seed"] == 17
    assert cifar["seed_architectures"] == [state["baseline_architecture"]]
    mnist = nas_parameters(default_goal(), state)
    assert mnist["search_space"] == "compact" and mnist["augmentation"] == "none"
    # v1 is the former local-objective search, kept for ablations.
    v1 = nas_parameters(default_goal(dataset="cifar10", nas_version="v1"), state)
    assert set(v1) == {"enabled", "dataset_name", "search_time_limit_seconds", "seed"}
    # A non-custom baseline cannot seed the search.
    assert "seed_architectures" not in nas_parameters(default_goal(), new_state())


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
        self.result_queries: list[str] = []
        self.groups: list[dict[str, Any]] = []
        self.accuracy = {"baseline_training": 0.90, "train_final": 0.96, "continue": 0.90}
        self.fail_next: dict[str, str] = {}
        self.cleanups = 0
        self.counter = 0
        self.model_report: dict[str, Any] = {}
        self.nas_architecture: dict[str, Any] = NAS_ARCHITECTURE
        self.pruned = False

    def dispatch(self, action: dict[str, Any], reason: str, run_id: str) -> dict[str, Any]:
        self.actions.append(action)
        name = action["action"]
        if name == "create_initial_model" and action["parameters"].get("from_trained_model"):
            self.warm_started = True
            report = {"warm_started": True, "source_worker": action["parameters"].get("source_worker", "worker1")}
            return {"status": "executed", "result": {"available": True, "data": report}}
        if name == "create_initial_model" and action["parameters"].get("best_checkpoint") == "install":
            self.checkpoint_installed = True
        if name in {"preprocess_dataset", "create_initial_model"}:
            return {"status": "executed", "result": {"available": True, "data": {"dataset_name": "mnist", **self.model_report}}}
        if name == "prune_model" and action["parameters"].get("restore_source"):
            self.pruned = False
            return {"status": "executed", "result": {"available": True, "data": {"restored": True}}}
        if name == "prune_model":
            self.pruned = True
            self.prune_ratio = action["parameters"]["ratio"]
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
                "_ratio": getattr(self, "prune_ratio", None),
                "_warm": name == "start_training" and getattr(self, "warm_started", False),
                "_continue": name == "start_training" and getattr(self, "checkpoint_installed", False),
            }
        )
        if name == "start_training":
            self.checkpoint_installed = False
        return {
            "status": "executed",
            "result": {"action": action, "result": {"available": True, "data": {"accepted": True, "group_id": group_id}}},
        }

    def results(self, group_id: str | None = None) -> dict[str, Any]:
        if group_id is not None:
            self.result_queries.append(group_id)
        # Every running experiment finishes when the supervisor next looks at it.
        for group in self.groups:
            if group["status"] != "running" or group.get("_seen") is None:
                group["_seen"] = True
                continue
            action = group["_action"]
            if group["kind"] == "decentralized":
                step = (
                    "continue"
                    if group.get("_continue")
                    else "fine_tune"
                    if group["_fine_tune"]
                    else "warm_final"
                    if group.get("_warm")
                    else "train_final"
                    if "architecture" in action["parameters"]
                    else "baseline_training"
                )
                if step in self.fail_next:
                    group.update(status="failed", error=self.fail_next.pop(step))
                    continue
                if step == "fine_tune" and f"fine_tune_{group['_ratio']}" in self.accuracy:
                    accuracy = self.accuracy[f"fine_tune_{group['_ratio']}"]
                else:
                    accuracy = self.accuracy[step]
                if isinstance(accuracy, list):
                    accuracy = accuracy.pop(0)
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

    def cleanup(self, recycle: bool = False) -> dict[str, Any]:
        # Count remediation cleanups separately from the once-per-run recycle check.
        if not recycle:
            self.cleanups += 1
        self.recycle_requests = getattr(self, "recycle_requests", 0) + int(recycle)
        return {"complete": True, "recycled": list(getattr(self, "recycle_next", []))} if recycle else {"complete": True}


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


def make_supervisor(
    sessions,
    cluster: FakeCluster,
    slm: FakeSLM,
    clock: Clock | None = None,
    global_evaluate=None,
    sync_workers=None,
) -> Supervisor:
    return Supervisor(
        sessions,
        dispatch=cluster.dispatch,
        master_results=cluster.results,
        model_call=slm,
        memory_cleanup=cluster.cleanup,
        global_evaluate=global_evaluate,
        sync_workers=sync_workers,
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
    assert cluster.actions[retrain_index]["parameters"]["rounds"] == 10  # never shorter than the goal (10 rounds)
    # The SLM chose the aggregation rule; synchronization comes from the goal.
    assert cluster.actions[retrain_index]["parameters"]["aggregation"] == "metropolis"
    assert cluster.actions[retrain_index]["parameters"]["sync_mode"] == "semi_sync"
    assert cluster.actions[retrain_index]["parameters"]["sync_deadline_seconds"] == 300
    baseline = next(action for action in cluster.actions if action["action"] == "start_training")
    assert baseline["parameters"]["aggregation"] == "fedavg"
    # Monitoring fetches only the experiment it watches, never all master groups.
    assert cluster.result_queries and set(cluster.result_queries) <= {group["group_id"] for group in cluster.groups}
    # MNIST (the default goal) trains on raw images in every training step.
    assert {a["parameters"]["augmentation"] for a in cluster.actions if a["action"] == "start_training"} == {"none"}
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


def test_a_coordinator_restart_mid_training_is_retried_once(sessions) -> None:
    cluster = FakeCluster()
    cluster.fail_next["baseline_training"] = (
        "Training was interrupted because the master coordinator restarted before this group completed."
    )
    cluster.accuracy["baseline_training"] = 0.97
    supervisor = make_supervisor(sessions, cluster, FakeSLM([]))
    finished = run_until_done(supervisor, supervisor.create_run(default_goal())["id"])

    assert finished["status"] == "completed"
    assert [a["action"] for a in cluster.actions].count("start_training") == 2


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
    assert hpo["parameters"]["search_time_limit_seconds"] <= 600


def test_prepare_data_keeps_each_dataset_at_its_native_size(sessions) -> None:
    for dataset, size in (("mnist", 28), ("cifar10", 32)):
        cluster = FakeCluster()
        slm = FakeSLM([tool_call("finish_workflow", {"outcome": "no_worthwhile_action", "summary": "x"}, "x")])
        supervisor = make_supervisor(sessions, cluster, slm)
        run_until_done(supervisor, supervisor.create_run(default_goal(dataset=dataset, target_accuracy=0.99))["id"])
        prepare = next(action for action in cluster.actions if action["action"] == "preprocess_dataset")
        assert prepare["parameters"]["target_size"] == {"height": size, "width": size}


def test_pruning_that_loses_too_much_accuracy_restores_the_unpruned_model(sessions) -> None:
    cluster = FakeCluster()
    cluster.nas_architecture = {**NAS_ARCHITECTURE, "id": "nas-cnn", "layers": CNN_LAYERS}
    cluster.accuracy["fine_tune"] = 0.80
    slm = FakeSLM([
        tool_call("run_nas", {}, "Search a better CNN."),
        tool_call("run_hpo", {}, "Tune the learning rate of the searched CNN."),
        tool_call("train_final", {}, "Retrain with the tuned CNN."),
    ])
    supervisor = make_supervisor(sessions, cluster, slm)
    run = supervisor.create_run(default_goal(target_accuracy=0.95))

    finished = run_until_done(supervisor, run["id"])

    assert finished["status"] == "completed"
    pruning = finished["state"]["pruning"]
    assert pruning["accepted"] is False
    assert pruning["restored_unpruned"] is True
    restores = [a for a in cluster.actions if a["action"] == "prune_model" and a["parameters"].get("restore_source")]
    # The rejected 30% prune is retried once at 15%; both are rejected and restored.
    assert len(restores) == 2
    ratios = [a["parameters"]["ratio"] for a in cluster.actions if a["action"] == "prune_model" and "ratio" in a["parameters"]]
    assert ratios == [0.3, 0.15]
    assert [item["ratio"] for item in finished["state"]["pruning_attempts"]] == [0.3, 0.15]
    assert cluster.pruned is False
    # The unpruned model stays the run's result.
    assert finished["state"]["best"]["step"] != "prune_model"


def test_rejected_pruning_is_retried_once_at_a_gentler_ratio(sessions) -> None:
    cluster = FakeCluster()
    cluster.nas_architecture = {**NAS_ARCHITECTURE, "id": "nas-cnn", "layers": CNN_LAYERS}
    cluster.accuracy["fine_tune_0.3"] = 0.80
    cluster.accuracy["fine_tune_0.15"] = 0.95
    slm = FakeSLM([
        tool_call("run_nas", {}, "Search a better CNN."),
        tool_call("run_hpo", {}, "Tune the learning rate of the searched CNN."),
        tool_call("train_final", {}, "Retrain with the tuned CNN."),
    ])
    supervisor = make_supervisor(sessions, cluster, slm)
    finished = run_until_done(supervisor, supervisor.create_run(default_goal(target_accuracy=0.95))["id"])

    assert finished["status"] == "completed"
    pruning = finished["state"]["pruning"]
    assert pruning["ratio"] == 0.15
    assert pruning["accepted"] is True
    assert pruning["architecture_id"].endswith("-pruned15")
    attempts = finished["state"]["pruning_attempts"]
    assert [(item["ratio"], item["accepted"]) for item in attempts] == [(0.3, False), (0.15, True)]
    assert cluster.pruned is True


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
    # A newest model within the accepted 2-point drop of the best may be pruned.
    state["experiments"][-1]["validation_accuracy"] = 0.5624
    state["best"] = {"validation_accuracy": 0.5638, "group_id": "g1"}
    assert can_prune(state)


def _global_result(validation: float, test: float) -> dict:
    def split(accuracy: float) -> dict:
        return {
            "workers_evaluated": 5,
            "mean_accuracy": accuracy,
            "std_accuracy": 0.01,
            "min_accuracy": accuracy - 0.02,
            "max_accuracy": accuracy + 0.02,
            "mean_macro_f1": accuracy - 0.05,
            "per_worker": {"worker1": {"accuracy": accuracy, "macro_f1": accuracy - 0.05, "per_class_recall": [1.0, 0.0]}},
        }

    return {"classes": ["0", "1"], "validation": split(validation), "test": split(test)}


def test_global_evaluation_replaces_worker_reported_accuracy_for_decisions(sessions) -> None:
    cluster = FakeCluster()
    calls = []

    def evaluate(dataset_name: str) -> dict:
        calls.append(dataset_name)
        return _global_result(0.61, 0.59)

    supervisor = make_supervisor(sessions, cluster, FakeSLM([]), global_evaluate=evaluate)
    run = supervisor.create_run(default_goal(target_accuracy=0.95, max_experiments=1))

    finished = run_until_done(supervisor, run["id"])

    baseline = finished["state"]["experiments"][0]
    assert calls and calls[0] == "mnist"
    assert baseline["validation_accuracy"] == pytest.approx(0.61)
    assert baseline["local_validation_accuracy"] != pytest.approx(0.61)
    assert baseline["test_accuracy"] == pytest.approx(0.59)
    assert baseline["test_per_worker"]["worker1"]["per_class_recall"] == [1.0, 0.0]
    assert "per_worker" not in baseline["test"]
    assert finished["state"]["best"]["validation_accuracy"] == pytest.approx(0.61)


def test_failed_global_evaluation_keeps_worker_accuracy_and_the_run_going(sessions) -> None:
    cluster = FakeCluster()

    def evaluate(dataset_name: str) -> dict:
        raise RuntimeError("evaluator missing")

    supervisor = make_supervisor(sessions, cluster, FakeSLM([]), global_evaluate=evaluate)
    run = supervisor.create_run(default_goal(target_accuracy=0.95, max_experiments=1))

    finished = run_until_done(supervisor, run["id"])

    baseline = finished["state"]["experiments"][0]
    assert finished["status"] == "completed"
    assert baseline["validation_accuracy"] == baseline["local_validation_accuracy"]
    assert "evaluator missing" in baseline["global_evaluation_error"]
    assert any(event["event_type"] == "global_evaluation_failed" for event in finished["events"])


def test_operator_evaluation_scores_newest_models_and_passes_the_run_filter(sessions) -> None:
    calls = []

    def evaluate(dataset_name: str, run_id=None) -> dict:
        calls.append((dataset_name, run_id))
        return _global_result(0.66, 0.64)

    supervisor = make_supervisor(sessions, FakeCluster(), FakeSLM([]), global_evaluate=evaluate)
    assert supervisor.evaluate_newest_models("cifar10")["test"]["mean_accuracy"] == pytest.approx(0.64)
    supervisor.evaluate_newest_models("cifar10", "mesh-seed-17-abcd1234")
    assert calls == [("cifar10", None), ("cifar10", "mesh-seed-17-abcd1234")]


def test_operator_evaluation_reports_missing_or_failing_evaluator(sessions) -> None:
    supervisor = make_supervisor(sessions, FakeCluster(), FakeSLM([]))
    with pytest.raises(SupervisorError) as missing:
        supervisor.evaluate_newest_models("mnist")
    assert missing.value.status_code == 503

    def broken(dataset_name: str, run_id=None) -> dict:
        raise RuntimeError("no worker model")

    supervisor = make_supervisor(sessions, FakeCluster(), FakeSLM([]), global_evaluate=broken)
    with pytest.raises(SupervisorError) as failed:
        supervisor.evaluate_newest_models("mnist")
    assert failed.value.status_code == 502 and "no worker model" in str(failed.value)


def test_final_training_warm_starts_from_the_best_trained_model(sessions) -> None:
    cluster = FakeCluster()
    cluster.nas_architecture = {**NAS_ARCHITECTURE, "id": "nas-cnn", "layers": CNN_LAYERS}
    cluster.accuracy.update(train_final=0.93, warm_final=0.95, fine_tune=0.90)
    slm = FakeSLM([
        tool_call("run_nas", {}, "search"),
        tool_call("learn_topology", {}, "learn the graph"),
        tool_call("train_final", {}, "final"),
    ])

    scores = iter([0.80, 0.93, 0.95, 0.90, 0.90])

    def evaluate(dataset_name: str) -> dict:
        score = next(scores)
        result = _global_result(score, score - 0.01)
        result["validation"]["per_worker"] = {"worker1": {"accuracy": 0.91}, "worker4": {"accuracy": 0.95}}
        return result

    supervisor = make_supervisor(sessions, cluster, slm, global_evaluate=evaluate)
    run = supervisor.create_run(default_goal(target_accuracy=0.99))

    finished = run_until_done(supervisor, run["id"])

    warm = [a for a in cluster.actions if a["action"] == "create_initial_model" and a["parameters"].get("from_trained_model")]
    assert len(warm) == 1
    # The source is the worker with the best *global validation* accuracy.
    assert warm[0]["parameters"]["source_worker"] == "worker4"
    final = next(a for a in cluster.actions[cluster.actions.index(warm[0]):] if a["action"] == "start_training")
    assert "architecture" not in final["parameters"]
    assert finished["state"]["warm_start"]["from_step"] == "learn_topology"


def test_no_warm_start_when_the_architecture_changed() -> None:
    from app.services.supervisor import Supervisor

    state = new_state()
    tuned = {"model_type": "custom", "id": "hpo", "layers": CNN_LAYERS}
    state["experiments"] = [{"step": "learn_topology", "kind": "decentralized", "status": "completed", "group_id": "g1"}]
    state["best"] = {"group_id": "g1", "validation_accuracy": 0.9}
    state["last_trained_architecture"] = {"model_type": "custom", "id": "nas", "layers": CNN_LAYERS}
    assert Supervisor._warm_start_source(state, tuned) is None
    state["last_trained_architecture"] = tuned
    assert Supervisor._warm_start_source(state, tuned)["from_group_id"] == "g1"
    state["best"] = {"group_id": "g0", "validation_accuracy": 0.95}
    assert Supervisor._warm_start_source(state, tuned) is None


def test_runs_start_by_recycling_bloated_workers_and_wait_for_them(sessions) -> None:
    from datetime import timedelta

    cluster = FakeCluster()
    cluster.recycle_next = ["worker2"]
    clock = Clock()
    supervisor = make_supervisor(sessions, cluster, FakeSLM([]), clock=clock)
    run = supervisor.create_run(default_goal(target_accuracy=0.95, max_experiments=1))

    supervisor.tick()
    waiting = supervisor.get_run(run["id"])
    assert cluster.recycle_requests == 1
    assert waiting["state"]["stage"] == "preflight"
    assert waiting["state"]["workers_recycled"] == ["worker2"]

    clock.now += timedelta(seconds=60)
    finished = run_until_done(supervisor, run["id"])
    # Recycling happens once per run, then the run proceeds normally.
    assert cluster.recycle_requests == 1
    assert finished["status"] == "completed"


def test_topology_sweep_is_a_short_screening_run(sessions) -> None:
    cluster = FakeCluster()
    slm = FakeSLM([
        tool_call("optimize_topology", {}, "Compare peer topologies on the non-IID shards."),
        tool_call("finish_workflow", {"outcome": "no_worthwhile_action", "summary": "Done."}, "Done."),
    ])
    supervisor = make_supervisor(sessions, cluster, slm)
    run_until_done(supervisor, supervisor.create_run(default_goal(target_accuracy=0.99))["id"])
    sweep = next(action for action in cluster.actions if action["action"] == "optimize_topology")
    # Every candidate topology is trained, so the sweep screens with few rounds.
    assert sweep["parameters"]["rounds"] == 4
    assert sweep["parameters"]["dataset_name"] == "mnist"
    assert sweep["parameters"]["alphas"] == [0.3]
    assert sweep["parameters"]["augmentation"] == "none"


def test_workers_left_unregistered_by_a_failed_sync_are_re_registered(sessions) -> None:
    # A backend restart while a master group runs leaves every worker not ready (HTTP 409).
    with sessions.begin() as session:
        for device in session.scalars(select(Device).where(Device.role == "worker")):
            device.training_ready = False
            device.training_error = "Master training peer sync failed (HTTP 409)"
    calls: list[int] = []

    def sync_workers() -> None:
        calls.append(1)
        with sessions.begin() as session:
            for device in session.scalars(select(Device).where(Device.role == "worker")):
                device.training_ready = True
                device.training_error = None
        return None

    cluster = FakeCluster()
    cluster.accuracy["baseline_training"] = 0.97
    supervisor = make_supervisor(sessions, cluster, FakeSLM([]), sync_workers=sync_workers)
    finished = run_until_done(supervisor, supervisor.create_run(default_goal())["id"])

    assert finished["status"] == "completed"
    assert len(calls) == 1
    assert any("Re-registered the workers" in event["message"] for event in finished["events"])


# --------------------------------------------------------------------------- continuation + best model


def _continuation_state() -> dict[str, Any]:
    state = new_state()
    state["completed_steps"] = ["baseline_training", "run_nas", "run_hpo", "learn_topology", "optimize_topology", "train_final"]
    state["nas_architecture"] = {**NAS_ARCHITECTURE, "layers": CNN_LAYERS}
    state["selected_topology"] = "learned"
    state["trained_configs"] = ["b2_c16_k5|learned"]
    state["last_trained_architecture"] = state["nas_architecture"]
    state["experiments"] = [
        {"step": "train_final", "kind": "decentralized", "status": "completed", "group_id": "g1", "validation_accuracy": 0.66}
    ]
    state["best"] = {"validation_accuracy": 0.66, "group_id": "g1"}
    state["best_checkpoint"] = {"group_id": "g1", "architecture": state["nas_architecture"]}
    return state


def test_continuation_is_offered_before_pruning_only_while_it_keeps_helping() -> None:
    goal = default_goal(dataset="cifar10", target_accuracy=0.70, max_experiments=12)
    state = _continuation_state()
    assert allowed_steps(goal, state) == ["continue_training", "finish_workflow"]
    assert fallback_decision(goal, state).step == "continue_training"

    state["continuations"] = [{"validation_accuracy": 0.68, "improved": True}]
    assert allowed_steps(goal, state)[0] == "continue_training"
    # A continuation without gain, or the second one, ends continuation; pruning is next.
    state["continuations"] = [{"validation_accuracy": 0.661, "improved": False}]
    assert allowed_steps(goal, state) == ["prune_model", "finish_workflow"]
    state["continuations"] = [{"improved": True}, {"improved": True}]
    assert "continue_training" not in allowed_steps(goal, state)

    # Never when the target is met, without a checkpoint, or after a prune.
    state = _continuation_state()
    state["best"]["validation_accuracy"] = 0.71
    assert "continue_training" not in allowed_steps(goal, state)
    state = _continuation_state()
    state["best_checkpoint"] = None
    assert "continue_training" not in allowed_steps(goal, state)
    state = _continuation_state()
    state["completed_steps"].append("prune_model")
    assert "continue_training" not in allowed_steps(goal, state)


def test_missed_target_continues_from_the_best_checkpoint_and_ends_with_it_installed(sessions) -> None:
    cluster = FakeCluster()
    cluster.nas_architecture = {**NAS_ARCHITECTURE, "id": "nas-cnn", "layers": CNN_LAYERS}
    cluster.accuracy.update(train_final=0.96, warm_final=0.96, fine_tune=0.97)
    cluster.accuracy["continue"] = [0.975, 0.976]
    supervisor = make_supervisor(sessions, cluster, FakeSLM([]))
    run = supervisor.create_run(default_goal(target_accuracy=0.99, max_experiments=10))

    finished = run_until_done(supervisor, run["id"], ticks=80)

    assert finished["status"] == "completed"
    state = finished["state"]
    assert [(item["validation_accuracy"], item["improved"]) for item in state["continuations"]] == [
        (0.975, True),
        (0.976, False),
    ]
    actions = cluster.actions
    continued = [
        index
        for index, action in enumerate(actions)
        if action["action"] == "create_initial_model" and action["parameters"].get("best_checkpoint") == "install"
    ]
    # Two continuations plus the final install, all of this run's checkpoint.
    assert len(continued) == 3
    assert all(actions[index]["parameters"]["checkpoint_id"] == run["id"] for index in continued)
    for index in continued[:2]:
        training = next(action for action in actions[index:] if action["action"] == "start_training")
        assert "architecture" not in training["parameters"]
        assert training["parameters"]["rounds"] == 10
    assert actions[-1]["parameters"].get("best_checkpoint") == "install"
    saves = [a for a in actions if a["action"] == "create_initial_model" and a["parameters"].get("best_checkpoint") == "save"]
    # New bests: baseline, learned topology, both continuations, then the accepted prune.
    assert len(saves) == 5
    # Continuation ran out before pruning; the accepted prune is the model left on the workers.
    assert state["pruning"]["accepted"] is True
    deployed = finished["result"]["deployed_model"]
    assert deployed["step"] == "prune_model" and deployed["installed"] is True
    assert any(event["event_type"] == "model_deployed" for event in finished["events"])


def test_the_checkpoint_comes_from_the_best_globally_validated_worker(sessions) -> None:
    cluster = FakeCluster()

    def evaluate(dataset_name: str) -> dict:
        result = _global_result(0.92, 0.91)
        result["validation"]["per_worker"] = {"worker1": {"accuracy": 0.90}, "worker3": {"accuracy": 0.94}}
        result["test"]["per_worker"] = {"worker1": {"accuracy": 0.89}, "worker3": {"accuracy": 0.93}}
        return result

    supervisor = make_supervisor(sessions, cluster, FakeSLM([tool_call("finish_workflow", {"outcome": "no_worthwhile_action", "summary": "done"})]), global_evaluate=evaluate)
    finished = run_until_done(supervisor, supervisor.create_run(default_goal(target_accuracy=0.99))["id"])

    save = next(a for a in cluster.actions if a["parameters"].get("best_checkpoint") == "save")
    assert save["parameters"]["source_worker"] == "worker3"
    deployed = finished["result"]["deployed_model"]
    assert deployed["source_worker"] == "worker3"
    assert deployed["validation_accuracy"] == pytest.approx(0.94)
    assert deployed["test_accuracy"] == pytest.approx(0.93)
    assert deployed["step"] == "baseline_training" and deployed["installed"] is True


def test_a_failed_checkpoint_save_disables_continuation_and_the_final_install(sessions) -> None:
    class NoCheckpoint(FakeCluster):
        def dispatch(self, action, reason, run_id):
            if action["parameters"].get("best_checkpoint") == "save":
                self.actions.append(action)
                return {"status": "executed", "result": {"available": False, "message": "worker1 unreachable"}}
            return super().dispatch(action, reason, run_id)

    cluster = NoCheckpoint()
    supervisor = make_supervisor(sessions, cluster, FakeSLM([tool_call("finish_workflow", {"outcome": "no_worthwhile_action", "summary": "done"})]))
    finished = run_until_done(supervisor, supervisor.create_run(default_goal(target_accuracy=0.99))["id"])

    assert finished["status"] == "completed"
    assert finished["state"]["best_checkpoint"] is None
    assert finished["result"]["deployed_model"] is None
    assert not any(a["parameters"].get("best_checkpoint") == "install" for a in cluster.actions)
    assert any("best model checkpoint" in event["message"] for event in finished["events"] if event["event_type"] == "warning")



def test_the_slm_cannot_claim_the_objective_when_the_target_is_missed(sessions) -> None:
    cluster = FakeCluster()
    slm = FakeSLM([tool_call("finish_workflow", {"outcome": "objective_achieved", "summary": "Close enough to the target."})])
    supervisor = make_supervisor(sessions, cluster, slm)
    finished = run_until_done(supervisor, supervisor.create_run(default_goal(target_accuracy=0.99))["id"])

    assert finished["result"]["outcome"] == "no_worthwhile_action"
    assert finished["result"]["summary"].startswith("Target 0.9900 not met (best validation accuracy 0.9000).")
    assert finished["status"] == "completed"
