"""Replay simulator of supervisor runs, built from the measured fair-campaign evidence.

Each fair run measured the effect of every search step on the global validation accuracy
with one training budget. The simulator replays a supervisor run for one of those
(dataset, split, seed) settings: it keeps a state dict in the supervisor's own format, so
``allowed_steps``, ``rule_decision``, ``prompt_state`` and ``tool_definitions`` from the
backend give exactly the prompt and tools the SLM sees in a real run. Applying a step
moves the state the way the supervisor would, with the accuracy the run measured.

Accuracy model: a configuration's validation accuracy is the run's baseline plus the
measured value of its topology plus the measured effects of the other applied changes
(NAS or HPO architecture, aggregation, sync). Effects are measured against the
configuration active before each step, so this is an additive approximation for orders
the run did not try.

``teacher_decision`` is the labelling policy for the fine-tuning dataset: it ranks the
allowed steps by their mean measured effect at the run's label skew in the training runs.
"""

from __future__ import annotations

import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "dtn" / "backend", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.services.supervisor_policy import (  # noqa: E402
    SEARCH_STEPS,
    Decision,
    allowed_steps,
    decision_messages,
    default_goal,
    experiments_used,
    final_training_config,
    new_state,
    prompt_state,
    rule_decision,
    tool_definitions,
    validate_step,
)
from evidence_runs import campaign_runs  # noqa: E402
from fair_summary import step_effects  # noqa: E402

DATASETS = ("mnist", "fashion_mnist", "femnist", "cifar10")
SPLITS = ("dirichlet_a01", "dirichlet_a05", "dirichlet_a09")
SKEW = {"dirichlet_a01": "strong label skew (α = 0.1)", "dirichlet_a05": "moderate label skew (α = 0.5)",
        "dirichlet_a09": "mild label skew (α = 0.9)"}
LABEL = {"run_nas": "NAS", "run_hpo": "HPO", "optimize_topology": "the topology sweep",
         "optimize_aggregation": "the aggregation sweep", "optimize_sync": "the sync sweep",
         "learn_topology": "the learned topology (APG)"}
# A step is worth its experiment when it gained at least this much on average (points).
MIN_GAIN_POINTS = 1.0
# Keep-or-revert tolerance of the fair runs (accept_tolerance).
ACCEPT_TOLERANCE = 0.005
# Typical minutes per step on the 8-Jetson cluster (10 rounds x 5 epochs trainings).
DURATION = {"baseline_training": 25, "run_nas": 25, "run_hpo": 20, "optimize_topology": 30,
            "optimize_aggregation": 25, "optimize_sync": 25, "learn_topology": 30, "measure_step": 25,
            "train_final": 25, "continue_training": 25, "prune_model": 20}
STEP_KIND = {"run_nas": "dataset_architecture_search", "run_hpo": "hyperparameter_search",
             "optimize_topology": "topology_optimization", "optimize_aggregation": "topology_optimization",
             "optimize_sync": "topology_optimization"}
FAILURE_MESSAGES = ("Worker worker3 timed out after 900 s", "CUDA out of memory on worker5",
                    "Experiment cancelled: worker6 went offline")
LAYERS = [{"type": "conv2d"}, {"type": "max_pooling2d"}, {"type": "conv2d"}, {"type": "flatten"}, {"type": "dense"}]


@dataclass
class RunEvidence:
    """What one fair run measured."""

    dataset: str
    split: str
    seed: int
    run_id: str
    base_val: float
    base_f1: float
    effects: dict[str, float]          # non-topology step -> validation change
    topology_value: dict[str, float]   # topology name -> value over the directed ring
    choices: dict[str, Any]


def _architecture(architecture_id: str) -> dict[str, Any]:
    return {"model_type": "custom", "id": architecture_id, "parameter_count": None, "layers": [dict(x) for x in LAYERS]}


def load_evidence() -> list[RunEvidence]:
    runs = []
    for dataset in DATASETS:
        for split in SPLITS:
            for run in campaign_runs(dataset, split):
                evidence = _evidence(dataset, split, run)
                if evidence:
                    runs.append(evidence)
    return runs


def _evidence(dataset: str, split: str, run: dict) -> RunEvidence | None:
    steps = [s for s in run["steps"] if s.get("status") == "completed"]
    base = next((s for s in steps if s["step"] == "baseline_training"), None)
    if base is None or not isinstance(base.get("validation_accuracy"), (int, float)):
        return None
    effects = {kind: item["d_val"] for kind, item in step_effects(run).items()}
    choices: dict[str, Any] = {"topology": "mesh", "aggregation": "fedavg", "sync": "semi_sync",
                               "nas": "nas-searched", "hpo": "hpo-tuned"}
    topology_value = {"directed_ring": 0.0}
    active_topology = "directed_ring"
    for s in steps:
        name = s["step"]
        if name == "run_nas" and s.get("architecture"):
            choices["nas"] = s["architecture"]
        if name == "run_hpo" and s.get("architecture"):
            choices["hpo"] = s["architecture"]
        if name == "optimize_topology" and s.get("topology"):
            choices["topology"] = s["topology"]
        if name == "measure_step" and s.get("measures") == "optimize_aggregation" and s.get("aggregation"):
            choices["aggregation"] = s["aggregation"]
        if name == "measure_step" and s.get("measures") == "optimize_sync" and s.get("sync_mode"):
            choices["sync"] = s["sync_mode"]
        # Topology values are chained: each topology step is measured against the active one.
        if name == "measure_step" and s.get("measures") == "optimize_topology" and "optimize_topology" in effects:
            topology_value[choices["topology"]] = topology_value.get(active_topology, 0.0) + effects["optimize_topology"]
            if s.get("accepted") is not False:
                active_topology = choices["topology"]
        if name == "learn_topology" and "learn_topology" in effects:
            topology_value["learned"] = topology_value.get(active_topology, 0.0) + effects["learn_topology"]
    other = {k: v for k, v in effects.items() if k not in {"optimize_topology", "learn_topology"}}
    return RunEvidence(dataset, split, int(run["seed"]), str(run["run_id"]), float(base["validation_accuracy"]),
                       float(base.get("macro_f1") or base["validation_accuracy"] - 0.05), other, topology_value, choices)


def priors(runs: list[RunEvidence]) -> dict[str, dict[str, float]]:
    """Mean measured effect (points) of every search step per split, over the given runs."""
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        for step, effect in run.effects.items():
            values[run.split][step].append(100 * effect)
        if run.choices["topology"] in run.topology_value:
            values[run.split]["optimize_topology"].append(100 * run.topology_value[run.choices["topology"]])
        if "learned" in run.topology_value:
            best_fixed = max(v for k, v in run.topology_value.items() if k != "learned")
            values[run.split]["learn_topology"].append(100 * (run.topology_value["learned"] - best_fixed))
    return {split: {step: statistics.fmean(v) for step, v in steps.items()} for split, steps in values.items()}


@dataclass
class Simulator:
    run: RunEvidence
    goal: dict[str, Any]
    rng: random.Random
    failure_rate: float = 0.04
    state: dict[str, Any] = field(default_factory=new_state)
    minutes_left: float = 0.0
    applied: dict[str, float] = field(default_factory=dict)
    topology: str = "directed_ring"
    min_free_mb: int = 4000

    def __post_init__(self) -> None:
        self.minutes_left = float(self.goal["time_budget_minutes"])
        self.min_free_mb = self.rng.randint(1800, 5200)
        baseline = _architecture("baseline")
        self.state["baseline_architecture"] = baseline
        self.state["last_trained_architecture"] = baseline
        self._train("baseline_training", self.run.base_val, architecture="baseline")
        self.state["best_checkpoint"] = {"architecture": baseline}
        self.state["trained_configs"].append(final_training_config(self.goal, self.state))

    # ------------------------------------------------------------------ accuracy model
    def value(self) -> float:
        return self.run.base_val + self.run.topology_value.get(self.topology, 0.0) + sum(self.applied.values())

    def _noisy(self, value: float) -> float:
        return round(min(0.999, max(0.05, value + self.rng.gauss(0, 0.002))), 4)

    def _best(self) -> float:
        return float((self.state.get("best") or {}).get("validation_accuracy") or 0.0)

    def _train(self, step: str, accuracy: float, architecture: str | None = None) -> float:
        accuracy = self._noisy(accuracy)
        nas = self.state.get("nas_architecture")
        record = {
            "step": step, "status": "completed", "kind": "decentralized", "validation_accuracy": accuracy,
            "macro_f1": round(max(0.0, accuracy - self.run.base_val + self.run.base_f1), 4),
            "architecture": architecture or (nas or {}).get("id") or "baseline",
            "topology": self.topology, "aggregation": self.state.get("selected_aggregation") or self.goal["aggregation"],
            "partial_rounds": 0, "group_id": f"sim-{len(self.state['experiments'])}",
        }
        self.state["experiments"].append(record)
        if step not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step)
        if accuracy > self._best():
            self.state["best"] = {"validation_accuracy": accuracy, "macro_f1": record["macro_f1"],
                                  "group_id": record["group_id"], "step": step,
                                  "architecture": record["architecture"], "topology": self.topology}
        self.minutes_left -= DURATION[step] * self.rng.uniform(0.8, 1.25)
        return accuracy

    # ------------------------------------------------------------------ steps
    def prompt(self) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[str]]:
        allowed = allowed_steps(self.goal, self.state)
        cluster = {"workers_ready": 8, "min_free_mb": self.min_free_mb}
        messages = decision_messages(prompt_state(self.goal, self.state, cluster, self.minutes_left))
        return messages, tool_definitions(allowed), allowed

    def rule(self) -> Decision | None:
        return rule_decision(self.goal, self.state, self.minutes_left)

    def apply(self, decision: Decision) -> bool:
        """Apply one decision; False when the run is finished."""
        step = decision.step
        if step == "finish_workflow":
            return False
        if step in STEP_KIND or step == "learn_topology" or step in {"train_final", "continue_training"}:
            if self.rng.random() < self.failure_rate:
                self.state["experiments"].append({"step": step, "status": "failed", "kind": STEP_KIND.get(step, "decentralized"),
                                                  "error": self.rng.choice(FAILURE_MESSAGES)})
                failures = self.state["failures"]
                failures[step] = int(failures.get(step, 0)) + 1
                self.minutes_left -= 10
                return True
        if step in STEP_KIND:
            self._search(step)
        elif step == "learn_topology":
            self._learn_topology()
        elif step == "train_final":
            self._train("train_final", self.value())
            self.state["trained_configs"].append(final_training_config(self.goal, self.state))
            self.state["last_trained_architecture"] = self.state.get("nas_architecture") or self.state["baseline_architecture"]
            self.state["best_checkpoint"] = {"architecture": self.state["last_trained_architecture"]}
        elif step == "continue_training":
            before = self._best()
            accuracy = self._train("continue_training", before + self.rng.uniform(-0.003, 0.015))
            gain = accuracy - before
            self.state["continuations"].append({"validation_accuracy": accuracy, "previous_best": before,
                                                "gain": round(gain, 6), "improved": gain >= 0.005})
        elif step == "prune_model":
            drop = self.rng.uniform(0.0, 0.03)
            accuracy = self._noisy(self._best() - drop)
            self.state["experiments"].append({"step": "prune_model", "status": "completed", "kind": "decentralized",
                                              "validation_accuracy": accuracy, "topology": self.topology,
                                              "architecture": "pruned30", "partial_rounds": 0})
            self.state["completed_steps"].append("prune_model")
            self.state["pruning"] = {"accepted": drop <= 0.02, "compression": 2.0 if drop <= 0.02 else None}
            self.minutes_left -= DURATION["prune_model"]
        else:
            raise ValueError(f"Unknown step {step}")
        return True

    def _search(self, step: str) -> None:
        choices = self.run.choices
        record: dict[str, Any] = {"step": step, "status": "completed", "kind": STEP_KIND[step]}
        snapshot = (dict(self.applied), self.topology, {k: self.state.get(k) for k in
                    ("nas_architecture", "selected_topology", "selected_aggregation", "selected_sync_mode")})
        if step in {"run_nas", "run_hpo"}:
            architecture = choices["nas" if step == "run_nas" else "hpo"]
            self.state["nas_architecture"] = _architecture(architecture)
            record["architecture"] = architecture
            # HPO tunes the NAS result, so its effect adds to NAS's.
            self.applied[step] = self.run.effects.get(step, 0.0)
        elif step == "optimize_topology":
            self.topology = choices["topology"]
            self.state["selected_topology"] = self.topology
            record["topology"] = self.topology
        elif step == "optimize_aggregation":
            self.state["selected_aggregation"] = choices["aggregation"]
            record["aggregation"] = choices["aggregation"]
            self.applied[step] = self.run.effects.get(step, 0.0)
        else:
            self.state["selected_sync_mode"] = choices["sync"]
            self.applied[step] = self.run.effects.get(step, 0.0)
        self.state["experiments"].append(record)
        self.state["completed_steps"].append(step)
        self.minutes_left -= DURATION[step] * self.rng.uniform(0.8, 1.25)
        if self.goal.get("measure_each_step"):
            best = self._best()
            accuracy = self._train("measure_step", self.value())
            self.state["experiments"][-1]["measures"] = step
            if accuracy < best - ACCEPT_TOLERANCE:
                self.applied, self.topology, selections = snapshot
                self.state.update(selections)
            self.state["last_trained_architecture"] = self.state.get("nas_architecture") or self.state["baseline_architecture"]

    def _learn_topology(self) -> None:
        previous_topology, best = self.topology, self._best()
        self.topology = "learned"
        if "learned" not in self.run.topology_value:
            self.run.topology_value["learned"] = self.run.topology_value.get(previous_topology, 0.0)
        accuracy = self._train("learn_topology", self.value())
        if accuracy >= best:
            self.state["selected_topology"] = "learned"
        else:
            self.topology = previous_topology


# ---------------------------------------------------------------------- teacher policy
def _reason(text: str) -> str:
    return text if len(text) <= 300 else text[:297] + "..."


def teacher_decision(goal: dict[str, Any], state: dict[str, Any], minutes_left: float, allowed: list[str],
                     prior: dict[str, float]) -> Decision:
    """The step the training evidence supports, with a one-sentence reason."""
    split = goal.get("dataset_split")
    skew = SKEW.get(split, "this label skew")
    best = float((state.get("best") or {}).get("validation_accuracy") or 0.0)
    gain = {step: prior.get(step, 0.0) for step in allowed if step in SEARCH_STEPS}
    ranked = sorted(gain, key=lambda s: -gain[s])
    fair = bool(goal.get("measure_each_step"))
    left = int(goal["max_experiments"]) - experiments_used(state)

    def choose(step: str, reason: str, **arguments: Any) -> Decision:
        arguments = validate_step(step, {**arguments, "reason": _reason(reason)}, allowed)
        return Decision(step, arguments, arguments.get("reason", ""), "teacher")

    def search_reason(step: str) -> str:
        g = gain[step]
        if step == "optimize_topology":
            return (f"With {skew} the workers learn slowly from their few ring peers; a better topology was the "
                    f"largest gain in earlier runs (about {g:+.1f} points).")
        if fair and g < MIN_GAIN_POINTS:
            return (f"This fair run measures every search step; {LABEL[step]} is the most useful one left "
                    f"(about {g:+.1f} points in earlier runs at {skew}).")
        return f"{LABEL[step][0].upper() + LABEL[step][1:]} is the largest expected gain left at {skew} (about {g:+.1f} points in earlier runs)."

    if fair:
        if ranked:
            return choose(ranked[0], search_reason(ranked[0]))
        if "prune_model" in allowed:
            return choose("prune_model", f"Every search step is measured; compress the best model ({best:.1%}) for the Jetsons.")
        return choose("finish_workflow", f"Every search step is measured; the best model reaches {best:.1%}.",
                      outcome="no_worthwhile_action", summary=f"All search steps measured; best validation accuracy {best:.4f}.")

    worthwhile = [s for s in ranked if gain[s] >= MIN_GAIN_POINTS]
    has_change = bool(state.get("nas_architecture") or state.get("selected_topology"))
    untrained = has_change and final_training_config(goal, state) not in state.get("trained_configs", [])
    if worthwhile and minutes_left >= 60:
        step = worthwhile[0]
        # A search only pays off if an experiment is left to train its result.
        if step == "learn_topology" or left >= 2:
            return choose(step, search_reason(step))
    if untrained and "train_final" in allowed:
        return choose("train_final", f"The search changed the configuration (topology {state.get('selected_topology') or goal['topology']}); "
                                     f"train it once so the gain is measured, best so far {best:.1%}.")
    target = float(goal["target_accuracy"])
    if "continue_training" in allowed and minutes_left >= 30 and best < target:
        return choose("continue_training", f"The target {target:.0%} is not met and no search step is worth an experiment; "
                                           f"continue training the best model ({best:.1%}).")
    if "prune_model" in allowed:
        return choose("prune_model", f"No remaining step is expected to add {MIN_GAIN_POINTS:.0f} point; compress the best model ({best:.1%}) for the Jetsons.")
    rest = [LABEL[s] for s in ranked]
    if rest and not worthwhile:
        why = f"the remaining steps ({', '.join(rest)}) gained under {MIN_GAIN_POINTS:.0f} point in earlier runs at {skew}"
    elif rest:
        why = f"only {left} experiment(s) and {int(minutes_left)} minutes are left, too little to search and retrain"
    else:
        why = "no improvement step is left"
    return choose("finish_workflow", f"Stop: {why}; the best model reaches {best:.1%}.",
                  outcome="no_worthwhile_action", summary=f"Best validation accuracy {best:.4f}; {why}.")


def make_goal(run: RunEvidence, fair: bool, rng: random.Random) -> dict[str, Any]:
    if fair:
        return default_goal(dataset=run.dataset, dataset_split=run.split, seed=run.seed, target_accuracy=rng.choice([0.95, 0.97, 0.99]),
                            max_experiments=16, time_budget_minutes=720, measure_each_step=True)
    return default_goal(dataset=run.dataset, dataset_split=run.split, seed=run.seed,
                        target_accuracy=rng.choice([0.9, 0.95, 0.97, 0.99]), max_experiments=rng.choice([3, 4, 5, 6, 8]),
                        time_budget_minutes=rng.choice([120, 180, 240, 360, 480]), measure_each_step=False)
