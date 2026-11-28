"""The fine-tuning simulator replays runs in the supervisor's own format; the teacher is sane."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import slm_sim  # noqa: E402
from slm_finetune_dataset import episodes  # noqa: E402

from app.services.supervisor_policy import validate_step  # noqa: E402

RUNS = slm_sim.load_evidence()
pytestmark = pytest.mark.skipif(not RUNS, reason="fair-campaign evidence not present")
PRIOR = slm_sim.priors([r for r in RUNS if r.dataset != "cifar10"])


def _run(dataset: str, split: str) -> slm_sim.RunEvidence:
    return next(r for r in RUNS if (r.dataset, r.split) == (dataset, split))


def test_priors_exclude_test_dataset_and_rank_topology_first():
    assert {"dirichlet_a01", "dirichlet_a05", "dirichlet_a09"} <= set(PRIOR)
    for split, values in PRIOR.items():
        assert max(values, key=values.get) == "optimize_topology", split


@pytest.mark.parametrize("fair", [True, False])
def test_first_decision_under_strong_skew_is_the_topology_sweep(fair):
    rng = random.Random(1)
    sim = slm_sim.Simulator(_run("femnist", "dirichlet_a01"), slm_sim.make_goal(_run("femnist", "dirichlet_a01"), fair, rng), rng, failure_rate=0)
    sim.goal["max_experiments"] = 6
    _, tools, allowed = sim.prompt()
    decision = slm_sim.teacher_decision(sim.goal, sim.state, sim.minutes_left, allowed, PRIOR["dirichlet_a01"])
    assert decision.step == "optimize_topology"
    assert {tool["function"]["name"] for tool in tools} == set(allowed)


def test_budget_mode_skips_low_value_steps_and_stops():
    run = _run("fashion_mnist", "dirichlet_a05")
    rng = random.Random(3)
    sim = slm_sim.Simulator(run, slm_sim.make_goal(run, False, rng), rng, failure_rate=0)
    sim.goal.update(max_experiments=8, time_budget_minutes=480, target_accuracy=0.99)
    sim.minutes_left = 450
    chosen = []
    for _ in range(20):
        rule = sim.rule()
        decision = rule or slm_sim.teacher_decision(sim.goal, sim.state, sim.minutes_left, sim.prompt()[2], PRIOR[run.split])
        chosen.append(decision.step)
        if not sim.apply(decision):
            break
    assert chosen[-1] == "finish_workflow"
    assert not {"run_hpo", "optimize_sync", "learn_topology"} & set(chosen)
    assert chosen.index("optimize_topology") < chosen.index("train_final")


def test_every_example_is_a_valid_tool_call_in_the_allowed_list():
    rng = random.Random(7)
    run = _run("mnist", "dirichlet_a09")
    for fair in (True, False):
        for example in episodes(run, PRIOR[run.split], fair, 5, 0.3, rng):
            call = example["messages"][-1]["tool_calls"][0]["function"]
            assert call["name"] in example["meta"]["allowed"]
            validate_step(call["name"], call["arguments"], example["meta"]["allowed"])
            assert call["arguments"]["reason"]
            assert example["messages"][0]["role"] == "system" and "State: " in example["messages"][0]["content"]
