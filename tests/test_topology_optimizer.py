from pathlib import Path

import pytest

from topology_opt.reward import RewardConfig, calculate_run_reward, worker_quality
from topology_opt.selector import select_candidate


V1_CONFIG = Path(__file__).resolve().parents[1] / "topology_opt" / "reward_config_v1.json"


def test_reward_combines_accuracy_convergence_communication_and_peer_wait():
    config = RewardConfig(
        weights={
            "accuracy_gain": 0.5,
            "convergence_speed": 0.2,
            "communication_efficiency": 0.15,
            "peer_wait_efficiency": 0.15,
        },
        target_accuracy=0.5,
        accuracy_gain_scale=0.25,
        max_bytes_per_worker_round=1000,
        max_exchange_seconds_per_worker_round=1,
        max_peer_wait_seconds_per_worker_round=10,
    )
    rounds = [
        {
            "workers": {
                f"worker{worker}": {
                    "accuracy": accuracy,
                    "bytes_sent": 100,
                    "exchange_seconds": 0.1,
                    "round_seconds": 2,
                    "peer_wait_seconds": 0.25,
                }
                for worker in range(1, 4)
            }
        }
        for accuracy in (0.3, 0.5, 0.7)
    ]

    result = calculate_run_reward(rounds, config)

    assert result["components"]["accuracy_gain"] == 1
    assert result["components"]["convergence_speed"] == round(2 / 3, 6)
    assert result["metrics"]["convergence_round"] == 2
    assert result["metrics"]["bytes_sent"] == 900
    assert result["metrics"]["runtime_seconds"] == 6
    assert result["metrics"]["peer_wait_seconds"] == 2.25
    assert result["components"]["peer_wait_efficiency"] == 0.975
    assert 0 <= result["reward"] <= 1


def test_reward_prefers_validation_accuracy_when_test_metrics_are_present():
    config = RewardConfig(
        weights={
            "accuracy_gain": 1,
            "convergence_speed": 0,
            "communication_efficiency": 0,
            "peer_wait_efficiency": 0,
        },
        target_accuracy=0.9,
        accuracy_gain_scale=1,
        max_bytes_per_worker_round=1000,
        max_exchange_seconds_per_worker_round=1,
        max_peer_wait_seconds_per_worker_round=10,
    )
    rounds = [
        {"workers": {"worker1": {"accuracy": 0.1, "validation_accuracy": 0.2}}},
        {"workers": {"worker1": {"accuracy": 0.9, "validation_accuracy": 0.3}}},
    ]

    result = calculate_run_reward(rounds, config)

    assert result["metrics"]["initial_accuracy"] == 0.2
    assert result["metrics"]["final_accuracy"] == 0.3
    assert result["metrics"]["accuracy_gain"] == 0.1


def test_single_round_reward_uses_pretraining_validation_baseline():
    config = RewardConfig(
        weights={
            "accuracy_gain": 1,
            "convergence_speed": 0,
            "communication_efficiency": 0,
            "peer_wait_efficiency": 0,
        },
        target_accuracy=0.9,
        accuracy_gain_scale=0.25,
        max_bytes_per_worker_round=1000,
        max_exchange_seconds_per_worker_round=1,
        max_peer_wait_seconds_per_worker_round=10,
    )
    rounds = [{
        "workers": {
            "worker1": {
                "initial_validation_accuracy": 0.1,
                "validation_accuracy": 0.35,
            },
        },
    }]

    result = calculate_run_reward(rounds, config)

    assert result["components"]["accuracy_gain"] == 1
    assert result["metrics"]["initial_accuracy"] == 0.1
    assert result["metrics"]["accuracy_gain"] == 0.25
    assert result["metrics"]["accuracy_gain_available"] is True


def test_single_round_reward_marks_missing_accuracy_gain_baseline_unavailable():
    config = RewardConfig(
        weights={
            "accuracy_gain": 1,
            "convergence_speed": 0,
            "communication_efficiency": 0,
            "peer_wait_efficiency": 0,
        },
        target_accuracy=0.9,
        accuracy_gain_scale=0.25,
        max_bytes_per_worker_round=1000,
        max_exchange_seconds_per_worker_round=1,
        max_peer_wait_seconds_per_worker_round=10,
    )
    rounds = [{"workers": {"worker1": {"validation_accuracy": 0.35}}}]

    result = calculate_run_reward(rounds, config)

    assert result["components"]["accuracy_gain"] == 0
    assert result["metrics"]["accuracy_gain_available"] is False


def test_peer_wait_efficiency_rewards_less_worker_waiting():
    config = RewardConfig.load(V1_CONFIG)
    rounds = [{
        "workers": {
            f"worker{worker}": {
                "validation_accuracy": 0.53,
                "bytes_sent": 6_000_000,
                "exchange_seconds": 10,
                "round_seconds": 45,
                "peer_wait_seconds": 5,
            }
            for worker in range(1, 4)
        },
    }]

    result = calculate_run_reward(rounds, config)

    assert result["components"]["communication_efficiency"] > 0
    assert result["components"]["peer_wait_efficiency"] > 0


def test_selector_reports_reward_choice_and_seeded_baselines():
    candidates = [
        {
            "topology": "chain",
            "alpha": 0.3,
            "completed_runs": 3,
            "mean_reward": 0.8,
            "mean_final_accuracy": 0.7,
        },
        {
            "topology": "star",
            "alpha": 0.5,
            "completed_runs": 3,
            "mean_reward": 0.7,
            "mean_final_accuracy": 0.8,
        },
    ]

    selection = select_candidate(candidates, seed=7)

    assert selection["selected_candidate"]["topology"] == "chain"
    assert selection["baselines"]["best_observed"]["topology"] == "star"
    assert selection["baselines"]["random_seeded"] == select_candidate(candidates, seed=7)[
        "baselines"
    ]["random_seeded"]



def _v2_round(accuracies, *, bytes_sent=1_000, wait=1.0, round_seconds=10.0, weight_bytes=1_000):
    return {
        "workers": {
            f"worker{index}": {
                "global_validation_accuracy": accuracy,
                "global_validation_macro_f1": accuracy,
                "validation_accuracy": 0.99,
                "bytes_sent": bytes_sent,
                "weight_bytes": weight_bytes,
                "peer_wait_seconds": wait,
                "round_seconds": round_seconds,
            }
            for index, accuracy in enumerate(accuracies, start=1)
        }
    }


def test_default_config_is_reward_v2():
    config = RewardConfig.load()
    assert config.version == 2
    assert set(config.weights) == {"quality", "learning_speed", "bandwidth_efficiency", "wait_efficiency"}
    assert RewardConfig.from_dict(config.as_dict()) == config


def test_v2_quality_counts_the_weakest_worker_and_ignores_local_accuracy():
    config = RewardConfig.load()
    even = calculate_run_reward([_v2_round([0.8, 0.8, 0.8])], config)
    one_weak = calculate_run_reward([_v2_round([0.95, 0.95, 0.5])], config)

    assert even["metrics"]["quality_source"] == "global_validation"
    assert even["components"]["quality"] == pytest.approx(0.8)
    # mean 0.8 but worst 0.5: q = 0.5 * 0.8 + 0.5 * 0.5
    assert one_weak["components"]["quality"] == pytest.approx(0.65)
    assert one_weak["metrics"]["min_worker_accuracy"] == pytest.approx(0.5)
    assert one_weak["metrics"]["passes_quality_gate"] is False


def test_v2_learning_speed_is_the_area_under_the_quality_curve():
    config = RewardConfig.load()
    fast = calculate_run_reward([_v2_round([0.9] * 3), _v2_round([0.9] * 3)], config)
    slow = calculate_run_reward([_v2_round([0.3] * 3), _v2_round([0.9] * 3)], config)

    assert fast["components"]["quality"] == slow["components"]["quality"]
    assert fast["components"]["learning_speed"] == pytest.approx(0.9)
    assert slow["components"]["learning_speed"] == pytest.approx(0.6)


def test_v2_bandwidth_is_relative_to_a_full_mesh_and_wait_to_round_time():
    config = RewardConfig.load()
    # 4 workers, 1 round: a full mesh sends model_bytes * 4 * 3 in total.
    mesh = calculate_run_reward([_v2_round([0.9] * 4, bytes_sent=3_000, wait=5.0)], config)
    ring = calculate_run_reward([_v2_round([0.9] * 4, bytes_sent=1_000, wait=2.0)], config)

    assert mesh["components"]["bandwidth_efficiency"] == pytest.approx(0.0)
    assert ring["components"]["bandwidth_efficiency"] == pytest.approx(1 - 4_000 / 12_000)
    assert mesh["components"]["wait_efficiency"] == pytest.approx(0.5)
    assert ring["components"]["wait_efficiency"] == pytest.approx(0.8)


def test_local_f1_ignores_classes_absent_from_the_worker_split():
    metrics = {
        "validation_accuracy": 0.97,
        "validation_macro_f1": 0.5,
        "validation_per_class_f1": [1.0, 0.0, 0.9],
        "validation_confusion_matrix_counts": [[10, 0, 0], [0, 0, 0], [0, 1, 9]],
    }
    f1, accuracy, source = worker_quality(metrics)
    assert source == "local_present_classes"
    assert f1 == pytest.approx(0.95)
    assert accuracy == pytest.approx(0.97)


def test_v2_no_longer_rewards_a_cheap_topology_with_a_failing_worker():
    """Regression from the 2026-11-17 MNIST sweep: chain beat mesh under reward v1."""
    config = RewardConfig.load()
    mesh = calculate_run_reward(
        [_v2_round([0.97] * 4, bytes_sent=3_000, wait=3.0)] * 4, config
    )
    chain = calculate_run_reward(
        [_v2_round([0.95, 0.95, 0.9, 0.5], bytes_sent=500, wait=2.0)] * 4, config
    )
    assert mesh["reward"] > chain["reward"]


def _v2_candidate(topology, reward, *, gate, bandwidth, std=0.0, min_worker=None):
    extra = {} if min_worker is None else {"mean_min_worker_accuracy": min_worker}
    return {
        **extra,
        "topology": topology,
        "alpha": 0.3,
        "completed_runs": 3,
        "reward_version": 2,
        "mean_reward": reward,
        "std_reward": std,
        "mean_final_accuracy": reward,
        "passes_quality_gate": gate,
        "mean_components": {"bandwidth_efficiency": bandwidth},
    }


def test_v2_selector_prefers_candidates_that_pass_the_quality_gate():
    selection = select_candidate(
        [
            _v2_candidate("chain", 0.9, gate=False, bandwidth=0.8),
            _v2_candidate("mesh", 0.7, gate=True, bandwidth=0.0),
        ]
    )
    assert selection["method"] == "quality_gated_reward_v2"
    assert selection["quality_gate_met"] is True
    assert selection["selected_candidate"]["topology"] == "mesh"


def test_v2_selector_breaks_statistical_ties_toward_less_bandwidth():
    selection = select_candidate(
        [
            _v2_candidate("mesh", 0.80, gate=True, bandwidth=0.0, std=0.03),
            _v2_candidate("ring", 0.78, gate=True, bandwidth=0.8),
            _v2_candidate("star", 0.70, gate=True, bandwidth=0.9),
        ]
    )
    assert selection["selected_candidate"]["topology"] == "ring"


def test_v2_selector_falls_back_to_best_reward_when_no_candidate_passes():
    selection = select_candidate(
        [
            _v2_candidate("chain", 0.6, gate=False, bandwidth=0.8),
            _v2_candidate("mesh", 0.7, gate=False, bandwidth=0.0),
        ]
    )
    assert selection["quality_gate_met"] is False
    assert selection["selected_candidate"]["topology"] == "mesh"


def test_v2_selector_keeps_quality_first_when_no_candidate_passes_the_gate():
    # Measured on CIFAR-10 (4 rounds): no topology reaches the 0.9 gate and chain's
    # bandwidth savings outscored mesh although mesh's weakest worker was far better.
    selection = select_candidate(
        [
            _v2_candidate("chain", 0.4706, gate=False, bandwidth=0.76, min_worker=0.3885),
            _v2_candidate("mesh", 0.4656, gate=False, bandwidth=0.06, min_worker=0.5495),
            _v2_candidate("bidirectional_ring", 0.4648, gate=False, bandwidth=0.73, min_worker=0.394),
        ]
    )
    assert selection["quality_gate_met"] is False
    assert selection["method"] == "relative_quality_gated_reward_v2"
    assert selection["selected_candidate"]["topology"] == "mesh"
