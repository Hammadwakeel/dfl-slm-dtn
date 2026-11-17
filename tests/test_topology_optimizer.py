from topology_opt.reward import RewardConfig, calculate_run_reward
from topology_opt.selector import select_candidate


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
    config = RewardConfig.load()
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
