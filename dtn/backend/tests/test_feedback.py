import pytest

from app.services.feedback import FeedbackSummaryError, build_training_feedback_summary


def _completed_group():
    return {
        "kind": "decentralized",
        "status": "completed",
        "dataset_name": "cifar_10k_demo",
        "topology": "bidirectional_ring",
        "rounds": 2,
        "epochs": 1,
        "nas_enabled": False,
        "started_at": 10.0,
        "completed_at": 32.0,
        "runs": [
            {
                "status": "completed",
                "rounds": [
                    {
                        "workers": {
                            "worker1": {"bytes_sent": 100},
                            "worker2": {"bytes_sent": 110},
                        }
                    },
                    {
                        "workers": {
                            "worker1": {
                                "bytes_sent": 120,
                                "validation_accuracy": 0.4,
                                "validation_macro_f1": 0.35,
                            },
                            "worker2": {
                                "bytes_sent": 130,
                                "validation_accuracy": 0.5,
                                "validation_macro_f1": 0.45,
                            },
                        }
                    },
                ],
            }
        ],
    }


def test_training_feedback_summary_is_bounded_and_uses_final_validation_metrics():
    summary = build_training_feedback_summary(
        _completed_group(),
        iteration=2,
        previous_accuracy=0.3,
    )

    assert len(summary) == 15
    assert summary["dataset_name"] == "cifar_10k_demo"
    assert summary["accuracy"] == pytest.approx(0.45)
    assert summary["macro_f1"] == pytest.approx(0.4)
    assert summary["communication_bytes"] == 460
    assert summary["duration_seconds"] == 22
    assert summary["worker_accuracy_spread"] == pytest.approx(0.1)
    assert summary["accuracy_delta"] == pytest.approx(0.15)


def test_training_feedback_rejects_incomplete_or_unmeasured_groups():
    failed = _completed_group()
    failed["status"] = "failed"
    with pytest.raises(FeedbackSummaryError, match="completed"):
        build_training_feedback_summary(failed, iteration=1)

    missing_metrics = _completed_group()
    del missing_metrics["runs"][0]["rounds"][-1]["workers"]["worker1"][
        "validation_macro_f1"
    ]
    with pytest.raises(FeedbackSummaryError, match="missing or invalid"):
        build_training_feedback_summary(missing_metrics, iteration=1)
