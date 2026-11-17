import json

import pytest

from scripts.finetune_slm_reply_style import natural_language_target, read_split


def make_row(module: str = "Topology Optimization") -> dict:
    answer = {
        "recommended_modules": [{"module": module, "priority": 1}],
        "reason": "The topology reward is low.",
        "next_action": "Apply the topology optimization module and reassess convergence.",
    }
    return {
        "messages": [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "DFL state: topology reward=0.1"},
            {"role": "assistant", "content": json.dumps(answer)},
        ]
    }


def test_natural_language_target_does_not_train_structured_actions() -> None:
    result = natural_language_target(make_row())

    assert result["user"] == "DFL state: topology reward=0.1"
    assert result["target"] == (
        "I recommend Topology Optimization. The topology reward is low. "
        "Next, Apply the topology optimization module and reassess convergence."
    )
    assert not result["target"].startswith("{")


def test_insufficient_evidence_is_rendered_as_natural_language() -> None:
    result = natural_language_target(make_row("Insufficient Evidence"))

    assert result["target"].startswith("There is not enough information")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(messages=[]),
        lambda row: row["messages"][2].update(content="{bad json"),
        lambda row: row["messages"][2].update(
            content=json.dumps(
                {
                    "recommended_modules": [{"module": "unrecognized"}],
                    "reason": "Reason",
                    "next_action": "Next",
                }
            )
        ),
    ],
)
def test_invalid_training_label_is_rejected(mutate) -> None:
    row = make_row()
    mutate(row)

    with pytest.raises(ValueError):
        natural_language_target(row)


def test_read_split_identifies_invalid_source_line(tmp_path) -> None:
    path = tmp_path / "dfl_train.jsonl"
    path.write_text(json.dumps(make_row()) + "\n{bad json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dfl_train.jsonl:2"):
        read_split(path)
