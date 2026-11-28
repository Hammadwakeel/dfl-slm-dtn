"""Build the supervisor-SLM fine-tuning dataset from the fair-campaign evidence.

    python scripts/slm_finetune_dataset.py [--episodes 40] [--epsilon 0.3]

Every example is one decision point of a simulated supervisor run (``scripts/slm_sim.py``):
the exact system prompt, state JSON and tool list the supervisor sends to the SLM, and as
the answer one tool call chosen by the teacher policy, with a one-sentence reason.

* Two run modes: fair (every search step is measured, the SLM orders them) and limited
  budget (few experiments; the SLM chooses which steps are worth running and when to stop).
* Trajectories follow the teacher with probability 1 - epsilon and a random allowed step
  otherwise, so the dataset also covers states a weaker policy reaches.
* Split by dataset: CIFAR-10 runs are the test set and never inform the teacher's priors;
  MNIST, Fashion-MNIST and FEMNIST runs give train and validation (validation = seed 42).

Writes ``finetune/slm-supervisor/{train,val,test}.jsonl`` and ``README.md`` with counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slm_sim import Simulator, load_evidence, make_goal, priors, teacher_decision  # noqa: E402

from app.services.supervisor_policy import Decision, validate_step  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "finetune" / "slm-supervisor"
TEST_DATASETS = ("cifar10",)
VALIDATION_SEED = 42
MAX_DECISIONS = 20


def random_decision(allowed: list[str], rng: random.Random) -> Decision:
    options = [step for step in allowed if step != "finish_workflow"] or allowed
    step = rng.choice(options)
    arguments = {"outcome": "no_worthwhile_action", "summary": "Stop."} if step == "finish_workflow" else {}
    return Decision(step, validate_step(step, arguments, allowed), "", "random")


def episodes(run, prior, fair: bool, count: int, epsilon: float, rng: random.Random):
    for _ in range(count):
        sim = Simulator(run, make_goal(run, fair, rng), rng)
        for _ in range(MAX_DECISIONS):
            rule = sim.rule()
            if rule is not None:
                if not sim.apply(rule):
                    break
                continue
            messages, tools, allowed = sim.prompt()
            label = teacher_decision(sim.goal, sim.state, sim.minutes_left, allowed, prior)
            yield {
                "messages": [
                    *messages,
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"type": "function", "function": {"name": label.step, "arguments": label.arguments}}
                    ]},
                ],
                "tools": tools,
                "meta": {"dataset": run.dataset, "split": run.split, "seed": run.seed, "run_id": run.run_id,
                         "mode": "fair" if fair else "budget", "label": label.step, "allowed": allowed},
            }
            behaviour = label if rng.random() >= epsilon else random_decision(allowed, rng)
            if not sim.apply(behaviour):
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=40, help="simulated runs per evidence run and mode")
    parser.add_argument("--epsilon", type=float, default=0.3, help="share of random exploration steps")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    runs = load_evidence()
    train_runs = [r for r in runs if r.dataset not in TEST_DATASETS]
    prior = priors(train_runs)
    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    seen: set[str] = set()
    for run in runs:
        split = "test" if run.dataset in TEST_DATASETS else "val" if run.seed == VALIDATION_SEED else "train"
        for fair in (True, False):
            for example in episodes(run, prior[run.split], fair, args.episodes, args.epsilon, rng):
                key = hashlib.sha1(json.dumps([example["messages"], example["tools"]], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                splits[split].append(example)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        rng.shuffle(rows)
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "priors.json").write_text(json.dumps(prior, indent=1), encoding="utf-8")
    write_readme(splits, prior, runs, args)
    for name, rows in splits.items():
        print(name, len(rows), dict(Counter(r["meta"]["label"] for r in rows)))


def write_readme(splits: dict[str, list[dict]], prior: dict, runs, args) -> None:
    steps = sorted({step for p in prior.values() for step in p})
    lines = [
        "# Supervisor-SLM fine-tuning dataset",
        "",
        f"Built by `scripts/slm_finetune_dataset.py --episodes {args.episodes} --epsilon {args.epsilon} --seed {args.seed}`",
        f"from {len(runs)} fair-campaign runs (`docs/evidence/fair-*`). Each line is one decision of a simulated",
        "supervisor run: the exact system prompt, state JSON and tools the supervisor sends to the SLM,",
        "and one teacher tool call with a reason. Split by dataset: CIFAR-10 is the test set and never",
        "informs the teacher; validation is seed 42 of MNIST, Fashion-MNIST and FEMNIST.",
        "",
        "## Examples",
        "",
        "| Split | Examples | Fair mode | Budget mode | Labels |",
        "|---|---|---|---|---|",
    ]
    for name, rows in splits.items():
        modes = Counter(r["meta"]["mode"] for r in rows)
        labels = ", ".join(f"{k} {v}" for k, v in Counter(r["meta"]["label"] for r in rows).most_common())
        lines.append(f"| {name} | {len(rows)} | {modes['fair']} | {modes['budget']} | {labels} |")
    lines += [
        "",
        "## Teacher priors (mean measured validation effect, points, training datasets only)",
        "",
        "| Split | " + " | ".join(steps) + " |",
        "|---|" + "---|" * len(steps),
    ]
    for split, values in sorted(prior.items()):
        lines.append(f"| {split} | " + " | ".join(f"{values.get(s, 0):+.2f}" for s in steps) + " |")
    lines += [
        "",
        "A step is worth an experiment in budget mode when its prior is at least +1 point; in fair",
        "mode every search step runs and the teacher orders them by prior.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
