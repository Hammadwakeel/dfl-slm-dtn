"""Base vs fine-tuned supervisor SLM on the real cluster, one training budget per dataset.

Every supervisor run is a fair run (``--measure-each-step``): the baseline and every
model the run compares with it are trained with the dataset's rounds x local epochs,
from the same initial weights. Harder datasets get a larger budget so the models
converge; the budget never changes inside a dataset.

With ``--max-experiments 7`` a run fits the baseline and three measured search steps
out of the six it could try, so the SLM's choice of steps decides the final accuracy.
That choice is the only difference between the two arms: both arms see the same
datasets, splits, seeds, budgets and supervisor code. The model on the master is
switched before each run, and the arm that runs first alternates by seed.

    python scripts/slm_compare_campaign.py --splits dirichlet_a05
    python scripts/slm_compare_campaign.py --splits dirichlet_a05 dirichlet_a01 --datasets cifar10

Progress lives in each arm's ``docs/evidence/slmcmp-*.json``; re-running resumes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import paper_campaign  # noqa: E402

MASTER = "hammad@192.168.11.15"
ENV_FILE = "/etc/dfl-slm-master.env"
SERVICE = "dfl-slm-master"
MODELS = {"base": "qwen3.5:4b", "tuned": "qwen3.5-dfl:4b"}
# rounds x local epochs per dataset (the same for the baseline and every compared model)
BUDGETS = {
    "mnist": {"rounds": 5, "local_epochs": 3, "time_budget": 360},
    "fashion_mnist": {"rounds": 10, "local_epochs": 5, "time_budget": 480},
    "femnist": {"rounds": 15, "local_epochs": 5, "time_budget": 720},
    "cifar10": {"rounds": 20, "local_epochs": 5, "time_budget": 1080},
}
SEEDS = (17, 42, 2026)
MAX_EXPERIMENTS = 7
TARGET = 0.995  # out of reach, so no run stops early on the target


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def ssh(command: str, timeout: int = 120) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", MASTER, command],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ssh failed ({result.returncode}): {result.stderr.strip()[-400:]}")
    return result.stdout


def current_model() -> str:
    line = ssh(f"sudo -n grep -E '^SLM_MODEL_NAME=' {ENV_FILE}").strip()
    return line.split("=", 1)[1] if "=" in line else ""


def switch_model(model: str) -> None:
    """Point the master's SLM service at ``model`` and wait until it serves it."""
    if model not in ssh("ollama list"):
        raise RuntimeError(f"{model} is not installed in Ollama on the master")
    if current_model() == model:
        return
    log(f"switching the master SLM to {model}")
    since = ssh("date '+%Y-%m-%d %H:%M:%S'").strip()
    ssh(f"sudo -n sed -i 's|^SLM_MODEL_NAME=.*|SLM_MODEL_NAME={model}|' {ENV_FILE} && sudo -n systemctl restart {SERVICE}")
    for _ in range(30):
        time.sleep(5)
        journal = ssh(f"journalctl -u {SERVICE} --since '{since}' --no-pager -o cat | grep 'Starting SLM master API' | tail -1")
        if f"using {model}" in journal:
            # unload the previous model so only one 4B model holds Jetson memory
            ssh("ollama stop qwen3.5:4b >/dev/null 2>&1; ollama stop qwen3.5-dfl:4b >/dev/null 2>&1; true")
            return
    raise RuntimeError(f"the master did not restart with {model}")


def report_path(dataset: str, split: str, arm: str) -> Path:
    return REPO_ROOT / "docs" / "evidence" / f"slmcmp-{dataset}-{split}-{arm}.md"


def run_arm(dataset: str, split: str, seed: int, arm: str) -> int:
    budget = BUDGETS[dataset]
    switch_model(MODELS[arm])
    log(f"{dataset} {split} seed {seed}: {arm} SLM ({MODELS[arm]}), {budget['rounds']} rounds x {budget['local_epochs']} epochs")
    return paper_campaign.main([
        "--dataset", dataset, "--split", split, "--measure-each-step",
        "--rounds", str(budget["rounds"]), "--local-epochs", str(budget["local_epochs"]),
        "--target", str(TARGET), "--seeds", str(seed),
        "--max-experiments", str(MAX_EXPERIMENTS), "--time-budget", str(budget["time_budget"]),
        "--out", str(report_path(dataset, split, arm)),
    ])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["dirichlet_a05"])
    parser.add_argument("--datasets", nargs="+", default=list(BUDGETS), choices=list(BUDGETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args(argv)
    for split in args.splits:
        for dataset in args.datasets:
            for index, seed in enumerate(args.seeds):
                arms = list(args.arms) if index % 2 == 0 else list(reversed(args.arms))
                for arm in arms:
                    if run_arm(dataset, split, seed, arm) != 0:
                        log(f"{dataset} {split} seed {seed} {arm} failed; stopping so it can be fixed")
                        return 1
    switch_model(MODELS["base"])
    log("campaign complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
