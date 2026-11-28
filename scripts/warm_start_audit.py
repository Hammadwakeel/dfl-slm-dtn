"""Which supervisor runs started their baseline from a previous run's trained weights?

A run installs its best model (trained weights) as the dataset's initial model when
it ends. The next run of the same dataset calls create_initial_model(restore_baseline),
which reuses an installed model of the same design unchanged, trained weights included.
So a run's baseline was warm-started when the last model installed for that dataset
before it had the baseline design (baseline layers and default training settings).
"""
import json
import re
import sqlite3
import sys

# Runs created after the fix (initial models record how their weights were made and a
# trained model is never reused) cannot be warm-started; created_at is UTC.
FIXED_AFTER = "2026-11-25 11:05"
c = sqlite3.connect("file:dtn/backend/data/dtn.sqlite3?mode=ro", uri=True)
rows = c.execute("select id, task, status, created_at, result, goal, state from agent_runs where task like 'Reach %' order by created_at").fetchall()


def design(arch):
    if not isinstance(arch, dict):
        return None
    layers = json.dumps(arch.get("layers"), sort_keys=True)
    return layers + json.dumps(arch.get("training") or {}, sort_keys=True)


last_installed = {}  # dataset -> (design, run id, step)
for rid, task, status, created, result, goal_json, state_json in rows:
    res = json.loads(result) if result else {}
    state = json.loads(state_json) if state_json else {}
    goal = json.loads(goal_json) if goal_json else {}
    dataset = goal.get("dataset") or task.rsplit(" ", 1)[-1]
    baseline = state.get("baseline_architecture")
    events = c.execute("select event_type, message from agent_run_events where run_id=? order by id", (rid,)).fetchall()
    started_baseline = any(e == "step_completed" and m.startswith("baseline_training completed") for e, m in events)
    prev = last_installed.get(dataset)
    warm = bool(started_baseline and prev and baseline and prev[0] == design(baseline)) and created[:16] < FIXED_AFTER
    base_msg = next((m for e, m in events if e == "step_completed" and m.startswith("baseline_training completed")), "")
    acc = re.search(r"accuracy ([0-9.]+)", base_msg)
    print(f"{created[:16]} {rid[:8]} {dataset:13s} {goal.get('dataset_split','?'):14s} seed {str(goal.get('seed')):5s} {status:9s} "
          f"baseline {acc.group(1) if acc else '-':8s} {'WARM from ' + prev[1][:8] + ' (' + prev[2] + ')' if warm else ''}")
    deployed = next((m for e, m in events if e == "model_deployed"), None)
    if deployed:
        ckpt = state.get("best_checkpoint") or {}
        arch = ckpt.get("architecture")
        if not isinstance(arch, dict):
            name = (res.get("deployed_model") or {}).get("architecture_id")
            arch = baseline if name == "baseline" else None
        step = re.search(r"\(([^,]+),", deployed)
        last_installed[dataset] = (design(arch) if isinstance(arch, dict) else None, rid, step.group(1) if step else "?")
