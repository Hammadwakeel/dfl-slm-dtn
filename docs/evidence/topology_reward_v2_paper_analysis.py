"""Cross-dataset analysis of the reward-v2 topology sweeps (MNIST alphas and CIFAR-10).

Usage:
    python topology_reward_v2_paper_analysis.py ENV_FILE REPO OUT_JSON LABEL=GROUP_ID [LABEL=GROUP_ID ...]

Reads the groups from the DTN backend (the token is read from ENV_FILE at runtime and
never printed), recomputes reward v1 and v2 for every run, and prints the tables used in
the paper report. OUT_JSON gets every per-run number.
"""
import itertools
import json
import statistics as st
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
envp, repo, out = sys.argv[1:4]
pairs = [arg.rsplit("=", 1) for arg in sys.argv[4:]]
sys.path.insert(0, repo)
from topology_opt.reward import RewardConfig, calculate_run_reward  # noqa: E402

env = {}
for line in open(envp, encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
request = urllib.request.Request(
    "http://127.0.0.1:8000/api/slm/results",
    headers={"Authorization": "Bearer " + env["DTN_API_TOKEN"]},
)
groups = {g["group_id"]: g for g in json.load(urllib.request.urlopen(request, timeout=120))["groups"]}
V1 = RewardConfig.load(repo + "/topology_opt/reward_config_v1.json")
TOPOLOGIES = ["directed_ring", "bidirectional_ring", "mesh", "chain", "star"]
GATES = [round(0.40 + 0.05 * i, 2) for i in range(12)]  # 0.40 ... 0.95


def mean_std(values):
    return (st.mean(values), st.pstdev(values)) if values else (None, None)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: -values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            result[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return result


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def gated_select(summary, per_run, threshold):
    """Selector v2 at an arbitrary weakest-worker threshold."""
    passing = [
        t for t in summary
        if all(run["global_min"] >= threshold for run in per_run[t])
    ]
    pool = passing or list(summary)
    best = max(pool, key=lambda t: summary[t]["v2"][0])
    tied = [
        t for t in pool
        if summary[best]["v2"][0] - summary[t]["v2"][0] <= summary[best]["v2"][1]
    ]
    chosen = max(tied, key=lambda t: summary[t]["B"])
    return chosen, bool(passing)


report = {}
for label, gid in pairs:
    group = groups[gid]
    v2 = RewardConfig.from_dict(group["reward_config"])
    per_run = {}
    for topology in TOPOLOGIES:
        runs = [r for r in group["runs"] if r["topology"] == topology and r["status"] == "completed"]
        rows = []
        for run in runs:
            a = calculate_run_reward(run["rounds"], V1)
            b = calculate_run_reward(run["rounds"], v2)
            first = list(run["rounds"][0]["workers"].values())
            last = list(run["rounds"][-1]["workers"].values())
            rows.append({
                "seed": run.get("seed"),
                "v1": a["reward"],
                "v2": b["reward"],
                "c": b["components"],
                "local_acc": st.mean(m["validation_accuracy"] for m in last),
                "global_acc": st.mean(m["global_validation_accuracy"] for m in last),
                "global_min": min(m["global_validation_accuracy"] for m in last),
                "global_f1": st.mean(m["global_validation_macro_f1"] for m in last),
                "initial_acc": st.mean(m.get("initial_validation_accuracy", float("nan")) for m in first),
                "weights_sha256": sorted({str(m.get("weights_sha256")) for m in first}),
                "curve": [
                    st.mean(m["global_validation_accuracy"] for m in rd["workers"].values())
                    for rd in run["rounds"]
                ],
                "min_curve": [
                    min(m["global_validation_accuracy"] for m in rd["workers"].values())
                    for rd in run["rounds"]
                ],
                "bytes": b["metrics"]["bytes_sent"],
                "wall": sum(
                    max(m["round_seconds"] for m in rd["workers"].values())
                    for rd in run["rounds"]
                ),
            })
        if rows:
            per_run[topology] = rows
    summary = {}
    for topology, rows in per_run.items():
        f = lambda key: mean_std([x[key] for x in rows])  # noqa: E731
        c = lambda key: st.mean(x["c"][key] for x in rows)  # noqa: E731
        summary[topology] = {
            "n": len(rows),
            "v1": f("v1"),
            "v2": f("v2"),
            "local_acc": f("local_acc"),
            "global_acc": f("global_acc"),
            "global_min": f("global_min"),
            "global_f1": f("global_f1"),
            "initial_acc": f("initial_acc"),
            "Q": c("quality"),
            "S": c("learning_speed"),
            "B": c("bandwidth_efficiency"),
            "W": c("wait_efficiency"),
            "mb": st.mean(x["bytes"] for x in rows) / 1e6,
            "wall": st.mean(x["wall"] for x in rows),
            "curve": [st.mean(col) for col in zip(*[x["curve"] for x in rows])],
            "min_curve": [st.mean(col) for col in zip(*[x["min_curve"] for x in rows])],
        }
    names = list(summary)
    acc = [summary[t]["global_acc"][0] for t in names]
    weakest = [summary[t]["global_min"][0] for t in names]
    rho = {
        "v1_vs_shared_acc": spearman([summary[t]["v1"][0] for t in names], acc),
        "v2_vs_shared_acc": spearman([summary[t]["v2"][0] for t in names], acc),
        "v1_vs_weakest": spearman([summary[t]["v1"][0] for t in names], weakest),
        "v2_vs_weakest": spearman([summary[t]["v2"][0] for t in names], weakest),
    }
    # Weight sensitivity of the ungated reward (0.05 grid, quality weight >= 0.2).
    grid, wins, total = [x / 20 for x in range(21)], {}, 0
    for wq, ws, wb in itertools.product(grid, repeat=3):
        ww = round(1 - wq - ws - wb, 10)
        if ww < -1e-9 or wq < 0.2:
            continue
        total += 1
        score = {
            t: st.mean(
                wq * x["c"]["quality"] + ws * x["c"]["learning_speed"]
                + wb * x["c"]["bandwidth_efficiency"] + ww * x["c"]["wait_efficiency"]
                for x in per_run[t]
            )
            for t in names
        }
        winner = max(score, key=score.get)
        wins[winner] = wins.get(winner, 0) + 1
    gate_curve = {str(thr): gated_select(summary, per_run, thr) for thr in GATES}
    report[label] = {
        "group_id": gid,
        "dataset": group.get("dataset_name"),
        "alphas": group.get("alphas"),
        "rounds": group.get("rounds"),
        "epochs": group.get("epochs"),
        "seeds": group.get("seeds"),
        "architecture": (group.get("architecture") or {}).get("id")
        if isinstance(group.get("architecture"), dict) else group.get("architecture"),
        "status": group.get("status"),
        "selection": group.get("selection"),
        "summary": summary,
        "spearman": rho,
        "weight_sensitivity": {"total": total, "wins": {t: n / total for t, n in wins.items()}},
        "gate_threshold_selection": gate_curve,
        "runs": per_run,
    }

    sel = group.get("selection") or {}
    print(f"\n=== {label}: {gid} | {group.get('dataset_name')} alpha={group.get('alphas')} "
          f"R={group.get('rounds')} E={group.get('epochs')} | status {group.get('status')} | "
          f"selected {(sel.get('selected_candidate') or {}).get('topology')} gate_met={sel.get('quality_gate_met')}")
    print(f"{'topology':<19}{'n':>2} {'v1':>13} {'v2':>13} {'init':>6} {'local':>6} {'shared acc':>13} "
          f"{'weakest':>13} {'Q':>6}{'S':>6}{'B':>6}{'W':>6} {'MB':>6} {'wall':>6}")
    pm = lambda v: f"{v[0]:.3f}±{v[1]:.3f}"  # noqa: E731
    pc = lambda v: f"{100*v[0]:.1f}±{100*v[1]:.1f}"  # noqa: E731
    for t, s in summary.items():
        print(f"{t:<19}{s['n']:>2} {pm(s['v1']):>13} {pm(s['v2']):>13} {100*s['initial_acc'][0]:>5.1f}% "
              f"{100*s['local_acc'][0]:>5.1f}% {pc(s['global_acc']):>13} {pc(s['global_min']):>13} "
              f"{s['Q']:>6.3f}{s['S']:>6.3f}{s['B']:>6.3f}{s['W']:>6.3f} {s['mb']:>6.2f} {s['wall']:>6.0f}")
    print("v1 ranking:", " > ".join(sorted(names, key=lambda t: -summary[t]["v1"][0])))
    print("v2 ranking:", " > ".join(sorted(names, key=lambda t: -summary[t]["v2"][0])))
    print("shared-acc ranking:", " > ".join(sorted(names, key=lambda t: -summary[t]["global_acc"][0])))
    print("spearman:", {k: (round(v, 2) if v is not None else None) for k, v in rho.items()})
    print("ungated weight sensitivity:", {t: f"{100*n/total:.1f}%" for t, n in sorted(wins.items(), key=lambda kv: -kv[1])})
    print("gated selection by threshold:", {thr: f"{t}{'' if ok else ' (no pass)'}" for thr, (t, ok) in gate_curve.items()})

json.dump(report, open(out, "w", encoding="utf-8"), indent=1, default=str)
print("\nwrote", out)
