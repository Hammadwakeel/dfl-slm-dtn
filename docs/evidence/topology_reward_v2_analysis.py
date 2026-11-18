"""Reward v1 vs v2 analysis of the 3-seed topology sweep (writes analysis.json)."""
import itertools, json, statistics as st, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
envp, repo, gid, out = sys.argv[1:5]
sys.path.insert(0, repo)
from topology_opt.reward import RewardConfig, calculate_run_reward
env = {}
for line in open(envp, encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); env[k.strip()] = v.strip().strip('"').strip("'")
req = urllib.request.Request("http://127.0.0.1:8000/api/slm/results", headers={"Authorization": "Bearer " + env["DTN_API_TOKEN"]})
g = next(x for x in json.load(urllib.request.urlopen(req, timeout=60))["groups"] if x["group_id"] == gid)
v1 = RewardConfig.load(repo + "/topology_opt/reward_config_v1.json"); v2 = RewardConfig.from_dict(g["reward_config"])
TOP = ["directed_ring", "bidirectional_ring", "mesh", "chain", "star"]
def ms(xs): return (st.mean(xs), st.pstdev(xs)) if xs else (None, None)
rows = {}
for t in TOP:
    runs = [r for r in g["runs"] if r["topology"] == t and r["status"] == "completed"]
    per = []
    for r in runs:
        a, b = calculate_run_reward(r["rounds"], v1), calculate_run_reward(r["rounds"], v2)
        last = r["rounds"][-1]["workers"].values()
        per.append({
            "seed": r.get("seed"), "v1": a["reward"], "v2": b["reward"], "c": b["components"], "m": b["metrics"],
            "local_acc": st.mean(m["validation_accuracy"] for m in last),
            "global_acc": st.mean(m["global_validation_accuracy"] for m in last),
            "global_min": min(m["global_validation_accuracy"] for m in last),
            "curve": [st.mean(m["global_validation_accuracy"] for m in rd["workers"].values()) for rd in r["rounds"]],
            "bytes": b["metrics"]["bytes_sent"], "wall": sum(max(m["round_seconds"] for m in rd["workers"].values()) for rd in r["rounds"]),
        })
    rows[t] = per
print(f"seeds: {g['seeds']} | status: {g['status']} | selection: {json.dumps(g.get('selection', {}).get('selected_candidate'))} gate_met={g.get('selection', {}).get('quality_gate_met')}\n")
hdr = f"{'topology':<19}{'n':>2} {'v1 reward':>15} {'v2 reward':>15} {'gate':>5} {'local acc':>10} {'shared acc':>15} {'weakest':>15} {'Q':>6}{'S':>6}{'B':>6}{'W':>6} {'MB sent':>8} {'wall s':>7}"
print(hdr)
summary = {}
for t in TOP:
    p = rows[t]
    if not p: continue
    f = lambda k: ms([x[k] for x in p]); c = lambda k: st.mean(x["c"][k] for x in p)
    s = {"n": len(p), "v1": f("v1"), "v2": f("v2"), "gate": sum(x["m"]["passes_quality_gate"] for x in p),
         "local_acc": f("local_acc"), "global_acc": f("global_acc"), "global_min": f("global_min"),
         "Q": c("quality"), "S": c("learning_speed"), "B": c("bandwidth_efficiency"), "W": c("wait_efficiency"),
         "mb": st.mean(x["bytes"] for x in p) / 1e6, "wall": st.mean(x["wall"] for x in p),
         "curve": [st.mean(col) for col in zip(*[x["curve"] for x in p])]}
    summary[t] = s
    pm = lambda v: f"{v[0]:.4f}±{v[1]:.4f}"; pc = lambda v: f"{100*v[0]:.1f}±{100*v[1]:.1f}%"
    print(f"{t:<19}{s['n']:>2} {pm(s['v1']):>15} {pm(s['v2']):>15} {s['gate']:>3}/{s['n']} {100*s['local_acc'][0]:>9.1f}% {pc(s['global_acc']):>15} {pc(s['global_min']):>15} "
          f"{s['Q']:>6.3f}{s['S']:>6.3f}{s['B']:>6.3f}{s['W']:>6.3f} {s['mb']:>8.2f} {s['wall']:>7.0f}")
print("\nv1 ranking:", " > ".join(sorted(summary, key=lambda t: -summary[t]["v1"][0])))
print("v2 ranking:", " > ".join(sorted(summary, key=lambda t: -summary[t]["v2"][0])))
print("shared-accuracy ranking:", " > ".join(sorted(summary, key=lambda t: -summary[t]["global_acc"][0])))
print("\nlearning curves (shared-split mean accuracy per round):")
for t, s in summary.items(): print(f"  {t:<19}", " ".join(f"{v:.3f}" for v in s["curve"]))
# Sensitivity: every weight vector on a 0.05 grid with quality weight >= 0.2.
grid, wins, total = [x / 20 for x in range(21)], {}, 0
for wq, ws, wb in itertools.product(grid, repeat=3):
    ww = round(1 - wq - ws - wb, 10)
    if ww < -1e-9 or wq < 0.2: continue
    total += 1
    score = {t: st.mean(wq * x["c"]["quality"] + ws * x["c"]["learning_speed"] + wb * x["c"]["bandwidth_efficiency"] + ww * x["c"]["wait_efficiency"] for x in rows[t]) for t in summary}
    w = max(score, key=score.get); wins[w] = wins.get(w, 0) + 1
print(f"\nweight sensitivity ({total} weight vectors, quality weight >= 0.2): " + ", ".join(f"{t} wins {100*n/total:.1f}%" for t, n in sorted(wins.items(), key=lambda kv: -kv[1])))
# Smallest bandwidth weight at which mesh loses (quality/speed/wait in default 0.5:0.2:0.15 ratio).
flip = None
for wb in [x / 100 for x in range(101)]:
    rest = 1 - wb; wq, ws, ww = rest * 0.5 / 0.85, rest * 0.2 / 0.85, rest * 0.15 / 0.85
    score = {t: st.mean(wq * x["c"]["quality"] + ws * x["c"]["learning_speed"] + wb * x["c"]["bandwidth_efficiency"] + ww * x["c"]["wait_efficiency"] for x in rows[t]) for t in summary}
    if max(score, key=score.get) != "mesh": flip = (wb, max(score, key=score.get)); break
print("mesh stays first until bandwidth weight =", flip)
json.dump({"group_id": gid, "seeds": g["seeds"], "selection": g.get("selection"), "summary": summary, "runs": rows,
           "sensitivity": {"total": total, "wins": wins, "bandwidth_flip": flip}}, open(out, "w", encoding="utf-8"), indent=1, default=str)
