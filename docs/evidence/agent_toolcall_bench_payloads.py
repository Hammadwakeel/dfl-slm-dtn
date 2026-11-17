"""Build tool-calling benchmark payloads for the master's Ollama (run on the PC)."""
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
out = Path(sys.argv[2])
sys.path.insert(0, str(repo))
from common.agent_tools import agent_tool_definitions  # noqa: E402

full_tools = agent_tool_definitions()

# A realistic ~4,000-character snapshot like the one the coordinator sends today.
devices = [
    {
        "node_id": f"worker{i}",
        "online": True,
        "training_ready": True,
        "memory_available_mb": 5900 - i * 37,
        "cpu_percent": 3.1 + i,
        "gpu_percent": 0.0,
        "temperature_c": 44.2 + i / 10,
    }
    for i in range(1, 6)
]
groups = [
    {
        "group_id": "directed_ring-1794214683-7c17a1",
        "kind": "decentralized",
        "dataset_name": "mnist",
        "status": "completed",
        "topology": "directed_ring",
        "rounds": 5,
        "epochs": 10,
        "dataset_split": "dirichlet_a03",
        "final_mean_validation_accuracy": 0.9124,
        "final_mean_validation_macro_f1": 0.9031,
        "per_worker_validation_accuracy": {f"worker{i}": round(0.89 + i / 200, 4) for i in range(1, 6)},
        "runtime_seconds": 1412.6,
        "peak_process_rss_mb": 1630.4,
        "bytes_sent": 2609813,
    }
]
snapshot = {
    "captured_at": "2026-11-17T08:00:00Z",
    "global_stop": False,
    "devices": devices,
    "datasets": [
        {
            "name": "mnist",
            "train_images": 60000,
            "test_images": 10000,
            "preprocessed": True,
            "shards": {"split": "dirichlet_a03", "workers": 5, "samples_per_worker": [9811, 12640, 8122, 15003, 14424]},
        }
    ],
    "experiments": {"active_group_id": None, "groups": groups},
    "previous_actions": [
        {"action": "preprocess_dataset", "status": "executed", "parameters": {"dataset_name": "mnist", "dataset_split": "dirichlet_a03"}},
        {"action": "create_initial_model", "status": "executed", "parameters": {"model_name": "mnist"}},
        {"action": "start_training", "status": "executed", "parameters": {"dataset_name": "mnist", "rounds": 5, "epochs": 10}},
    ],
}
snapshot_text = json.dumps(snapshot, separators=(",", ":"))
while len(snapshot_text) < 4000:
    snapshot["previous_actions"].append(
        {"action": "get_results", "status": "executed", "result": {"groups": 1, "note": "baseline completed"}}
    )
    snapshot_text = json.dumps(snapshot, separators=(",", ":"))

goal = (
    "Goal: reach at least 0.95 mean worker validation accuracy on mnist. The baseline DFL "
    "run has completed. Choose the next step."
)
long_system = (
    "You operate this distributed-learning network only through the provided function tools. "
    "For every workflow decision, you MUST issue exactly one native function-tool call from "
    "the provided tools, or call finish_agent_workflow. Never describe a tool call in prose. "
    "Follow the exact schema and use only the supplied snapshot as evidence. This workflow is "
    "bound to dataset 'mnist'. System snapshot JSON:\n" + snapshot_text
)

compact_tools = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required, "additionalProperties": False},
        },
    }
    for name, description, props, required in [
        ("run_nas", "Search a better small CNN architecture on every worker's local data.",
         {"budget_seconds": {"type": "integer", "minimum": 60, "maximum": 1800}}, []),
        ("learn_topology", "Learn a better peer-to-peer communication graph from measured losses and delays.",
         {}, []),
        ("run_hpo", "Tune learning rate, batch size and local epochs on worker-local validation data.",
         {"trials": {"type": "integer", "minimum": 2, "maximum": 12}}, []),
        ("train_final", "Retrain with the best architecture, hyperparameters and topology found so far.",
         {"rounds": {"type": "integer", "minimum": 1, "maximum": 20}, "local_epochs": {"type": "integer", "minimum": 1, "maximum": 10}},
         ["rounds", "local_epochs"]),
        ("finish_workflow", "End the workflow with an outcome and short evidence.",
         {"outcome": {"enum": ["objective_achieved", "no_worthwhile_action", "blocked"]},
          "summary": {"type": "string", "maxLength": 300}}, ["outcome", "summary"]),
    ]
]
compact_state = {
    "goal": {"dataset": "mnist", "target_validation_accuracy": 0.95},
    "stage": "baseline_completed",
    "done": ["preprocess", "initial_model", "baseline_dfl"],
    "best": {"validation_accuracy": 0.9124, "macro_f1": 0.9031, "architecture": "b1_c8_k3", "topology": "directed_ring"},
    "cluster": {"workers_online": 5, "min_worker_free_mb": 5715, "active_job": None},
    "budget": {"experiments_left": 6, "minutes_left": 600},
}
compact_system = (
    "You are the DFL supervisor. Decide the single next step by calling exactly one tool. "
    "Base the decision only on the state JSON. Prefer the step most likely to raise validation "
    "accuracy toward the target; finish when the target is met or no step is worthwhile.\n"
    "State: " + json.dumps(compact_state, separators=(",", ":"))
)


def payload(system, tools, ctx):
    return {
        "model": "qwen3.5:4b",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": goal}],
        "tools": tools,
        "stream": False,
        "think": False,
        "options": {"num_ctx": ctx, "num_predict": 512, "temperature": 0.1},
        "keep_alive": "15m",
    }


out.mkdir(parents=True, exist_ok=True)
scenarios = {
    "A_full_tools_ctx4096": payload(long_system, full_tools, 4096),
    "B_compact_ctx4096": payload(compact_system, compact_tools, 4096),
    "C_compact_ctx8192": payload(compact_system, compact_tools, 8192),
    "D_full_tools_ctx8192": payload(long_system, full_tools, 8192),
}
valid = {
    "A_full_tools_ctx4096": [t["function"]["name"] for t in full_tools],
    "B_compact_ctx4096": [t["function"]["name"] for t in compact_tools],
    "C_compact_ctx8192": [t["function"]["name"] for t in compact_tools],
    "D_full_tools_ctx8192": [t["function"]["name"] for t in full_tools],
}
for name, body in scenarios.items():
    (out / f"{name}.json").write_text(json.dumps(body), encoding="utf-8")
    print(name, "request chars:", len(json.dumps(body)))
(out / "valid_tools.json").write_text(json.dumps(valid), encoding="utf-8")
