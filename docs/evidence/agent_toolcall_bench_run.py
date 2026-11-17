"""Run tool-calling scenarios against the local Ollama on the master (python3, stdlib only)."""
import json
import sys
import time
import urllib.request
from pathlib import Path

base = Path(sys.argv[1])
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
valid = json.loads((base / "valid_tools.json").read_text())


def post(path, body):
    req = urllib.request.Request(
        "http://127.0.0.1:11434" + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def mem_available_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return -1


results = {}
for name in sorted(valid):
    body = json.loads((base / f"{name}.json").read_text())
    rows = []
    for i in range(repeats):
        started = time.time()
        try:
            r = post("/api/chat", body)
        except Exception as e:  # noqa: BLE001
            rows.append({"error": str(e)})
            continue
        msg = r.get("message", {})
        calls = msg.get("tool_calls") or []
        first = calls[0]["function"] if calls else None
        rows.append({
            "seconds": round(time.time() - started, 1),
            "prompt_tokens": r.get("prompt_eval_count"),
            "output_tokens": r.get("eval_count"),
            "tool_call": bool(calls),
            "tool": first["name"] if first else None,
            "valid_name": bool(first and first["name"] in valid[name]),
            "arguments": first.get("arguments") if first else None,
            "content_preview": (msg.get("content") or "")[:160],
        })
    ps = json.loads(urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=10).read())
    loaded = [{"size_mb": m.get("size", 0) // 2**20, "vram_mb": m.get("size_vram", 0) // 2**20,
               "ctx": m.get("context_length")} for m in ps.get("models", [])]
    results[name] = {"rows": rows, "loaded": loaded, "mem_available_mb": mem_available_mb()}
    ok = sum(1 for x in rows if x.get("valid_name"))
    print(f"{name}: valid tool calls {ok}/{len(rows)}; prompt tokens {[x.get('prompt_tokens') for x in rows]}; "
          f"tools {[x.get('tool') for x in rows]}; secs {[x.get('seconds') for x in rows]}; loaded {loaded}; "
          f"MemAvailable {results[name]['mem_available_mb']} MB", flush=True)
(base / "results.json").write_text(json.dumps(results, indent=1))
