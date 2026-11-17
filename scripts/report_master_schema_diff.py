"""Report the PC/master action-schema diff after authenticated idle checks."""

from __future__ import annotations

import difflib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import Settings  # noqa: E402
from deploy_master_nas import Backend, _preflight  # noqa: E402


SCHEMA_PATH = "/opt/dfl-slm-dtn/common/slm_action_schema.json"


def main() -> None:
    settings = Settings.from_env()
    master = next(node for node in settings.nodes if node.role == "master")
    ssh_user = os.environ.get("M4_SSH_USER", "hammad")
    backend = Backend("http://127.0.0.1:8000", settings.api_token)
    _preflight(backend, settings, ssh_user, master.ip)

    remote = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            f"{ssh_user}@{master.ip}",
            f"cat -- {SCHEMA_PATH}",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if remote.returncode:
        raise RuntimeError(f"Could not read master schema (SSH exit {remote.returncode})")

    local_path = ROOT / "common" / "slm_action_schema.json"
    local_text = local_path.read_text(encoding="utf-8")
    diff = difflib.unified_diff(
        local_text.splitlines(keepends=True),
        remote.stdout.splitlines(keepends=True),
        fromfile="PC/common/slm_action_schema.json",
        tofile="MASTER/common/slm_action_schema.json",
    )
    output = "".join(diff)
    print(output if output else "PC and master schema contents match.")


if __name__ == "__main__":
    main()
