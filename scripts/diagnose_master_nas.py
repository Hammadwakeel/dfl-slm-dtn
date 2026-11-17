"""Read-only comparison of master Jetson logs and deployed source hashes."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402


SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".sh", ".service"}


def _redact(text: str, secrets: Iterable[str]) -> str:
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        text = text.replace(secret, "<REDACTED>")
    text = re.sub(r"(?i)(Bearer\s+)\S+", r"\1<REDACTED>", text)
    text = re.sub(
        r"(?i)\b([A-Z0-9_]*(?:TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL)[A-Z0-9_]*=)\S+",
        r"\1<REDACTED>",
        text,
    )
    return text


def _ssh(user: str, host: str, command: str, timeout: int = 45) -> str:
    completed = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            f"{user}@{host}",
            command,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"SSH to master failed (exit {completed.returncode}): "
            f"{completed.stderr.strip()[:500]}"
        )
    return completed.stdout


def _source_files() -> list[Path]:
    files: list[Path] = []
    for directory in (ROOT / "master", ROOT / "common"):
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
                and ".venv" not in path.parts
            ):
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def _remote_hashes(host: str, user: str, paths: list[str]) -> dict[str, str | None]:
    remote_root = "/opt/dfl-slm-dtn"
    commands = [
        "for f in "
        + " ".join(shlex.quote(f"{remote_root}/{path}") for path in paths)
        + "; do "
        + 'if test -f "$f"; then sha256sum "$f"; else printf "MISSING  %s\\n" "$f"; fi; '
        + "done"
    ]
    output = _ssh(user, host, "sh -c " + shlex.quote(commands[0]))
    result: dict[str, str | None] = {}
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, remote_path = parts
        relative = remote_path.replace(remote_root + "/", "", 1)
        result[relative] = None if digest == "MISSING" else digest
    return result


def main() -> None:
    settings = Settings.from_env()
    master = next(node for node in settings.nodes if node.role == "master")
    ssh_user = os.environ.get("M4_SSH_USER", "hammad")
    secrets = [node.token for node in settings.nodes]
    secrets.extend(
        value
        for value in (settings.api_token, settings.master_slm_token)
        if value
    )

    journal = _ssh(
        ssh_user,
        master.ip,
        "journalctl -u dfl-slm-master.service -n 60 --no-pager --output=short-iso",
    )
    print("--- master service journal: last 60 lines, credentials redacted ---")
    for line in journal.splitlines()[-60:]:
        print(_redact(line, secrets))

    files = _source_files()
    relative_paths = [path.relative_to(ROOT).as_posix() for path in files]
    remote = _remote_hashes(master.ip, ssh_user, relative_paths)
    print("--- PC vs master SHA-256 comparison ---")
    stale: list[str] = []
    for path, relative in zip(files, relative_paths):
        local_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        remote_hash = remote.get(relative)
        print(f"{relative} PC={local_hash} MASTER={remote_hash or 'MISSING'}")
        if remote_hash != local_hash:
            stale.append(relative)
    print(f"master_code_stale={'yes' if stale else 'no'}")
    if stale:
        print("differing_paths=" + ",".join(stale))


if __name__ == "__main__":
    main()
