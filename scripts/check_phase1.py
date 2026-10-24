"""Verify authenticated status endpoints on the deployed Jetson nodes."""

import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from common.api_client import ApiClient, ApiClientError, NodeEndpoint

ROOT = Path(__file__).resolve().parent.parent
NODES = (
    ("master", "192.168.11.15", 5002, "MASTER_API_TOKEN", "master"),
    ("worker1", "192.168.11.123", 5001, "WORKER1_API_TOKEN", "worker"),
    ("worker2", "192.168.11.14", 5001, "WORKER2_API_TOKEN", "worker"),
    ("worker3", "192.168.11.13", 5001, "WORKER3_API_TOKEN", "worker"),
)


def load_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required local secret file was not found: {path}")
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, value = stripped.partition("=")
        if not separator or not name:
            raise ValueError(f"Invalid environment entry at {path}:{line_number}")
        values[name] = value
    return values


def verify_unauthorized(base_url: str, token: str | None = None) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    request = Request(
        base_url.rstrip("/") + "/v1/status",
        headers=headers,
        method="GET",
    )
    try:
        with urlopen(request, timeout=5) as response:
            raise AssertionError(
                f"Unauthenticated status request unexpectedly returned HTTP {response.status}"
            )
    except HTTPError as error:
        if error.code != 401:
            raise AssertionError(
                f"Unauthenticated status request returned HTTP {error.code}, expected 401"
            ) from error


def verify_status(
    node_id: str,
    role: str,
    address: str,
    port: int,
    token: str,
) -> dict[str, object]:
    base_url = f"http://{address}:{port}"
    verify_unauthorized(base_url)
    verify_unauthorized(base_url, "invalid-phase1-check-token")
    payload = ApiClient(
        NodeEndpoint(node_id=node_id, base_url=base_url, token=token)
    ).get_status()
    if payload.get("node_id") != node_id or payload.get("role") != role:
        raise AssertionError(f"{node_id} returned an unexpected identity")
    if payload.get("hostname") != f"dfl-{node_id}":
        raise AssertionError(f"{node_id} returned an unexpected hostname")
    memory = payload.get("memory")
    if not isinstance(memory, dict):
        raise AssertionError(f"{node_id} did not return RAM counters")
    for field in ("total_bytes", "available_bytes", "used_bytes"):
        if not isinstance(memory.get(field), int) or memory[field] <= 0:
            raise AssertionError(f"{node_id} returned an invalid RAM field: {field}")
    temperatures = payload.get("temperatures_celsius")
    if not isinstance(temperatures, dict) or not temperatures:
        raise AssertionError(f"{node_id} did not return thermal readings")
    return payload


def main() -> int:
    try:
        values = load_env(ROOT / ".env")
        for node_id, address, port, token_name, role in NODES:
            token = values.get(token_name, "")
            if len(token) < 32 or token.startswith("replace-"):
                raise ValueError(f"{token_name} is missing or is a placeholder")
            payload = verify_status(node_id, role, address, port, token)
            print(
                json.dumps(
                    {
                        "node_id": node_id,
                        "hostname": payload["hostname"],
                        "authenticated": True,
                        "unauthorized_rejected": True,
                        "cpu_utilization_percent": payload["cpu_utilization_percent"],
                        "gpu_utilization_percent": payload["gpu_utilization_percent"],
                        "memory": payload["memory"],
                        "temperatures_celsius": payload["temperatures_celsius"],
                        "sampled_at": payload["sampled_at"],
                    },
                    sort_keys=True,
                )
            )
    except (ApiClientError, AssertionError, FileNotFoundError, OSError, ValueError) as error:
        print(f"Phase 1 integration check failed: {error}", file=sys.stderr)
        return 1
    print("Phase 1 authentication and status checks passed for all four Jetsons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
