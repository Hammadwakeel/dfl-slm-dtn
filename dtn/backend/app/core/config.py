"""Environment and YAML-backed backend settings."""

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class DeviceConfig:
    node_id: str
    hostname: str
    ip: str
    role: str
    port: int
    token: str

    @property
    def base_url(self) -> str:
        return f"http://{self.ip}:{self.port}"


@dataclass(frozen=True)
class Settings:
    app_name: str
    host: str
    port: int
    api_token: str
    allowed_origins: tuple[str, ...]
    database_url: str
    nodes: tuple[DeviceConfig, ...]
    poll_interval_seconds: float
    request_timeout_seconds: float
    history_retention_count: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(REPO_ROOT / ".env", override=False)
        api_token = os.environ.get("DTN_API_TOKEN", "")
        if len(api_token) < 32 or api_token.startswith("replace-"):
            raise ValueError("DTN_API_TOKEN must be a random secret with at least 32 characters")

        origins = tuple(
            origin.strip()
            for origin in os.environ.get(
                "DTN_ALLOWED_ORIGINS", "http://localhost:5173,http://192.168.11.12:5173"
            ).split(",")
            if origin.strip()
        )
        if not origins:
            raise ValueError("DTN_ALLOWED_ORIGINS must contain at least one origin")

        database_url = os.environ.get(
            "DTN_DATABASE_URL",
            f"sqlite:///{(REPO_ROOT / 'dtn' / 'backend' / 'data' / 'dtn.sqlite3').as_posix()}",
        )
        if not database_url.startswith("sqlite:///"):
            raise ValueError("This initial DTN backend supports SQLite URLs only")

        raw_poll_interval = float(os.environ.get("DTN_POLL_INTERVAL_SECONDS", "3"))
        raw_timeout = float(os.environ.get("DTN_DEVICE_TIMEOUT_SECONDS", "2.5"))
        retention_count = int(os.environ.get("DTN_HISTORY_RETENTION_COUNT", "2000"))
        if raw_poll_interval < 1 or raw_timeout <= 0 or retention_count < 10:
            raise ValueError("Poll interval, device timeout, or history retention is out of range")

        registry_path = Path(
            os.environ.get("DTN_NODES_CONFIG", str(REPO_ROOT / "config" / "nodes.yaml"))
        )
        if not registry_path.is_absolute():
            registry_path = (REPO_ROOT / registry_path).resolve()
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        if not isinstance(registry, dict):
            raise ValueError(f"Device registry must be a YAML mapping: {registry_path}")
        nodes = _load_devices(registry)
        return cls(
            app_name="DFL-SLM Digital Twin Network",
            host=os.environ.get("DTN_HOST", "0.0.0.0"),
            port=int(os.environ.get("DTN_BACKEND_PORT", "8000")),
            api_token=api_token,
            allowed_origins=origins,
            database_url=database_url,
            nodes=nodes,
            poll_interval_seconds=raw_poll_interval,
            request_timeout_seconds=raw_timeout,
            history_retention_count=retention_count,
        )


def _load_devices(registry: dict[str, Any]) -> tuple[DeviceConfig, ...]:
    devices: list[DeviceConfig] = []
    master = registry.get("master")
    if not isinstance(master, dict):
        raise ValueError("Device registry must define a master node")
    entries: list[tuple[str, str, dict[str, Any]]] = [
        ("master", "master", master),
        *[
            (str(item.get("id", "")), "worker", item)
            for item in registry.get("workers", [])
            if isinstance(item, dict)
        ],
    ]
    seen_ids: set[str] = set()
    for node_id, role, item in entries:
        if not node_id or node_id in seen_ids:
            raise ValueError(f"Missing or duplicate node ID in {role} registry")
        seen_ids.add(node_id)
        token_key = f"{node_id.upper()}_API_TOKEN"
        token = os.environ.get(token_key, "")
        if len(token) < 32 or token.startswith("replace-"):
            raise ValueError(f"{token_key} must be set to the matching Jetson API token")
        ip = str(item.get("ip", ""))
        hostname = str(item.get("hostname", ""))
        if not ip or not hostname:
            raise ValueError(f"Node {node_id} must define an IP and hostname")
        port = int(item.get("port", item.get("service_port", 0)))
        if not 1 <= port <= 65535:
            raise ValueError(f"Node {node_id} has an invalid service port")
        devices.append(
            DeviceConfig(
                node_id=node_id,
                hostname=hostname,
                ip=ip,
                role=role,
                port=port,
                token=token,
            )
        )
    if len(devices) != 4:
        raise ValueError(f"Expected one master and three workers, found {len(devices)} devices")
    return tuple(devices)


def create_local_api_token() -> str:
    """Generate a cryptographically random local API token for setup scripts."""
    return secrets.token_urlsafe(48)
