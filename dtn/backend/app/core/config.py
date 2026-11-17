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
    training_port: int | None = None
    is_managed: bool = False

    @property
    def base_url(self) -> str:
        host = f"[{self.ip}]" if ":" in self.ip else self.ip
        return f"http://{host}:{self.port}"

    @property
    def training_url(self) -> str:
        if self.training_port is None:
            raise ValueError(f"Training API port is not configured for {self.node_id}")
        host = f"[{self.ip}]" if ":" in self.ip else self.ip
        return f"http://{host}:{self.training_port}"


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
    master_slm_url: str = "http://192.168.11.15:5003"
    master_slm_token: str = ""
    slm_timeout_seconds: float = 150.0
    dataset_root: Path = REPO_ROOT / "data" / "datasets"
    feedback_rule_fallback_enabled: bool = False
    agent_max_tool_calls: int = 8
    agent_history_messages: int = 8
    agent_action_history_limit: int = 20
    supervisor_min_free_memory_mb: int = 1200
    supervisor_tick_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(REPO_ROOT / ".env", override=False)
        api_token = os.environ.get("DTN_API_TOKEN", "")
        if len(api_token) < 32 or api_token.startswith("replace-"):
            raise ValueError("DTN_API_TOKEN must be a random secret with at least 32 characters")
        master_slm_token = os.environ.get("MASTER_SLM_API_TOKEN", "")
        if len(master_slm_token) < 32 or master_slm_token.startswith("replace-"):
            raise ValueError(
                "MASTER_SLM_API_TOKEN must be a random secret with at least 32 characters"
            )

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
        master = registry["master"]
        master_slm_port = int(master.get("slm_port", 0))
        if not 1 <= master_slm_port <= 65535:
            raise ValueError("Master registry entry must define a valid slm_port")
        slm_timeout = float(os.environ.get("DTN_SLM_TIMEOUT_SECONDS", "150"))
        if slm_timeout <= 0:
            raise ValueError("DTN_SLM_TIMEOUT_SECONDS must be positive")
        raw_feedback_fallback = os.environ.get(
            "DTN_FEEDBACK_RULE_FALLBACK",
            "false",
        ).strip().lower()
        if raw_feedback_fallback not in {"true", "false"}:
            raise ValueError("DTN_FEEDBACK_RULE_FALLBACK must be true or false")
        agent_max_tool_calls = int(os.environ.get("DTN_AGENT_MAX_TOOL_CALLS", "8"))
        agent_history_messages = int(os.environ.get("DTN_AGENT_HISTORY_MESSAGES", "8"))
        agent_action_history_limit = int(
            os.environ.get("DTN_AGENT_ACTION_HISTORY_LIMIT", "20")
        )
        supervisor_min_free_memory_mb = int(
            os.environ.get("DTN_SUPERVISOR_MIN_FREE_MEMORY_MB", "1200")
        )
        supervisor_tick_seconds = float(os.environ.get("DTN_SUPERVISOR_TICK_SECONDS", "10"))
        if not 256 <= supervisor_min_free_memory_mb <= 7000 or not 2 <= supervisor_tick_seconds <= 300:
            raise ValueError("Supervisor memory admission or tick interval is out of range")
        if (
            not 1 <= agent_max_tool_calls <= 32
            or not 0 <= agent_history_messages <= 32
            or not 1 <= agent_action_history_limit <= 100
        ):
            raise ValueError("Agent tool-call and history limits are out of range")
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
            master_slm_url=os.environ.get(
                "MASTER_SLM_URL",
                f"http://{master['ip']}:{master_slm_port}",
            ).rstrip("/"            ),
            master_slm_token=master_slm_token,
            slm_timeout_seconds=slm_timeout,
            dataset_root=Path(
                os.environ.get(
                    "DTN_DATASET_ROOT",
                    str(REPO_ROOT / "data" / "datasets"),
                )
            ).resolve(),
            feedback_rule_fallback_enabled=raw_feedback_fallback == "true",
            agent_max_tool_calls=agent_max_tool_calls,
            agent_history_messages=agent_history_messages,
            agent_action_history_limit=agent_action_history_limit,
            supervisor_min_free_memory_mb=supervisor_min_free_memory_mb,
            supervisor_tick_seconds=supervisor_tick_seconds,
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
        raw_training_port = item.get("training_port")
        training_port = int(raw_training_port) if raw_training_port is not None else None
        if training_port is not None and not 1 <= training_port <= 65535:
            raise ValueError(f"Node {node_id} has an invalid training service port")
        devices.append(
            DeviceConfig(
                node_id=node_id,
                hostname=hostname,
                ip=ip,
                role=role,
                port=port,
                token=token,
                training_port=training_port,
            )
        )
    worker_ids = sorted(device.node_id for device in devices if device.role == "worker")
    expected_worker_ids = [f"worker{index}" for index in range(1, len(worker_ids) + 1)]
    if len(worker_ids) < 3 or worker_ids != expected_worker_ids:
        raise ValueError("Device registry must define at least three sequentially numbered workers")
    return tuple(devices)


def create_local_api_token() -> str:
    """Generate a cryptographically random local API token for setup scripts."""
    return secrets.token_urlsafe(48)
