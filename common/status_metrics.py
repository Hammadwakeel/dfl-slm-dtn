"""Real system metrics collected from Linux procfs, sysfs, and tegrastats."""

import logging
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
TEGRASTATS = Path("/usr/bin/tegrastats")
THERMAL_ROOT = Path("/sys/class/thermal")


def read_cpu_counters(path: Path = Path("/proc/stat")) -> tuple[int, int]:
    """Return total and idle CPU jiffies from the aggregate procfs row."""
    with path.open("r", encoding="ascii") as source:
        for line in source:
            if line.startswith("cpu "):
                values = [int(value) for value in line.split()[1:]]
                if len(values) < 4:
                    raise ValueError("Aggregate CPU counters are incomplete")
                idle = values[3] + (values[4] if len(values) > 4 else 0)
                return sum(values), idle
    raise ValueError("Aggregate CPU counters were not found in /proc/stat")


def calculate_cpu_percent(
    previous: tuple[int, int] | None,
    current: tuple[int, int],
) -> float | None:
    """Calculate aggregate CPU busy percentage between two procfs samples."""
    if previous is None:
        return None
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0 or idle_delta < 0:
        return None
    return round(max(0.0, min(100.0, (1 - idle_delta / total_delta) * 100)), 2)


def read_memory() -> dict[str, int]:
    """Return total, available, and used RAM in bytes from /proc/meminfo."""
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open("r", encoding="ascii") as source:
        for line in source:
            name, separator, value = line.partition(":")
            if separator and name in {"MemTotal", "MemAvailable"}:
                values[name] = int(value.strip().split()[0]) * 1024

    if "MemTotal" not in values or "MemAvailable" not in values:
        raise ValueError("MemTotal or MemAvailable was not found in /proc/meminfo")
    total = values["MemTotal"]
    available = values["MemAvailable"]
    if total <= 0 or available < 0 or available > total:
        raise ValueError("Memory counters from /proc/meminfo are out of range")
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": total - available,
    }


def read_temperatures(root: Path = THERMAL_ROOT) -> dict[str, float]:
    """Read all available thermal zones, naming each by its sysfs type."""
    readings: dict[str, float] = {}
    if not root.is_dir():
        return readings

    for zone in sorted(root.glob("thermal_zone*")):
        try:
            raw = int((zone / "temp").read_text(encoding="ascii").strip())
            label = (zone / "type").read_text(encoding="ascii").strip()
        except (OSError, TypeError, UnicodeDecodeError, ValueError) as error:
            LOGGER.warning("Could not read thermal zone %s: %s", zone.name, error)
            continue

        celsius = raw / 1000 if abs(raw) >= 1000 else float(raw)
        if not -20 <= celsius <= 150:
            LOGGER.warning("Ignoring out-of-range temperature for %s", zone.name)
            continue
        readings[label or zone.name] = round(celsius, 2)
    return readings


def parse_tegrastats(line: str) -> dict[str, float | None]:
    """Parse CPU and GPU utilization percentages from one tegrastats row."""
    cpu_match = re.search(r"CPU\s*\[([^\]]*)\]", line)
    cpu_values = (
        [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)%\s*@", cpu_match.group(1))]
        if cpu_match
        else []
    )
    gpu_match = re.search(r"GR3D_FREQ\s+(\d+(?:\.\d+)?)%", line)
    return {
        "cpu_utilization_percent": (
            round(sum(cpu_values) / len(cpu_values), 2) if cpu_values else None
        ),
        "gpu_utilization_percent": float(gpu_match.group(1)) if gpu_match else None,
    }


def read_tegrastats(executable: Path = TEGRASTATS) -> dict[str, float | None]:
    """Sample live CPU/GPU utilization, returning null when tegrastats is absent."""
    try:
        result = subprocess.run(
            [str(executable), "--interval", "1000"],
            capture_output=True,
            check=False,
            text=True,
            timeout=1.5,
        )
    except FileNotFoundError:
        LOGGER.warning("tegrastats is unavailable; GPU utilization cannot be sampled")
        return {"cpu_utilization_percent": None, "gpu_utilization_percent": None}
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        rows = [row for row in output.splitlines() if row.strip()]
        if rows:
            return parse_tegrastats(rows[-1])
        LOGGER.warning("tegrastats did not return a sample before the timeout")
        return {"cpu_utilization_percent": None, "gpu_utilization_percent": None}

    if result.returncode != 0 and not result.stdout.strip():
        LOGGER.warning("tegrastats exited with status %s: %s", result.returncode, result.stderr.strip())
        return {"cpu_utilization_percent": None, "gpu_utilization_percent": None}
    rows = [row for row in result.stdout.splitlines() if row.strip()]
    if not rows:
        LOGGER.warning("tegrastats returned no readings")
        return {"cpu_utilization_percent": None, "gpu_utilization_percent": None}
    return parse_tegrastats(rows[-1])


def collect_status(
    node_id: str,
    role: str,
    previous_cpu: tuple[int, int] | None,
) -> tuple[dict[str, Any], tuple[int, int]]:
    """Collect a real status snapshot and the counters for the next sample."""
    current_cpu = read_cpu_counters()
    memory = read_memory()
    tegra = read_tegrastats()
    tegra_cpu = tegra["cpu_utilization_percent"]
    cpu_percent = tegra_cpu
    if cpu_percent is None:
        cpu_percent = calculate_cpu_percent(previous_cpu, current_cpu)
    cpu_source = "tegrastats" if tegra_cpu is not None else (
        "procfs" if cpu_percent is not None else "unavailable"
    )
    return (
        {
            "node_id": node_id,
            "role": role,
            "hostname": socket.gethostname(),
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "cpu_utilization_percent": cpu_percent,
            "gpu_utilization_percent": tegra["gpu_utilization_percent"],
            "memory": memory,
            "temperatures_celsius": read_temperatures(),
            "metrics_source": {
                "cpu": cpu_source,
                "gpu": (
                    "tegrastats"
                    if tegra["gpu_utilization_percent"] is not None
                    else "tegrastats_unavailable"
                ),
            },
        },
        current_cpu,
    )
