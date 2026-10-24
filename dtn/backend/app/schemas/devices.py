"""Validated response schemas for Jetson registry and status data."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryStatus(BaseModel):
    total_bytes: int = Field(ge=1)
    available_bytes: int = Field(ge=0)
    used_bytes: int = Field(ge=0)


class StatusPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    node_id: str
    role: str
    hostname: str
    sampled_at: datetime
    cpu_utilization_percent: float | None = Field(default=None, ge=0, le=100)
    gpu_utilization_percent: float | None = Field(default=None, ge=0, le=100)
    memory: MemoryStatus
    temperatures_celsius: dict[str, float]


class DeviceResponse(BaseModel):
    node_id: str
    role: str
    hostname: str
    ip: str
    port: int
    online: bool
    last_seen: datetime | None
    last_error: str | None
    latest_status: dict[str, Any] | None

    @field_validator("last_seen", mode="before")
    @classmethod
    def normalize_last_seen(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class StatusHistoryItem(BaseModel):
    id: int
    node_id: str
    sampled_at: datetime
    cpu_utilization_percent: float | None
    gpu_utilization_percent: float | None
    memory: dict[str, Any]
    temperatures_celsius: dict[str, float]
    payload: dict[str, Any]

    @field_validator("sampled_at", mode="before")
    @classmethod
    def normalize_sampled_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
