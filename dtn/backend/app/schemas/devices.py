"""Validated response schemas for Jetson registry and status data."""

import ipaddress
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    training_port: int | None = None
    is_managed: bool = False
    training_ready: bool = False
    training_error: str | None = None
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


class DeviceRegistrationRequest(BaseModel):
    node_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    hostname: str = Field(min_length=1, max_length=128)
    ip: str = Field(min_length=1, max_length=64)
    port: int = Field(ge=1, le=65535)
    training_port: int = Field(ge=1, le=65535)
    api_token: str = Field(min_length=32, max_length=512)

    @field_validator("node_id")
    @classmethod
    def node_id_is_normalized(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,31}", normalized):
            raise ValueError("Device ID may contain lowercase letters, numbers, '.', '_' and '-'")
        return normalized

    @field_validator("hostname")
    @classmethod
    def hostname_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Device name must not be blank")
        return value.strip()

    @field_validator("api_token")
    @classmethod
    def api_token_is_not_placeholder(cls, value: str) -> str:
        if value.startswith("replace-"):
            raise ValueError("Use the device's actual API token")
        return value

    @field_validator("ip")
    @classmethod
    def ip_is_local_or_private(cls, value: str) -> str:
        try:
            address = ipaddress.ip_address(value)
        except ValueError as error:
            raise ValueError("Enter a valid IPv4 or IPv6 address") from error
        if not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
        ) or address.is_multicast or address.is_unspecified or address.is_reserved:
            raise ValueError("Only private, loopback, or link-local device addresses are allowed")
        return str(address)


class DeviceUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hostname: str | None = Field(default=None, min_length=1, max_length=128)
    ip: str | None = Field(default=None, min_length=1, max_length=64)
    port: int | None = Field(default=None, ge=1, le=65535)
    training_port: int | None = Field(default=None, ge=1, le=65535)
    api_token: str | None = Field(default=None, min_length=32, max_length=512)

    @model_validator(mode="after")
    def requires_changes(self) -> "DeviceUpdateRequest":
        if not self.model_fields_set or all(
            getattr(self, field) is None for field in self.model_fields_set
        ):
            raise ValueError("Provide at least one device field to update")
        return self

    @field_validator("hostname")
    @classmethod
    def hostname_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Device name must not be blank")
        return value.strip() if value is not None else None

    @field_validator("ip")
    @classmethod
    def ip_is_local_or_private(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            address = ipaddress.ip_address(value)
        except ValueError as error:
            raise ValueError("Enter a valid IPv4 or IPv6 address") from error
        if not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
        ) or address.is_multicast or address.is_unspecified or address.is_reserved:
            raise ValueError("Only private, loopback, or link-local device addresses are allowed")
        return str(address)

    @field_validator("api_token")
    @classmethod
    def api_token_is_not_placeholder(cls, value: str | None) -> str | None:
        if value is not None and value.startswith("replace-"):
            raise ValueError("Use the device's actual API token")
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
