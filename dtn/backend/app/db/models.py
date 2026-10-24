"""SQLAlchemy models for the device registry and real status history."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Device(Base):
    __tablename__ = "devices"

    node_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    hostname: Mapped[str] = mapped_column(String(128), nullable=False)
    ip: Mapped[str] = mapped_column(String(64), nullable=False)
    port: Mapped[int] = mapped_column(Integer, nullable=False)
    online: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(String(512))
    latest_status: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    history: Mapped[list["DeviceStatus"]] = relationship(
        back_populates="device",
        cascade="all, delete-orphan",
        order_by="DeviceStatus.sampled_at.desc()",
    )


class DeviceStatus(Base):
    __tablename__ = "device_status_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(
        ForeignKey("devices.node_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sampled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    cpu_utilization_percent: Mapped[float | None] = mapped_column(Float)
    gpu_utilization_percent: Mapped[float | None] = mapped_column(Float)
    memory: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    temperatures_celsius: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    device: Mapped[Device] = relationship(back_populates="history")
