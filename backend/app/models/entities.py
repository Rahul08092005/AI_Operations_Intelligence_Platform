"""SQLAlchemy 2.0 ORM models for AI Operations Intelligence Platform."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.database.session import Base


class Machine(Base):
    """Machine master catalog."""

    __tablename__ = "machines"

    machine_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    machine_type: Mapped[str] = mapped_column(String(50), nullable=False)
    location: Mapped[str] = mapped_column(String(100), nullable=False)
    installation_date: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    data_origin: Mapped[str] = mapped_column(String(50), nullable=False)
    scenario: Mapped[str] = mapped_column(String(50), nullable=False)
    ai4i_type: Mapped[str] = mapped_column(String(1), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    sensors: Mapped[list[SensorRaw]] = relationship("SensorRaw", back_populates="machine")
    maintenance: Mapped[list[MaintenanceRecord]] = relationship(
        "MaintenanceRecord", back_populates="machine"
    )
    production: Mapped[list[ProductionDaily]] = relationship(
        "ProductionDaily", back_populates="machine"
    )


class SensorRaw(Base):
    """High-frequency sensor telemetry (5-minute resolution)."""

    __tablename__ = "sensors_raw"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    machine_id: Mapped[str] = mapped_column(
        String(10), ForeignKey("machines.machine_id"), nullable=False, index=True
    )
    temperature: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    vibration: Mapped[float | None] = mapped_column(Numeric(6, 3), nullable=True)
    pressure: Mapped[float | None] = mapped_column(Numeric(6, 3), nullable=True)
    motor_current: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)
    energy_consumption: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    rpm: Mapped[float | None] = mapped_column(Numeric(7, 2), nullable=True)
    comp_state: Mapped[float | None] = mapped_column(Numeric(3, 1), nullable=True)
    operating_hours: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    machine: Mapped[Machine] = relationship("Machine", back_populates="sensors")

    __table_args__ = (
        Index("ix_sensors_machine_time", "machine_id", "timestamp"),
    )


class MaintenanceRecord(Base):
    """Maintenance events, inspections, and overhauls."""

    __tablename__ = "maintenance_records"

    maintenance_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(10), ForeignKey("machines.machine_id"), nullable=False, index=True
    )
    maintenance_type: Mapped[str] = mapped_column(String(20), nullable=False)
    scheduled_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    performed_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    duration_hours: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    cost_gbp: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    parts_replaced: Mapped[Any] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    technician_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    root_cause: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_mode: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    machine: Mapped[Machine] = relationship("Machine", back_populates="maintenance")


class ProductionDaily(Base):
    """Daily production summary per machine."""

    __tablename__ = "production_daily"

    production_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    machine_id: Mapped[str] = mapped_column(
        String(10), ForeignKey("machines.machine_id"), nullable=False, index=True
    )
    shift: Mapped[str] = mapped_column(String(10), nullable=False, default="daily")
    units_produced: Mapped[int] = mapped_column(Integer, nullable=False)
    units_target: Mapped[int] = mapped_column(Integer, nullable=False)
    scrap_units: Mapped[int] = mapped_column(Integer, nullable=False)
    operating_hours: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    downtime_hours: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    downtime_reason: Mapped[str] = mapped_column(String(50), nullable=False)
    oee: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    machine: Mapped[Machine] = relationship("Machine", back_populates="production")

    __table_args__ = (
        UniqueConstraint("date", "machine_id", "shift", name="uq_production_daily_shift"),
    )


class BusinessDaily(Base):
    """Plant-wide daily business metrics."""

    __tablename__ = "business_daily"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    plant_id: Mapped[str] = mapped_column(String(20), nullable=False, default="PLANT-GLOBAL")
    total_production_units: Mapped[int] = mapped_column(Integer, nullable=False)
    total_scrap_units: Mapped[int] = mapped_column(Integer, nullable=False)
    total_operating_hours: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False)
    total_downtime_hours: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False)
    total_energy_kwh: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    energy_cost_gbp: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    maintenance_cost_gbp: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    revenue_gbp: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    inventory_level: Mapped[int] = mapped_column(Integer, nullable=False)
    backorder_units: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class BusinessTransaction(Base):
    """Online retail transactions powering demand model."""

    __tablename__ = "business_transactions"

    transaction_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    invoice_no: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    stock_code: Mapped[str] = mapped_column(String(20), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    customer_id: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    country: Mapped[str] = mapped_column(String(50), nullable=False)
    transaction_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    is_cancellation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    line_total: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AI4IState(Base):
    """Tabular equipment state observations from AI4I 2020 dataset."""

    __tablename__ = "ai4i_state"

    state_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    udi: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    product_id: Mapped[str] = mapped_column(String(10), nullable=False)
    machine_type: Mapped[str] = mapped_column(String(20), nullable=False)
    machine_variant: Mapped[str] = mapped_column(String(1), nullable=False)
    air_temperature_k: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    process_temperature_k: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    rotational_speed_rpm: Mapped[int] = mapped_column(Integer, nullable=False)
    torque_nm: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    tool_wear_min: Mapped[int] = mapped_column(Integer, nullable=False)
    machine_failure: Mapped[bool] = mapped_column(Boolean, nullable=False)
    twf: Mapped[bool] = mapped_column(Boolean, nullable=False)
    hdf: Mapped[bool] = mapped_column(Boolean, nullable=False)
    pwf: Mapped[bool] = mapped_column(Boolean, nullable=False)
    osf: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rnf: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class MetroPTFailureWindow(Base):
    """Ground truth failure events extracted from MetroPT dataset."""

    __tablename__ = "metropt_failure_windows"

    failure_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    failure_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
