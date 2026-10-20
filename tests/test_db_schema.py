"""Unit tests for SQLAlchemy 2.0 ORM models and schema initialization."""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

# Allow importing backend app
backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database.session import Base, check_db, engine  # noqa: E402
from app.models import (  # noqa: E402
    AI4IState,
    BusinessDaily,
    BusinessTransaction,
    Machine,
    MaintenanceRecord,
    MetroPTFailureWindow,
    ProductionDaily,
    SensorRaw,
)

EXPECTED_TABLES = {
    "machines",
    "sensors_raw",
    "maintenance_records",
    "production_daily",
    "business_daily",
    "business_transactions",
    "ai4i_state",
    "metropt_failure_windows",
}


def test_orm_models_registered_on_metadata():
    """All 8 entity tables must be registered on Base.metadata."""
    registered = set(Base.metadata.tables.keys())
    assert EXPECTED_TABLES.issubset(registered), (
        f"Missing tables: {EXPECTED_TABLES - registered}"
    )


def test_sqlite_in_memory_schema_and_crud():
    """Verify schema creation and basic CRUD in an isolated in-memory engine."""
    mem_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=mem_engine)

    now = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

    with Session(mem_engine) as session:
        # 1. Insert Machine
        m = Machine(
            machine_id="M-101",
            machine_type="Air Compressor",
            location="Plant A, Line 1",
            installation_date=date(2021, 3, 15),
            status="Running",
            data_origin="metropt_derived",
            scenario="healthy",
            ai4i_type="L",
        )
        session.add(m)
        session.flush()

        # 2. Insert SensorRaw
        s = SensorRaw(
            timestamp=now,
            machine_id="M-101",
            temperature=62.5,
            vibration=1.85,
            pressure=7.02,
            motor_current=6.4,
            energy_consumption=0.31,
            rpm=1480.0,
            comp_state=0.0,
            operating_hours=28000.0,
        )
        session.add(s)

        # 3. Insert MaintenanceRecord
        maint = MaintenanceRecord(
            maintenance_id="maint-uuid-1",
            machine_id="M-101",
            maintenance_type="inspection",
            scheduled_date=now,
            performed_date=now,
            duration_hours=2.5,
            cost_gbp=250.0,
            parts_replaced=[],
            technician_notes="All clear",
            root_cause=None,
            failure_mode=None,
        )
        session.add(maint)

        # 4. Insert ProductionDaily
        prod = ProductionDaily(
            production_id="prod-uuid-1",
            date=date(2026, 4, 1),
            machine_id="M-101",
            shift="daily",
            units_produced=580,
            units_target=600,
            scrap_units=12,
            operating_hours=18.5,
            downtime_hours=5.5,
            downtime_reason="None",
            oee=0.7654,
        )
        session.add(prod)

        # 5. Insert BusinessDaily
        biz = BusinessDaily(
            date=date(2026, 4, 1),
            plant_id="PLANT-GLOBAL",
            total_production_units=12500,
            total_scrap_units=240,
            total_operating_hours=210.5,
            total_downtime_hours=77.5,
            total_energy_kwh=450.0,
            energy_cost_gbp=99.0,
            maintenance_cost_gbp=250.0,
            revenue_gbp=181250.0,
            inventory_level=24000,
            backorder_units=0,
        )
        session.add(biz)

        # 6. Insert BusinessTransaction
        trans = BusinessTransaction(
            transaction_id="trans-uuid-1",
            invoice_no="536365",
            stock_code="85123A",
            description="WHITE HANGING HEART T-LIGHT HOLDER",
            quantity=6,
            unit_price=2.55,
            customer_id="17850",
            country="United Kingdom",
            transaction_date=now,
            is_cancellation=False,
            line_total=15.30,
        )
        session.add(trans)

        # 7. Insert AI4IState
        ai4i = AI4IState(
            state_id="ai4i-uuid-1",
            udi=1,
            product_id="M14860",
            machine_type="Milling Machine",
            machine_variant="M",
            air_temperature_k=298.1,
            process_temperature_k=308.6,
            rotational_speed_rpm=1551,
            torque_nm=42.8,
            tool_wear_min=0,
            machine_failure=False,
            twf=False,
            hdf=False,
            pwf=False,
            osf=False,
            rnf=False,
        )
        session.add(ai4i)

        # 8. Insert MetroPTFailureWindow
        fw = MetroPTFailureWindow(
            failure_id="FAIL-001",
            start_time=now,
            end_time=now,
            failure_type="Air Leak",
            source="Data Description_Metro.pdf",
        )
        session.add(fw)

        session.commit()

        # Query back and verify
        stmt = select(Machine).where(Machine.machine_id == "M-101")
        machine_res = session.scalars(stmt).one()
        assert machine_res.machine_type == "Air Compressor"
        assert len(machine_res.sensors) == 1
        assert float(machine_res.sensors[0].vibration) == 1.85


def test_postgresql_connection_if_available():
    """If PostgreSQL is running, verify connection and schema creation."""
    if not check_db():
        pytest.skip("PostgreSQL not reachable. Start via 'docker compose up -d db' to test.")

    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        stmt = select(Machine)
        _ = session.scalars(stmt).all()
