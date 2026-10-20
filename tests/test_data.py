"""Unit tests for raw and unified datasets in AI Operations Intelligence Platform."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ml.preprocessing.config import (
    TIMELINE_END,
    TIMELINE_START,
    UNIFIED,
)
from ml.preprocessing.fleet import FLEET


@pytest.fixture(scope="session")
def machines_df() -> pd.DataFrame:
    path = UNIFIED / "machines.csv"
    if not path.exists():
        pytest.skip("machines.csv not generated")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def maintenance_df() -> pd.DataFrame:
    path = UNIFIED / "maintenance_raw.csv"
    if not path.exists():
        pytest.skip("maintenance_raw.csv not generated")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def production_df() -> pd.DataFrame:
    path = UNIFIED / "production_raw.csv"
    if not path.exists():
        pytest.skip("production_raw.csv not generated")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def business_daily_df() -> pd.DataFrame:
    path = UNIFIED / "business_daily_raw.csv"
    if not path.exists():
        pytest.skip("business_daily_raw.csv not generated")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def ai4i_df() -> pd.DataFrame:
    path = UNIFIED / "ai4i_state_raw.csv"
    if not path.exists():
        pytest.skip("ai4i_state_raw.csv not generated")
    return pd.read_csv(path)


def test_machines_catalog(machines_df: pd.DataFrame):
    """machines.csv must have 12 rows, all valid machine IDs, and valid columns."""
    assert len(machines_df) == 12
    fleet_ids = [m.machine_id for m in FLEET]
    assert list(machines_df["machine_id"]) == fleet_ids
    assert set(machines_df["status"]).issubset({"Running", "Warning", "Critical", "Maintenance"})


def test_maintenance_records(maintenance_df: pd.DataFrame, machines_df: pd.DataFrame):
    """maintenance_raw.csv must have 30-60 rows, valid JSON, and FKs to machines."""
    assert 30 <= len(maintenance_df) <= 60
    assert set(maintenance_df["machine_id"]).issubset(set(machines_df["machine_id"]))
    assert (maintenance_df["duration_hours"] > 0).all()
    assert (maintenance_df["cost_gbp"] > 0).all()

    # Parts replaced valid JSON
    for parts in maintenance_df["parts_replaced"]:
        parsed = json.loads(parts)
        assert isinstance(parsed, list)

    # M-104 inspection anchor
    m104_recs = maintenance_df[
        (maintenance_df["machine_id"] == "M-104")
        & (maintenance_df["performed_date"].str.startswith("2026-09-20"))
    ]
    assert len(m104_recs) == 1
    expected_text = "Minor air leak suspected on discharge manifold"
    assert expected_text in m104_recs.iloc[0]["technician_notes"]


def test_production_daily(production_df: pd.DataFrame, machines_df: pd.DataFrame):
    """production_raw.csv must have exactly 2,556 rows, OEE in [0, 1], and scrap elevation."""
    assert len(production_df) == 2556
    assert set(production_df["machine_id"]).issubset(set(machines_df["machine_id"]))

    dates = pd.to_datetime(production_df["date"]).dt.date
    assert dates.min() == TIMELINE_START
    assert dates.max() == TIMELINE_END

    assert ((production_df["oee"] >= 0.0) & (production_df["oee"] <= 1.0)).all()

    # M-104 scrap rate elevated in final 14 days
    m104 = production_df[production_df["machine_id"] == "M-104"].copy()
    m104["scrap_rate"] = m104["scrap_units"] / m104["units_produced"]
    early_scrap = m104.iloc[:-14]["scrap_rate"].mean()
    late_scrap = m104.iloc[-14:]["scrap_rate"].mean()
    assert late_scrap > 2.0 * early_scrap


def test_business_daily(business_daily_df: pd.DataFrame):
    """business_daily_raw.csv must have 213 rows, positive revenue, non-negative inventory."""
    assert len(business_daily_df) == 213
    assert (business_daily_df["inventory_level"] >= 0).all()
    assert (business_daily_df["backorder_units"] >= 0).all()
    assert (business_daily_df["revenue_gbp"] > 0).all()
    assert (business_daily_df["total_energy_kwh"] > 0).all()
    assert (business_daily_df["plant_id"] == "PLANT-GLOBAL").all()


def test_ai4i_state(ai4i_df: pd.DataFrame):
    """ai4i_state_raw.csv must have 10,000 rows, valid failure flags, realistic physics."""
    assert len(ai4i_df) == 10000
    assert (ai4i_df["air_temperature_k"] > 250.0).all()
    assert (ai4i_df["rotational_speed_rpm"] > 500).all()
    total_failures = ai4i_df["machine_failure"].sum()
    assert 300 <= total_failures <= 400
