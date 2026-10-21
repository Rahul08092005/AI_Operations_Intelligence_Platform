"""Validation rules and physical limits for pipeline validation and cleaning.

Hard physical limits represent physically impossible boundaries. Values outside
are sensor faults or transmission errors, not merely unusual observations.
"""

from __future__ import annotations

from typing import NamedTuple


class PhysicalLimit(NamedTuple):
    hard_min: float
    hard_max: float
    unit: str
    description: str


# Hard physical limits per signal/column
SENSOR_LIMITS: dict[str, PhysicalLimit] = {
    "temperature": PhysicalLimit(
        -20.0, 150.0, "deg_c", "Compressor/bearing temperature in degrees Celsius"
    ),
    "vibration": PhysicalLimit(0.0, 30.0, "mm/s", "RMS vibration in mm/s"),
    "pressure": PhysicalLimit(0.0, 20.0, "bar", "Discharge/line pressure in bar"),
    "rpm": PhysicalLimit(0.0, 4000.0, "rpm", "Spindle or shaft rotational speed"),
    "energy_consumption": PhysicalLimit(
        0.0, 10.0, "kwh", "Energy consumption per 5-minute interval in kWh"
    ),
    "motor_current": PhysicalLimit(0.0, 100.0, "a", "Motor current in Amperes"),
    "operating_hours": PhysicalLimit(0.0, 200000.0, "hours", "Accumulated machine operating hours"),
    "comp_state": PhysicalLimit(0.0, 1.0, "binary", "Compressor load state (0=load, 1=offload)"),
}

PRODUCTION_LIMITS: dict[str, PhysicalLimit] = {
    "production_quantity": PhysicalLimit(0.0, 20000.0, "units", "Daily machine production units"),
    "target_quantity": PhysicalLimit(0.0, 20000.0, "units", "Daily target production units"),
    "downtime_minutes": PhysicalLimit(0.0, 1440.0, "minutes", "Daily downtime minutes"),
}

MAINTENANCE_LIMITS: dict[str, PhysicalLimit] = {
    "maintenance_cost": PhysicalLimit(0.0, 100000.0, "gbp", "Maintenance event cost in GBP"),
}

ALLOWED_MAINTENANCE_TYPES: set[str] = {
    "Inspection",
    "Preventive",
    "Corrective",
    "Predictive",
}

NON_PRODUCT_CODE_REGEX: str = r"^\d{5}[A-Za-z]*$"
