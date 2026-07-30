"""Importing every model here (and this module in alembic/env.py)
registers all tables on Base.metadata — without this, Alembic
autogenerate would silently see an empty schema.
"""

from app.data.models.equipment import Equipment
from app.data.models.risk_score import RiskScore
from app.data.models.sensor_reading import SensorReading
from app.data.models.work_order import WorkOrder

__all__ = ["Equipment", "SensorReading", "WorkOrder", "RiskScore"]
