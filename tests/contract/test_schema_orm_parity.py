"""Verifies the Pydantic schemas (the cross-repo contract shape) stay
in sync with their ORM counterparts (the actual persisted shape).
Both `app/schemas/equipment.py` and `app/schemas/work_order.py` state
in their docstrings that they match their ORM model "exactly" — this
test enforces that claim rather than trusting the comment alone.
"""

from app.data.models.equipment import Equipment as EquipmentModel
from app.data.models.work_order import WorkOrder as WorkOrderModel
from app.schemas.equipment import Equipment as EquipmentSchema
from app.schemas.work_order import WorkOrder as WorkOrderSchema


def _orm_column_names(model_cls) -> set[str]:
    return {c.name for c in model_cls.__table__.columns}


def test_equipment_schema_covers_every_orm_column():
    schema_fields = set(EquipmentSchema.model_fields.keys())
    orm_columns = _orm_column_names(EquipmentModel)
    assert schema_fields == orm_columns


def test_work_order_schema_covers_every_orm_column():
    schema_fields = set(WorkOrderSchema.model_fields.keys())
    orm_columns = _orm_column_names(WorkOrderModel)
    assert schema_fields == orm_columns


def test_equipment_schema_round_trips_a_real_orm_instance():
    from datetime import UTC, datetime

    orm_row = EquipmentModel(
        id="eq-1",
        plant_id="plant-1",
        type="conveyor_motor",
        install_date=datetime(2022, 1, 1, tzinfo=UTC),
        criticality_tier=2,
    )
    # model_validate(..., from_attributes=True) is exactly how a route
    # would convert an ORM row to the contract shape — if this fails,
    # the contract is broken for real usage, not just in the abstract.
    schema_instance = EquipmentSchema.model_validate(orm_row, from_attributes=True)
    assert schema_instance.id == "eq-1"
    assert schema_instance.criticality_tier == 2
