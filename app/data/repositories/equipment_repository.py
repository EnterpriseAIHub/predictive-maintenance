from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.models.equipment import Equipment


def get_by_id(db: Session, equipment_id: str) -> Equipment | None:
    return db.get(Equipment, equipment_id)


def list_all(db: Session) -> list[Equipment]:
    return list(db.scalars(select(Equipment).order_by(Equipment.id)))
