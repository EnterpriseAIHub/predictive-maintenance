from datetime import datetime

from pydantic import BaseModel


class Equipment(BaseModel):
    id: str
    plant_id: str
    type: str
    install_date: datetime
    criticality_tier: int
