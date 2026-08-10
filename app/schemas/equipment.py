"""Equipment schema.

Per the platform architecture, `Equipment` is a shared entity owned by
`platform-data-contracts`, not by this repo. That shared package
doesn't exist yet — this repo is being built first and must still be
independently runnable (NFR1) — so a matching definition lives here
temporarily. When `platform-data-contracts` is built, this file is
replaced by an import from that package; nothing else in this repo
should need to change, since the shape is intended to match exactly.
"""

from datetime import datetime

from pydantic import BaseModel


class Equipment(BaseModel):
    id: str
    plant_id: str
    type: str
    install_date: datetime
    criticality_tier: int
