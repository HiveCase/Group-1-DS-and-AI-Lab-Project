from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class DecisionRequest(BaseModel):
    decision: str = Field(pattern='^(approved|denied|request_more_info)$')
    reasoning_note: str = Field(min_length=1)
    settlement_amount: Decimal | None = None


class DecisionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    decision: str
    reasoning_note: str
    settlement_amount: Decimal | None = None
    decided_at: datetime | None = None


class AdjusterDashboardSummary(BaseModel):
    pending_count: int
    approved_count: int
    denied_count: int


class AdjusterDashboardResponse(BaseModel):
    summary: AdjusterDashboardSummary
    claims: list[dict]
