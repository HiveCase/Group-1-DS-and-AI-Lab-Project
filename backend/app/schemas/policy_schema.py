from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

class PolicyLookupRequest(BaseModel):
    policy_number: str = Field(min_length=1)

class PolicyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    policy_number: str
    coverage_type: str
    status: str
    effective_date: date
    policy_limit: Decimal | None = None
