from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

class ClaimCreate(BaseModel):
    policy_number: str = Field(min_length=1)
    claimant_name: str = Field(min_length=1)
    contact_info: str = Field(min_length=1)
    incident_date: date
    incident_description: str = Field(min_length=1)
    claimed_amount: Decimal = Field(gt=0)


class ClaimPhotoRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    file_path: str
    original_filename: str
    mime_type: str
    annotated_path: str | None = None


class AnalysisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: str
    severity_label: str | None = None
    severity_score: Decimal | None = None
    policy_findings: dict | None = None
    recommendation: str | None = None
    confidence_score: Decimal | None = None
    explanation: str | None = None

class ClaimRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    claim_id: str
    status: str
    message: str = 'Claim received'
    claimant_name: str | None = None
    incident_date: date | None = None
    incident_description: str | None = None
    claimed_amount: Decimal | None = None
    submitted_at: datetime | None = None
    photos: list[ClaimPhotoRead] = []
    analysis_result: AnalysisRead | None = None


class ClaimListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    claim_id: str
    status: str
    claimant_name: str
    claimed_amount: Decimal
    submitted_at: datetime


class ClaimListResponse(BaseModel):
    claims: list[ClaimListItem]
