from datetime import date
from decimal import Decimal

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.models import AnalysisResult, Claim, ClaimPhoto
from app.services.photo_storage_service import PhotoStorageService
from app.services.policy_service import PolicyService


class ClaimService:
    def __init__(self, db: Session):
        self.db = db
        self.policy_service = PolicyService(db)

    async def create_claim(
        self,
        policy_number: str,
        claimant_name: str,
        contact_info: str,
        incident_date: date,
        incident_description: str,
        claimed_amount: Decimal,
        photos: list[UploadFile],
        vehicle_no: str | None = None,
    ) -> Claim:
        policy = self.policy_service.get_by_number(policy_number)
        if not policy or policy.status != "active":
            raise HTTPException(status_code=404, detail="policy not found")

        claim = Claim(
            claim_id=self._next_claim_id(),
            policy=policy,
            claimant_name=claimant_name,
            contact_info=contact_info,
            vehicle_no=vehicle_no,
            incident_date=incident_date,
            incident_description=incident_description,
            claimed_amount=claimed_amount,
            status="submitted",
        )
        self.db.add(claim)
        self.db.flush()

        storage = PhotoStorageService()
        for photo in photos:
            path, original_name, mime_type = await storage.save(claim.claim_id, photo)
            claim.photos.append(ClaimPhoto(file_path=path, original_filename=original_name, mime_type=mime_type))

        claim.analysis_result = AnalysisResult(status="pending")
        self.db.commit()
        self.db.refresh(claim)
        return claim

    def get_claim(self, claim_id: str) -> Claim:
        claim = self.db.query(Claim).filter(Claim.claim_id == claim_id).first()
        if not claim:
            raise HTTPException(status_code=404, detail="claim not found")
        return claim

    def list_claims(self, status: str | list[str] | None = None) -> list[Claim]:
        query = self.db.query(Claim).order_by(Claim.submitted_at.desc())
        if isinstance(status, str):
            query = query.filter(Claim.status == status)
        elif status:
            query = query.filter(Claim.status.in_(status))
        return query.all()

    def _next_claim_id(self) -> str:
        next_id = (self.db.query(Claim).count() or 0) + 1001
        return f"CLM-{next_id}"
