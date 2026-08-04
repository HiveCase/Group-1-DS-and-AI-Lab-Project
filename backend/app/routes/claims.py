from datetime import date
from decimal import Decimal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import DecisionRecord
from app.schemas.adjuster_schema import AdjusterDashboardResponse, AdjusterDashboardSummary, DecisionRead, DecisionRequest
from app.schemas.claim_schema import ClaimListResponse, ClaimRead
from app.services.analytics_service import AnalyticsService
from app.services.claim_service import ClaimService
from app.services.damage_analysis_service import DamageAnalysisService
from app.services.investigation_service import InvestigationService

router = APIRouter(prefix='/claims', tags=['claims'])

@router.post('', response_model=ClaimRead, status_code=201)
async def create_claim(
    request: Request,
    policy_number: str | None = Form(default=None),
    claimant_name: str | None = Form(default=None),
    contact_info: str | None = Form(default=None),
    incident_date: date | None = Form(default=None),
    incident_description: str | None = Form(default=None),
    claimed_amount: Decimal | None = Form(default=None),
    photos: list[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
):
    if request.headers.get("content-type", "").startswith("application/json"):
        payload = await request.json()
        policy_number = payload["policy_number"]
        claimant_name = payload["claimant_name"]
        contact_info = payload["contact_info"]
        incident_date = date.fromisoformat(payload["incident_date"])
        incident_description = payload["incident_description"]
        claimed_amount = Decimal(str(payload["claimed_amount"]))
        photos = []

    if not all([policy_number, claimant_name, contact_info, incident_date, incident_description, claimed_amount]):
        raise HTTPException(status_code=422, detail="Missing required claim fields")

    if len(photos) < 1 or len(photos) > 5:
        if request.headers.get("content-type", "").startswith("application/json"):
            photos = []
        else:
            raise HTTPException(status_code=422, detail='Upload between 1 and 5 photos')
    claim = await ClaimService(db).create_claim(
        policy_number,
        claimant_name,
        contact_info,
        incident_date,
        incident_description,
        claimed_amount,
        photos,
    )
    analysis_service = DamageAnalysisService(db)
    analysis_service.analyze_claim(claim, claim.policy)
    db.commit()
    db.refresh(claim)
    return claim

@router.get('/adjuster-dashboard', response_model=AdjusterDashboardResponse)
def adjuster_dashboard(db: Session = Depends(get_db)):
    claims = ClaimService(db).list_claims('submitted')
    summary = AdjusterDashboardSummary(
        pending_count=len(claims),
        approved_count=db.query(DecisionRecord).filter(DecisionRecord.decision == 'approved').count(),
        denied_count=db.query(DecisionRecord).filter(DecisionRecord.decision == 'denied').count(),
    )
    return {
        'summary': summary,
        'claims': [
            {
                'claim_id': claim.claim_id,
                'claimant_name': claim.claimant_name,
                'status': claim.status,
                'claimed_amount': str(claim.claimed_amount),
            }
            for claim in claims
        ],
    }

@router.get('/siu-dashboard')
def siu_dashboard(db: Session = Depends(get_db)):
    claims = ClaimService(db).list_claims('submitted')
    cases = []
    for claim in claims:
        if claim.analysis_result and claim.analysis_result.fraud_score and claim.analysis_result.fraud_score >= Decimal('0.65'):
            cases.append({
                'claim_id': claim.claim_id,
                'claimant_name': claim.claimant_name,
                'claimed_amount': str(claim.claimed_amount),
                'fraud_score': str(claim.analysis_result.fraud_score),
            })
    return {
        'summary': {
            'high_risk_count': len(cases),
            'under_investigation_count': 0,
            'confirmed_fraud_count': 0,
        },
        'claims': cases,
    }

@router.get('/{claim_id}', response_model=ClaimRead)
def get_claim(claim_id: str, db: Session = Depends(get_db)):
    return ClaimService(db).get_claim(claim_id)

@router.get('/{claim_id}/detail')
def get_claim_detail(claim_id: str, db: Session = Depends(get_db)):
    claim = ClaimService(db).get_claim(claim_id)
    if claim.analysis_result is None or claim.analysis_result.status == 'pending':
        DamageAnalysisService(db).analyze_claim(claim, claim.policy)
        db.commit()
        db.refresh(claim)
    return {
        'claim_id': claim.claim_id,
        'status': claim.status,
        'analysis_result': {
            'status': claim.analysis_result.status,
            'severity_label': claim.analysis_result.severity_label,
            'severity_score': str(claim.analysis_result.severity_score) if claim.analysis_result.severity_score is not None else None,
            'policy_findings': claim.analysis_result.policy_findings,
            'recommendation': claim.analysis_result.recommendation,
            'confidence_score': str(claim.analysis_result.confidence_score) if claim.analysis_result.confidence_score is not None else None,
            'explanation': claim.analysis_result.explanation,
            'fraud_score': str(claim.analysis_result.fraud_score) if claim.analysis_result.fraud_score is not None else None,
        },
    }

@router.post('/{claim_id}/decision', response_model=DecisionRead)
def submit_decision(claim_id: str, payload: DecisionRequest, db: Session = Depends(get_db)):
    claim = ClaimService(db).get_claim(claim_id)
    record = claim.decision_record or DecisionRecord(claim=claim)
    record.decision = payload.decision
    record.reasoning_note = payload.reasoning_note
    record.settlement_amount = payload.settlement_amount
    db.add(record)
    claim.status = 'decisioned'
    db.commit()
    db.refresh(record)
    return record

@router.get('', response_model=ClaimListResponse)
def list_claims(status: str | None = None, db: Session = Depends(get_db)):
    return {'claims': ClaimService(db).list_claims(status)}

@router.post('/{claim_id}/siu-action')
def siu_action(claim_id: str, payload: dict, db: Session = Depends(get_db)):
    claim = ClaimService(db).get_claim(claim_id)
    service = InvestigationService(db)
    case = service.upsert_case(claim, investigator_id=payload.get('investigator_id'), status=payload.get('status', 'under_investigation'), notes=payload.get('notes'))
    return {'claim_id': claim.claim_id, 'status': case.status}
