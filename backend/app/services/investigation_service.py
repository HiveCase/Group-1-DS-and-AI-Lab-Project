from app.db.models import InvestigationCase


class InvestigationService:
    def __init__(self, db):
        self.db = db

    def list_cases(self):
        return self.db.query(InvestigationCase).order_by(InvestigationCase.created_at.desc()).all()

    def get_case(self, claim_id: str):
        return self.db.query(InvestigationCase).join(InvestigationCase.claim).filter(InvestigationCase.claim.has(claim_id=claim_id)).first()

    def upsert_case(self, claim, investigator_id: str | None = None, status: str = 'under_investigation', notes: str | None = None):
        case = self.db.query(InvestigationCase).filter(InvestigationCase.claim_id == claim.id).first()
        if not case:
            case = InvestigationCase(claim_id=claim.id)
            self.db.add(case)
        case.investigator_id = investigator_id
        case.status = status
        case.notes = notes
        self.db.commit()
        self.db.refresh(case)
        return case
