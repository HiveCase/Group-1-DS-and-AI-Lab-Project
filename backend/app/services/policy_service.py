from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from app.db.models import Policy, PolicyClause


SEED_POLICIES = [
    {
        "policy_number": "POL-001",
        "coverage_type": "Comprehensive",
        "status": "active",
        "effective_date": date(2025, 1, 1),
        "policy_limit": Decimal("5000.00"),
    },
    {
        "policy_number": "POL-002",
        "coverage_type": "Collision",
        "status": "active",
        "effective_date": date(2025, 6, 1),
        "policy_limit": Decimal("3500.00"),
    },
]

SEED_CLAUSES = [
    ("CL-AUTO-001", "Comprehensive coverage applies to accidental body damage up to the policy limit."),
    ("CL-AUTO-002", "Settlement for cosmetic bumper or panel damage is capped by the active policy limit."),
]


class PolicyService:
    def __init__(self, db: Session):
        self.db = db

    def seed_defaults(self) -> None:
        for item in SEED_POLICIES:
            existing = self.get_by_number(item["policy_number"])
            if not existing:
                self.db.add(Policy(**item))
        self.db.commit()

        for clause_id, text in SEED_CLAUSES:
            if not self.db.query(PolicyClause).filter(PolicyClause.clause_id == clause_id).first():
                self.db.add(PolicyClause(clause_id=clause_id, text=text, clause_metadata={"source": "seed"}))
        self.db.commit()

    def get_by_number(self, policy_number: str) -> Policy | None:
        return self.db.query(Policy).filter(Policy.policy_number == policy_number).first()
