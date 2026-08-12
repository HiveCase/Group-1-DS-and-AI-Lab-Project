from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from app.db.models import Policy, PolicyClause


SEED_POLICIES = [
    {
        "policy_number": "POL-001",
        "policy_holder_name": "Ada Lovelace",
        "coverage_type": "Comprehensive",
        "status": "active",
        "effective_date": date(2025, 1, 1),
        "expiry_date": date(2028, 1, 1),
        "policy_limit": Decimal("5000.00"),
    },
    {
        "policy_number": "POL-002",
        "policy_holder_name": "Grace Hopper",
        "coverage_type": "Collision",
        "status": "active",
        "effective_date": date(2025, 6, 1),
        "expiry_date": date(2028, 6, 1),
        "policy_limit": Decimal("3500.00"),
    },
    {
        "policy_number": "POL-003",
        "policy_holder_name": "Tanmay Mal",
        "coverage_type": "Comprehensive",
        "status": "active",
        "effective_date": date(2025, 3, 1),
        "expiry_date": date(2028, 3, 1),
        "policy_limit": Decimal("4000.00"),
    },
    {
        "policy_number": "POL-004",
        "policy_holder_name": "Alan Turing",
        "coverage_type": "Comprehensive Premium",
        "status": "active",
        "effective_date": date(2025, 2, 1),
        "expiry_date": date(2028, 2, 1),
        "policy_limit": Decimal("6000.00"),
    },
    {
        "policy_number": "POL-005",
        "policy_holder_name": "Katherine Johnson",
        "coverage_type": "Third Party",
        "status": "active",
        "effective_date": date(2025, 4, 1),
        "expiry_date": date(2028, 4, 1),
        "policy_limit": Decimal("2500.00"),
    },
]

# Fields added after the initial seed; existing policy rows (already present
# in a deployed DB) are backfilled with these in seed_defaults() rather than
# being skipped, since seed_defaults() only *inserts* missing policy_numbers.
_BACKFILL_FIELDS = ("policy_holder_name", "expiry_date")

# Maps each seeded policy to the synthetic policy-wording PDF that
# PolicyClauseService ingests for real clause retrieval. Files live under
# backend/app/rag_scripts/data/policy_pdfs/synthetic/.
POLICY_PDF_MAP = {
    "POL-001": "policy_1_bharat_suraksha.pdf",
    "POL-002": "policy_2_safedrive_assurance.pdf",
    "POL-003": "policy_3_quickclaim_general.pdf",
    "POL-004": "policy_4_autoguard_premium.pdf",
    "POL-005": "policy_5_valuemotor.pdf",
}

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
            else:
                for field in _BACKFILL_FIELDS:
                    if getattr(existing, field, None) is None and field in item:
                        setattr(existing, field, item[field])
        self.db.commit()

        for clause_id, text in SEED_CLAUSES:
            if not self.db.query(PolicyClause).filter(PolicyClause.clause_id == clause_id).first():
                self.db.add(PolicyClause(clause_id=clause_id, text=text, clause_metadata={"source": "seed"}))
        self.db.commit()

    def total_claimed_for_policy(self, policy: Policy, exclude_claim_id: str | None = None) -> Decimal:
        """Sum of claimed_amount across the policy's other non-denied claims --
        computed live from the claims table rather than stored, so it can
        never drift out of sync as claims are added/decisioned."""
        total = Decimal("0")
        for claim in policy.claims or []:
            if claim.claim_id == exclude_claim_id:
                continue
            if claim.status == "denied":
                continue
            total += claim.claimed_amount or Decimal("0")
        return total

    def get_by_number(self, policy_number: str) -> Policy | None:
        return self.db.query(Policy).filter(Policy.policy_number == policy_number).first()
