from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from app.db.models import Policy, PolicyClause


SEED_POLICIES = [
    {
        "policy_number": "POL-001",
        "policy_holder_name": "Ada Lovelace",
        "coverage_type": "Comprehensive",
        "policy_effective_date": date(2025, 1, 1),
        "policy_expiry_date": date(2028, 1, 1),
        "policy_limit": Decimal("5000.00"),
        "vehicle_purchase_date": date(2024, 3, 1),
        "vehicle_registration_no": "A1",
    },
    {
        "policy_number": "POL-002",
        "policy_holder_name": "Grace Hopper",
        "coverage_type": "Collision",
        "policy_effective_date": date(2025, 6, 1),
        "policy_expiry_date": date(2028, 6, 1),
        "policy_limit": Decimal("3500.00"),
        "vehicle_purchase_date": date(2023, 1, 15),
        "vehicle_registration_no": "B2",
    },
    {
        "policy_number": "POL-003",
        "policy_holder_name": "Tanmay Mal",
        "coverage_type": "Comprehensive",
        "policy_effective_date": date(2025, 3, 1),
        "policy_expiry_date": date(2026, 3, 1),
        "policy_limit": Decimal("4000.00"),
        "vehicle_purchase_date": date(2020, 5, 1),
        "vehicle_registration_no": "C3",
    },
    {
        "policy_number": "POL-004",
        "policy_holder_name": "Alan Turing",
        "coverage_type": "Comprehensive Premium",
        "policy_effective_date": date(2025, 2, 1),
        "policy_expiry_date": date(2025, 9, 1),
        "policy_limit": Decimal("6000.00"),
        "vehicle_purchase_date": date(2024, 11, 1),
        "vehicle_registration_no": "D4",
    },
    {
        "policy_number": "POL-005",
        "policy_holder_name": "Katherine Johnson",
        "coverage_type": "Third Party",
        "policy_effective_date": date(2025, 4, 1),
        "policy_expiry_date": date(2028, 4, 1),
        "policy_limit": Decimal("2500.00"),
        "vehicle_purchase_date": date(2026, 2, 1),
        "vehicle_registration_no": "E5",
    },
    {
        "policy_number": "POL-006",
        "policy_holder_name": "Dinesh Pal",
        "coverage_type": "Comprehensive Premium",
        "policy_effective_date": date(2025, 1, 1),
        "policy_expiry_date": date(2027, 4, 1),
        "policy_limit": Decimal("9500.00"),
        "vehicle_purchase_date": date(2022, 6, 1),
        "vehicle_registration_no": "F6",
    },
    {
        "policy_number": "POL-007",
        "policy_holder_name": "Devi Prasad",
        "coverage_type": "Third Party",
        "policy_effective_date": date(2026, 4, 1),
        "policy_expiry_date": date(2028, 1, 1),
        "policy_limit": Decimal("12500.00"),
        "vehicle_purchase_date": date(2025, 9, 1),
        "vehicle_registration_no": "G7",
    },
]

# Fields added after the initial seed; existing policy rows (already present
# in a deployed DB) are backfilled with these in seed_defaults() rather than
# being skipped, since seed_defaults() only *inserts* missing policy_numbers.
# "status" is deliberately absent: it's now Policy.status, a derived
# property computed from policy_effective_date/policy_expiry_date, not a
# stored column -- there is nothing to backfill.
_BACKFILL_FIELDS = ("policy_holder_name", "policy_expiry_date", "vehicle_purchase_date", "vehicle_registration_no")

# Maps each seeded policy to the synthetic policy-wording PDF that
# PolicyClauseService ingests for real clause retrieval. Files live under
# backend/app/rag_scripts/data/policy_pdfs/synthetic/.
POLICY_PDF_MAP = {
    "POL-001": "policy_1_bharat_suraksha.pdf",
    "POL-002": "policy_2_safedrive_assurance.pdf",
    "POL-003": "policy_3_quickclaim_general.pdf",
    "POL-004": "policy_4_autoguard_premium.pdf",
    "POL-005": "policy_5_valuemotor.pdf",
    "POL-006": "policy_4_autoguard_premium.pdf",
    "POL-007": "policy_5_valuemotor.pdf",
}

SEED_CLAUSES = [
    ("CL-AUTO-001", "Comprehensive coverage applies to accidental body damage up to the policy limit."),
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
