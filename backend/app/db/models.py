from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Date, DateTime, ForeignKey, Index, Numeric, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Policy(Base):
    __tablename__ = "policies"

    id: Mapped[int] = mapped_column(primary_key=True)
    policy_number: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    coverage_type: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(50), index=True)
    effective_date: Mapped[date] = mapped_column(Date)
    policy_limit: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)

    claims: Mapped[list["Claim"]] = relationship(back_populates="policy")
    clauses: Mapped[list["PolicyClause"]] = relationship(back_populates="policy")


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    policy_id: Mapped[int] = mapped_column(ForeignKey("policies.id"))
    claimant_name: Mapped[str] = mapped_column(String(120))
    contact_info: Mapped[str] = mapped_column(String(160))
    incident_date: Mapped[date] = mapped_column(Date)
    incident_description: Mapped[str] = mapped_column(Text)
    claimed_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    status: Mapped[str] = mapped_column(String(50), index=True, default="submitted")
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    policy: Mapped[Policy] = relationship(back_populates="claims")
    photos: Mapped[list["ClaimPhoto"]] = relationship(back_populates="claim", cascade="all, delete-orphan")
    analysis_result: Mapped["AnalysisResult | None"] = relationship(back_populates="claim", uselist=False)
    decision_record: Mapped["DecisionRecord | None"] = relationship(back_populates="claim", uselist=False)


class ClaimPhoto(Base):
    __tablename__ = "claim_photos"

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), index=True)
    file_path: Mapped[str] = mapped_column(String(500))
    original_filename: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(100))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    annotated_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    claim: Mapped[Claim] = relationship(back_populates="photos")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), unique=True, index=True)
    severity_label: Mapped[str | None] = mapped_column(String(50), nullable=True)
    severity_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    fraud_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    detections: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    policy_findings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    recommendation: Mapped[str | None] = mapped_column(String(50), nullable=True)
    confidence_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    needs_human_review: Mapped[bool | None] = mapped_column(default=False, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    claim: Mapped[Claim] = relationship(back_populates="analysis_result")


class DecisionRecord(Base):
    __tablename__ = "decision_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), unique=True, index=True)
    decision: Mapped[str] = mapped_column(String(50))
    reasoning_note: Mapped[str] = mapped_column(Text)
    settlement_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    claim: Mapped[Claim] = relationship(back_populates="decision_record")


class InvestigationCase(Base):
    __tablename__ = "investigation_cases"

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), unique=True, index=True)
    investigator_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="under_investigation")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    claim: Mapped[Claim] = relationship()


class PolicyClause(Base):
    __tablename__ = "policy_clauses"

    id: Mapped[int] = mapped_column(primary_key=True)
    policy_id: Mapped[int | None] = mapped_column(ForeignKey("policies.id"), nullable=True)
    clause_id: Mapped[str] = mapped_column(String(80), unique=True)
    text: Mapped[str] = mapped_column(Text)
    clause_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    embedding_id: Mapped[str | None] = mapped_column(String(120), nullable=True)

    policy: Mapped[Policy | None] = relationship(back_populates="clauses")


Index("ix_claim_policy_status", Claim.policy_id, Claim.status)
