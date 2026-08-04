# Data Model

## Overview
The MVP keeps the core domain model simple and explicit so the backend can evolve without spreading persistence concerns through the business layer.

## Entities

### Policy
Represents a seeded insurance policy that can be looked up by policy number.

**Fields**
- policy_number: string, unique
- coverage_type: string
- status: string
- effective_date: date
- policy_limit: decimal, optional
- clauses: relationship to PolicyClause

**Validation rules**
- policy_number must be present and unique
- coverage_type and status must be non-empty

### Claim
Represents a claimant-submitted insurance claim.

**Fields**
- id: integer, primary key
- claim_id: string, unique
- policy_id: integer, foreign key
- claimant_name: string
- contact_info: string
- incident_date: date
- incident_description: text
- claimed_amount: decimal
- status: string
- submitted_at: datetime
- created_at: datetime
- updated_at: datetime
- analysis_result_id: integer, optional
- decision_record_id: integer, optional

**Validation rules**
- claim_id must be generated automatically for each new claim
- status must be one of submitted, under review, approved, denied
- claimed_amount must be greater than zero

### ClaimPhoto
Represents one uploaded photo attached to a claim.

**Fields**
- id: integer, primary key
- claim_id: integer, foreign key
- file_path: string
- original_filename: string
- mime_type: string
- uploaded_at: datetime
- width: integer, optional
- height: integer, optional

**Validation rules**
- file_path must point to a stored file
- claim_id must reference an existing claim

### AnalysisResult
Represents the output created by the AI pipeline for a claim.

**Fields**
- id: integer, primary key
- claim_id: integer, foreign key
- severity_label: string
- severity_score: decimal
- detections: json
- policy_findings: json
- recommendation: string
- confidence_score: decimal
- explanation: text
- status: string
- created_at: datetime

**Validation rules**
- severity_label must be one of Minor, Moderate, Severe
- recommendation must be one of Approve, Investigate, Deny
- confidence_score must stay between 0 and 1

### DecisionRecord
Represents the adjuster’s final decision for a claim.

**Fields**
- id: integer, primary key
- claim_id: integer, foreign key
- decision: string
- reasoning_note: text
- settlement_amount: decimal, optional
- decided_at: datetime

**Validation rules**
- decision must be one of Approve, Deny, Request More Info
- reasoning_note must be present

### PolicyClause
Represents a clause entry used for semantic retrieval.

**Fields**
- id: integer, primary key
- clause_id: string
- text: text
- metadata: json
- embedding_id: string, optional

**Validation rules**
- clause_id and text must be present

## Relationships
- One Policy has many Claims.
- One Claim has many ClaimPhotos.
- One Claim has zero or one AnalysisResult.
- One Claim has zero or one DecisionRecord.
- One Policy has many PolicyClauses.

## State transitions
- Claim status moves from submitted to under review when analysis is being processed or an adjuster opens it.
- Claim status moves to approved, denied, or remains under review after a decision is submitted.
