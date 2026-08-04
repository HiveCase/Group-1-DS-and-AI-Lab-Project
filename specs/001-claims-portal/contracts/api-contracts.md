# API Contracts

## Policy lookup

### POST /policies/lookup
Request body:
- policy_number: string

Success response:
- 200 OK
- policy: object with policy_number, coverage_type, status, effective_date, policy_limit

Error response:
- 404 Not Found when the policy number is unknown

## Create claim

### POST /claims
Request body:
- policy_number: string
- claimant_name: string
- contact_info: string
- incident_date: string (ISO date)
- incident_description: string
- claimed_amount: number
- photos: array of file uploads

Success response:
- 201 Created
- claim_id: string
- status: submitted
- message: string

## Get claim by ID

### GET /claims/{claim_id}
Success response:
- 200 OK
- claim: object with core claim details and current status

## List claims for dashboard

### GET /claims?status=submitted,under_review
Success response:
- 200 OK
- claims: array of dashboard summary objects

## Get analysis result

### GET /claims/{claim_id}/analysis
Success response:
- 200 OK
- analysis: object with severity_label, severity_score, recommendation, confidence_score, explanation, detections, policy_findings

## Submit adjuster decision

### POST /claims/{claim_id}/decision
Request body:
- decision: string
- reasoning_note: string
- settlement_amount: number, optional

Success response:
- 200 OK
- claim_status: updated status
- decision_record: object
