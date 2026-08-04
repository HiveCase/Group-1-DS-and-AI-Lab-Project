# Feature Specification: AI-Assisted Claim Portal

**Feature Branch**: `001-claims-portal`

**Created**: 2026-08-04

**Status**: Draft

**Input**: User description: "Build a minimal AI-assisted car damage insurance claim portal with Claimant and Adjuster portals, and expand it to include SIU and Supervisor portals as part of the same cohesive experience."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - File a claim as a claimant (Priority: P1)
A claimant with a valid policy number can log in through a simple policy lookup, review basic policy details, and submit a new claim with incident information and one to five vehicle photos.

**Why this priority**: This is the core value of the system because it creates the first claim record and starts the review workflow.

**Independent Test**: A claimant can submit a complete claim and receive a claim ID without any adjuster action.

**Acceptance Scenarios**:

1. **Given** the claimant enters a valid policy number, **When** the lookup succeeds, **Then** the system shows the policy details and enables claim submission.
2. **Given** the claimant submits a complete claim form with one to five photos, **When** the submission is accepted, **Then** the claim is saved with status "submitted" and the claimant sees a confirmation with a unique claim ID.
3. **Given** the claimant enters an invalid policy number, **When** they try to continue, **Then** the system blocks access and shows a clear error.

---

### User Story 2 - Review and analyze a submitted claim as an adjuster (Priority: P1)
An adjuster can open a submitted claim, review the details and uploaded photos, and see AI-generated severity, coverage, and recommendation information that supports a final decision.

**Why this priority**: This is the main operational workflow for the adjuster and the primary value of the AI-assisted experience.

**Independent Test**: An adjuster can open a claim and review all required analysis details without needing to process any other claim.

**Acceptance Scenarios**:

1. **Given** a claim is in submitted status, **When** the adjuster opens it, **Then** the system displays full claim details, photos, and the available AI analysis panel.
2. **Given** the AI analysis has completed, **When** the adjuster reviews the claim, **Then** the system shows severity information, relevant coverage notes, and an overall recommendation with a confidence score.
3. **Given** the claim has no ready analysis yet, **When** the adjuster opens it, **Then** the system shows that analysis is still in progress and does not block the review workflow.

---

### User Story 3 - Decide and update claim outcome (Priority: P2)
An adjuster can make a final decision on a claim, record a reasoning note, and update the claim status with a timestamp so the claimant can later see the current outcome.

**Why this priority**: This completes the workflow and provides a clear decision trail for both roles.

**Independent Test**: An adjuster can approve, deny, or request more information for a claim and store the decision.

**Acceptance Scenarios**:

1. **Given** an adjuster has reviewed a claim, **When** they approve it and enter a settlement amount, **Then** the claim status updates to approved and the decision is recorded.
2. **Given** an adjuster chooses to deny or request more information, **When** they submit the decision, **Then** the claim status updates accordingly and the reasoning note is stored.
3. **Given** a claimant later looks up the claim by its ID, **When** the status has changed, **Then** the claimant sees the current status without seeing the underlying AI details.

### User Story 4 - Review escalations as an SIU analyst (Priority: P2)
An SIU analyst can enter a dedicated portal, review claims flagged for investigation, and see the claim history, decision trail, and AI-generated risk context needed to decide whether the case should be escalated or closed.

**Why this priority**: This adds a focused investigation workflow for suspicious or high-risk claims without changing the core claimant and adjuster experience.

**Independent Test**: An SIU analyst can open an escalated claim and review the investigation context in a single view.

**Acceptance Scenarios**:

1. **Given** a claim has been tagged for special review, **When** the SIU analyst opens the case, **Then** the system shows the full claim timeline, decisions, and supporting evidence in one place.
2. **Given** the SIU analyst reviews the case details, **When** they choose to open a formal investigation or clear the flag, **Then** the system records the action with a timestamp and investigator ID.
3. **Given** a claim is not flagged for SIU review, **When** the SIU analyst opens the portal, **Then** the system does not surface it in the investigation queue.
4. **Given** a claim is opened from the SIU dashboard, **When** the analyst views it, **Then** the system reuses the same AI analysis panel as the adjuster view, including the severity output, coverage check, and fraud factors that drove the score.

### User Story 5 - Monitor portfolio trends as a supervisor (Priority: P2)
A supervisor can enter a management portal, view high-level claim and workflow metrics, and monitor the health of the claim operation across claimant, adjuster, and SIU activities.

**Why this priority**: This gives leadership visibility into outstanding work and operational trends without interrupting day-to-day claim handling.

**Independent Test**: A supervisor can open the management view and see summary metrics and recent activity without needing to review each claim individually.

**Acceptance Scenarios**:

1. **Given** the supervisor opens the portal, **When** the dashboard loads, **Then** the system displays summary counts, status distribution, and recent claim activity.
2. **Given** the supervisor reviews the metrics, **When** they inspect a trend or exception, **Then** the system provides enough context to understand the underlying operational issue.
3. **Given** no claims are available for a selected view, **When** the supervisor loads the page, **Then** the system shows an empty state that explains the absence of data.
4. **Given** the supervisor opens the analytics view, **When** they review the AI-specific metrics, **Then** the system shows the average detected damage severity breakdown, the frequency of coverage-limit or outside-coverage flags, and the current AI pipeline operational status.

### Edge Cases

- What happens if a claimant uploads more than five photos?
- How does the system handle a claim lookup when no matching claim ID exists?
- What happens if AI analysis is still running when the adjuster opens the claim?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST support claimant login through a simple policy number lookup against seeded policy data.
- **FR-002**: The system MUST display basic policy details to the claimant after a successful lookup.
- **FR-003**: The system MUST allow a claimant to submit a new claim with required incident details, claimed amount, and one to five photos.
- **FR-004**: The system MUST assign each claim a unique claim ID and store it with an initial status of "submitted".
- **FR-005**: The system MUST allow a claimant to look up a claim by ID and view its current status.
- **FR-006**: The system MUST provide an adjuster dashboard that lists claims in submitted or under review status and shows summary counts.
- **FR-007**: The system MUST present AI-assisted analysis for each claim, including severity, coverage guidance, and an overall recommendation.
- **FR-008**: The system MUST allow an adjuster to make a final decision of approve, deny, or request more information.
- **FR-009**: The system MUST record the final decision, reasoning note, settlement amount when applicable, and a timestamp.
- **FR-010**: The system MUST trigger AI analysis after claim submission without blocking the claimant confirmation flow.
- **FR-011**: The system MUST keep AI analysis details visible to the adjuster while not exposing them to the claimant.
- **FR-012**: The system MUST support a claim status flow that includes submitted, under review, approved, and denied states.
- **FR-013**: The system MUST provide an SIU portal that allows analysts to review flagged claims, inspect the decision trail, and record escalation or closure actions.
- **FR-014**: The SIU dashboard MUST be auto-filtered to claims above the fraud-score threshold, and each claim card MUST prominently display the claim ID, claimant, claim type, amount, and fraud score.
- **FR-015**: The SIU claim detail view MUST reuse the same AI analysis panel as the adjuster workflow, including severity output, policy-clause coverage findings, and the fraud factors that drove the score.
- **FR-016**: The system MUST provide a Supervisor portal that presents portfolio-level summaries, claim-status distribution, recent operational activity, and AI-specific metrics for severity and policy-coverage outcomes.
- **FR-017**: The Supervisor view MUST include a read-only system status section showing AI pipeline health and basic throughput metrics such as analysis time and claims processed today.
- **FR-018**: The frontend MUST present the Claimant, Adjuster, SIU, and Supervisor experiences through a professional, cohesive design system with shared layout, visual standards, and reusable component styles across all portals.
- **FR-019**: The frontend MUST include a shared AppShell layout and a portal-selection landing page before portal-specific views are styled in detail.

### Key Entities *(include if feature involves data)*

- **Policy**: Represents the insured contract that can be looked up by policy number and includes coverage type, status, and effective date.
- **Claim**: Represents a submitted insurance claim and contains the claimant details, incident information, amount claimed, status, and associated analysis results.
- **Claim Photo**: Represents one uploaded image attached to a claim and is used as input for the analysis workflow.
- **AI Analysis Result**: Represents the severity, coverage, recommendation, and confidence information produced for a claim.
- **Decision Record**: Represents the adjuster’s final decision, reasoning note, settlement amount, and decision timestamp.
- **Investigation Case**: Represents an SIU follow-up action linked to a claim and captures the reason for review, analyst notes, and the final escalation or closure outcome.
- **Portal View**: Represents a shared user-facing experience across the Claimant, Adjuster, SIU, and Supervisor portals and supports the shared layout and design system requirements.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A claimant can successfully submit a claim and receive a claim ID within five minutes of starting the process.
- **SC-002**: AI-assisted analysis is available to the adjuster within roughly one minute for typical local test claims.
- **SC-003**: At least 90% of submitted claims can be reviewed and resolved by an adjuster without leaving the portal.
- **SC-004**: Claimants can look up their claim and understand its current status without seeing the underlying AI analysis details.
- **SC-005**: Adjusters can review a claim and reach a decision in a single workflow without needing a separate tool.
- **SC-006**: SIU analysts can identify and act on high-risk claims from a focused dashboard within a short review session without needing a separate investigation tool.
- **SC-007**: Supervisors can understand current portfolio health and the state of the AI pipeline from a single read-only dashboard without opening individual claim records.

## Assumptions

- A small set of seeded policy records is available for claimant lookup during the MVP.
- The MVP focuses on common vehicle damage claim scenarios and does not need full production-grade fraud or payment workflows.
- The local environment includes the existing AI assets needed for severity analysis and policy clause retrieval.
- The SIU and Supervisor experiences are lightweight management views for the MVP rather than full investigative or analytics platforms.
- Mobile support is not required for the first release.
