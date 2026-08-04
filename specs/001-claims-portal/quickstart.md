# Quickstart

## Prerequisites
- Docker Compose and Docker Engine installed locally
- Python 3.11+
- Node.js 20+
- Local access to the existing YOLO model weights and policy clause dataset

## Setup
1. Create environment files for the backend and frontend using the project defaults.
2. Place the existing YOLO weights under backend/models and the clause dataset under backend/data.
3. Ensure the uploads directory exists and is writable.

## Run locally
1. Build and start the services with Docker Compose.
2. Open the frontend entry point in the browser and use the claimant flow to submit a claim.
3. Open the adjuster view to review the AI analysis and make a decision.

## Validation scenarios
- Submit a claim with one or more photos and confirm the claim ID and submitted status appear.
- Wait for the background analysis to complete and verify that the adjuster view shows severity, clause findings, recommendation, and explanation.
- Approve or deny the claim and confirm the status updates with a new decision record.

## Test commands
- Backend tests: pytest
- Frontend tests: npm run test
- Full local validation: docker compose up --build
