from fastapi.testclient import TestClient

from app.db.database import SessionLocal
from app.main import app
from app.services.analytics_service import AnalyticsService
from app.services.claim_service import ClaimService
from app.services.investigation_service import InvestigationService
from _auth_helpers import admin_auth_headers

client = TestClient(app)
client.headers.update(admin_auth_headers(client))


def _submit_claim(policy_number='POL-001', claimed_amount=1200):
    response = client.post(
        '/claims',
        json={
            'policy_number': policy_number,
            'claimant_name': 'Ada Lovelace',
            'contact_info': 'ada@example.com',
            'incident_date': '2026-08-01',
            'incident_description': 'Rear bumper damage',
            'claimed_amount': claimed_amount,
        },
    )
    assert response.status_code == 201
    return response.json()['claim_id']


def test_decision_updates_claim_status_per_decision_type():
    claim_id = _submit_claim()

    approved = client.post(f'/claims/{claim_id}/decision', json={
        'decision': 'approved',
        'reasoning_note': 'Consistent with evidence',
        'settlement_amount': 900,
    })
    assert approved.status_code == 200
    assert client.get(f'/claims/{claim_id}').json()['status'] == 'approved'

    claim_id_2 = _submit_claim()
    denied = client.post(f'/claims/{claim_id_2}/decision', json={
        'decision': 'denied',
        'reasoning_note': 'Outside policy coverage',
    })
    assert denied.status_code == 200
    assert client.get(f'/claims/{claim_id_2}').json()['status'] == 'denied'

    claim_id_3 = _submit_claim()
    more_info = client.post(f'/claims/{claim_id_3}/decision', json={
        'decision': 'request_more_info',
        'reasoning_note': 'Need repair estimate',
    })
    assert more_info.status_code == 200
    assert client.get(f'/claims/{claim_id_3}').json()['status'] == 'under review'


def test_investigation_service_upsert_and_get_case():
    claim_id = _submit_claim()
    db = SessionLocal()
    try:
        claim = ClaimService(db).get_claim(claim_id)
        service = InvestigationService(db)

        case = service.upsert_case(claim, investigator_id='INV-1', status='under_investigation', notes='Reviewing photos')
        assert case.status == 'under_investigation'

        fetched = service.get_case(claim_id)
        assert fetched is not None
        assert fetched.investigator_id == 'INV-1'

        updated = service.upsert_case(claim, investigator_id='INV-1', status='confirmed_fraud', notes='Confirmed')
        assert updated.id == case.id
        assert updated.status == 'confirmed_fraud'
    finally:
        db.close()


def test_siu_action_endpoint_persists_investigation_status():
    claim_id = _submit_claim()
    response = client.post(f'/claims/{claim_id}/siu-action', json={
        'investigator_id': 'INV-2',
        'status': 'under_investigation',
        'notes': 'Flagged for review',
    })
    assert response.status_code == 200
    assert response.json()['status'] == 'under_investigation'


def test_analytics_summary_includes_system_status_and_throughput():
    _submit_claim()
    db = SessionLocal()
    try:
        summary = AnalyticsService(db).build_summary()
    finally:
        db.close()

    assert 'claims_processed_today' in summary
    assert summary['claims_processed_today'] >= 1
    assert 'system_status' in summary
    assert summary['system_status']['pipeline_status'] in {'operational', 'degraded'}


def test_analytics_summary_endpoint():
    response = client.get('/analytics/summary')
    assert response.status_code == 200
    assert 'summary' in response.json()
