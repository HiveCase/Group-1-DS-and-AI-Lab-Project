from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_claim_analysis_and_adjuster_workflow():
    response = client.post(
        '/claims',
        json={
            'policy_number': 'POL-001',
            'claimant_name': 'Ada Lovelace',
            'contact_info': 'ada@example.com',
            'incident_date': '2026-08-01',
            'incident_description': 'Rear bumper damage',
            'claimed_amount': 1200,
        },
    )
    assert response.status_code == 201

    claim_id = response.json()['claim_id']

    dashboard = client.get('/claims/adjuster-dashboard')
    assert dashboard.status_code == 200
    assert dashboard.json()['summary']['pending_count'] >= 1

    detail = client.get(f'/claims/{claim_id}/detail')
    assert detail.status_code == 200
    payload = detail.json()
    assert payload['analysis_result']['status'] == 'completed'
    assert payload['analysis_result']['severity_label'] in {'Minor', 'Moderate', 'Severe'}

    decision = client.post(
        f'/claims/{claim_id}/decision',
        json={
            'decision': 'approved',
            'reasoning_note': 'Looks consistent',
            'settlement_amount': 1000,
        },
    )
    assert decision.status_code == 200
    assert decision.json()['decision'] == 'approved'
