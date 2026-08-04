from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200


def test_policy_lookup():
    response = client.post('/policies/lookup', json={'policy_number': 'POL-001'})
    assert response.status_code == 200
    assert response.json()['policy_number'] == 'POL-001'


def test_create_claim():
    response = client.post('/claims', json={
        'policy_number': 'POL-001',
        'claimant_name': 'Ada Lovelace',
        'contact_info': 'ada@example.com',
        'incident_date': '2026-08-01',
        'incident_description': 'Rear bumper damage',
        'claimed_amount': 1200,
    })
    assert response.status_code == 201
    assert response.json()['status'] == 'submitted'
