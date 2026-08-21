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
    # detections (carrying per-detection saliency, when computed) must be
    # exposed to the frontend, not just persisted internally.
    assert 'detections' in payload['analysis_result']

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


def test_annotated_photo_endpoint_returns_image_with_detections(tmp_path, monkeypatch):
    from PIL import Image
    from app.routes import claims as claims_route

    # A plain white square genuinely has zero real detections; force a
    # deterministic one so this test exercises the annotation/persistence
    # path rather than depending on what the real model happens to find.
    monkeypatch.setattr(
        claims_route._orchestrator.damage_service,
        "detect_from_path",
        lambda image_path: [{"class_name": "dent", "bbox": [10, 10, 100, 100], "mask_polygon": None, "confidence": 0.8}],
    )

    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (640, 640), color="white").save(image_path)

    with open(image_path, "rb") as fh:
        response = client.post(
            '/claims',
            data={
                'policy_number': 'POL-001',
                'claimant_name': 'Ada Lovelace',
                'contact_info': 'ada@example.com',
                'incident_date': '2026-08-01',
                'incident_description': 'Rear bumper damage',
                'claimed_amount': '1200',
            },
            files={'photos': ('sample.jpg', fh, 'image/jpeg')},
        )
    assert response.status_code == 201
    claim_id = response.json()['claim_id']

    photo_response = client.get(f'/claims/{claim_id}/annotated-photo')
    assert photo_response.status_code == 200
    assert photo_response.headers['content-type'] == 'image/jpeg'
    assert len(photo_response.content) > 0

    claim_response = client.get(f'/claims/{claim_id}')
    assert claim_response.status_code == 200
    photos = claim_response.json()['photos']
    assert photos[0]['annotated_path']


def test_claim_submission_syncs_retrieved_clauses_into_policy_clauses_table():
    """policy_clauses should end up populated with the real clauses a
    claim's RAG retrieval actually surfaced -- not just the two generic
    seed rows -- once analysis for a claim with a recognizable damage
    class in its description completes. Each row must be traceable back to
    the specific claim that cited it, and a second, different claim citing
    the same clause must get its own row rather than being dropped --
    clause_id is cited by many claims over time, so it can't be the sole
    uniqueness key."""
    from app.db.database import SessionLocal
    from app.db.models import Claim, PolicyClause

    payload = {
        'policy_number': 'POL-001',
        'claimant_name': 'Ada Lovelace',
        'contact_info': 'ada@example.com',
        'incident_date': '2026-08-09',
        'incident_description': 'There is a dent and a scratch on the rear panel',
        'claimed_amount': 500,
    }
    response_1 = client.post('/claims', json=payload)
    assert response_1.status_code == 201
    claim_id_1 = response_1.json()['claim_id']

    response_2 = client.post('/claims', json=payload)
    assert response_2.status_code == 201
    claim_id_2 = response_2.json()['claim_id']

    db = SessionLocal()
    try:
        real_clauses = db.query(PolicyClause).filter(PolicyClause.policy_id.isnot(None)).all()
        assert real_clauses, "expected at least one RAG-retrieved clause to be synced"
        assert any(c.clause_metadata and c.clause_metadata.get("damage_class") for c in real_clauses)

        claim_1_pk = db.query(Claim).filter(Claim.claim_id == claim_id_1).one().id
        claim_2_pk = db.query(Claim).filter(Claim.claim_id == claim_id_2).one().id
        clauses_1 = {c.clause_id for c in real_clauses if c.claim_id == claim_1_pk}
        clauses_2 = {c.clause_id for c in real_clauses if c.claim_id == claim_2_pk}
        assert clauses_1, "expected claim 1's citations to carry its own claim_id"
        assert clauses_2, "expected claim 2's citations to carry its own claim_id"
        assert clauses_1 & clauses_2, "two claims with identical facts should cite overlapping clauses"
    finally:
        db.close()


def test_annotated_photo_endpoint_404s_without_photos():
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
    claim_id = response.json()['claim_id']

    photo_response = client.get(f'/claims/{claim_id}/annotated-photo')
    assert photo_response.status_code == 404
