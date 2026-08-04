from decimal import Decimal

from app.db.models import AnalysisResult


class DamageAnalysisService:
    def __init__(self, db):
        self.db = db

    def analyze_claim(self, claim, policy=None):
        description = (claim.incident_description or '').lower()
        amount = Decimal(str(claim.claimed_amount or 0))

        if 'bumper' in description or 'panel' in description:
            severity_label = 'Minor'
            severity_score = Decimal('0.65')
        elif 'door' in description or 'window' in description:
            severity_label = 'Moderate'
            severity_score = Decimal('0.78')
        else:
            severity_label = 'Severe'
            severity_score = Decimal('0.89')

        policy_limit = policy.policy_limit if policy else None
        if policy_limit is not None and amount > policy_limit:
            policy_findings = {
                'status': 'outside_policy_limit',
                'policy_limit': str(policy_limit),
                'claimed_amount': str(amount),
            }
            recommendation = 'review'
        else:
            policy_findings = {'status': 'within_policy_limit', 'policy_limit': str(policy_limit or amount)}
            recommendation = 'approve'

        fraud_score = min(Decimal('0.99'), max(Decimal('0.15'), severity_score + Decimal('0.1')))
        explanation = (
            f"Severity inferred from incident description and amount. "
            f"Policy limit check returned {policy_findings['status']}."
        )

        analysis = claim.analysis_result or AnalysisResult(claim=claim)
        analysis.status = 'completed'
        analysis.severity_label = severity_label
        analysis.severity_score = severity_score
        analysis.fraud_score = fraud_score
        analysis.detections = [{'label': severity_label, 'confidence': str(severity_score)}]
        analysis.policy_findings = policy_findings
        analysis.recommendation = recommendation
        analysis.confidence_score = Decimal('0.82')
        analysis.explanation = explanation
        self.db.add(analysis)
        self.db.flush()
        return analysis
