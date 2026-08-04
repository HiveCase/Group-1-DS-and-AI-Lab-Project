from sqlalchemy import func

from app.db.models import AnalysisResult, Claim, DecisionRecord


class AnalyticsService:
    def __init__(self, db):
        self.db = db

    def build_summary(self):
        claims = self.db.query(Claim)
        total_claims = claims.count()
        pending_count = claims.filter(Claim.status == 'submitted').count()
        approved_count = self.db.query(DecisionRecord).filter(DecisionRecord.decision == 'approved').count()
        denied_count = self.db.query(DecisionRecord).filter(DecisionRecord.decision == 'denied').count()
        avg_fraud_score = self.db.query(func.avg(AnalysisResult.fraud_score)).scalar() or 0
        severity_counts = {
            'Minor': self.db.query(AnalysisResult).filter(AnalysisResult.severity_label == 'Minor').count(),
            'Moderate': self.db.query(AnalysisResult).filter(AnalysisResult.severity_label == 'Moderate').count(),
            'Severe': self.db.query(AnalysisResult).filter(AnalysisResult.severity_label == 'Severe').count(),
        }
        coverage_flag_rate = self.db.query(AnalysisResult).filter(
            AnalysisResult.policy_findings.isnot(None)
        ).filter(AnalysisResult.policy_findings.contains({'status': 'outside_policy_limit'})).count() / max(total_claims, 1)
        return {
            'total_claims': total_claims,
            'pending_count': pending_count,
            'approved_count': approved_count,
            'denied_count': denied_count,
            'average_fraud_score': float(avg_fraud_score),
            'severity_counts': severity_counts,
            'coverage_flag_rate': float(coverage_flag_rate),
            'claims_processed_today': 0,
        }
