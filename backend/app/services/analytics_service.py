from datetime import datetime, timedelta

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
            'claims_processed_today': self._claims_processed_today(),
            'system_status': self._system_status(),
        }

    def _claims_processed_today(self) -> int:
        # completed_at/created_at are stored via datetime.utcnow(), so the
        # "today" boundary must also be computed in UTC -- comparing against
        # a local-timezone date would miss rows for hours around local
        # midnight in any timezone ahead of UTC.
        now_utc = datetime.utcnow()
        today_start = datetime(now_utc.year, now_utc.month, now_utc.day)
        return self.db.query(AnalysisResult).filter(
            AnalysisResult.status == 'completed',
            AnalysisResult.completed_at.isnot(None),
            AnalysisResult.completed_at >= today_start,
        ).count()

    def _system_status(self) -> dict:
        completed = self.db.query(AnalysisResult).filter(
            AnalysisResult.status == 'completed',
            AnalysisResult.completed_at.isnot(None),
        ).order_by(AnalysisResult.completed_at.desc()).limit(50).all()

        durations = [
            (result.completed_at - result.created_at).total_seconds()
            for result in completed
            if result.completed_at and result.created_at
        ]
        avg_analysis_time_seconds = round(sum(durations) / len(durations), 2) if durations else None

        recent_failures = self.db.query(AnalysisResult).filter(
            AnalysisResult.status == 'failed',
            AnalysisResult.created_at >= datetime.utcnow() - timedelta(hours=1),
        ).count()

        pending_count = self.db.query(AnalysisResult).filter(AnalysisResult.status == 'pending').count()

        return {
            'pipeline_status': 'degraded' if recent_failures > 0 else 'operational',
            'avg_analysis_time_seconds': avg_analysis_time_seconds,
            'claims_awaiting_analysis': pending_count,
            'recent_failure_count': recent_failures,
        }
