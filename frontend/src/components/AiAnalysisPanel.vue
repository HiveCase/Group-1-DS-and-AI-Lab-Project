<template>
  <div class="card ai-analysis-panel">
    <h2 class="section-title">AI analysis</h2>

    <div v-if="!analysis || analysis.status === 'pending'" class="inline-group">
      <i class="pi pi-spin pi-spinner" />
      <span class="muted">Analysis is still in progress. This panel will update automatically.</span>
    </div>

    <div v-else-if="analysis.status === 'failed'" class="inline-group">
      <Tag value="Analysis failed" severity="danger" />
      <span class="muted">The AI pipeline could not complete. A human review is required.</span>
    </div>

    <template v-else>
      <div v-if="isFallbackReport" class="nested-card status-warning">
        <strong><i class="pi pi-exclamation-triangle" /> Default report -- not AI-assessed</strong>
        <p class="muted">
          The report-synthesis model could not be reached (e.g. rate-limited or misconfigured), so the
          recommendation, confidence score, and next steps below are a fixed fallback template, not a
          genuine model assessment of this claim. The damage table and policy findings above are still
          real detection/retrieval output.
          <span v-if="fallbackReason">Reason: {{ fallbackReason }}</span>
        </p>
      </div>

      <div v-if="isRecommendationOverridden" class="nested-card status-warning">
        <strong><i class="pi pi-exclamation-triangle" /> Recommendation overridden</strong>
        <p class="muted">{{ recommendationOverrideReason }}</p>
      </div>

      <div v-if="analysis.needs_human_review" class="nested-card status-warning">
        <strong><i class="pi pi-exclamation-triangle" /> Needs human review</strong>
        <p class="muted">{{ escalationReason }}</p>
      </div>

      <div class="inline-group">
        <Tag :value="`Severity: ${analysis.severity_label || 'Unknown'}`" :severity="severityTagColor(analysis.severity_label)" />
        <Tag :value="recommendationTagText" :severity="recommendationTagColor(analysis.recommendation)" />
        <Tag v-if="analysis.confidence_score != null" :value="`Confidence: ${percent(analysis.confidence_score)}`" severity="secondary" />
        <Tag v-if="analysis.fraud_score != null" :value="`Fraud score: ${percent(analysis.fraud_score)}`" :severity="fraudTagColor(analysis.fraud_score)" />
        <Tag v-if="isFallbackReport" value="Default report" severity="warn" />
      </div>

      <div v-if="fraudReason" class="nested-card">
        <strong>Fraud factors</strong>
        <ul v-if="fraudSignals.length" class="clause-list">
          <li v-for="(signal, index) in fraudSignals" :key="index">{{ signal }}</li>
        </ul>
        <p v-else class="muted">{{ fraudReason }}</p>
      </div>

      <div v-if="damageTable.length" class="nested-card">
        <strong>Damage table</strong>
        <DataTable :value="damageTable" size="small">
          <Column field="class" header="Class" />
          <Column field="severity" header="Severity" />
          <Column header="Confidence">
            <template #body="{ data }">{{ percent(data.confidence) }}</template>
          </Column>
        </DataTable>
      </div>

      <div v-if="clauseFindings.length" class="nested-card">
        <strong>Policy coverage findings</strong>
        <ul class="clause-list">
          <li v-for="finding in clauseFindings" :key="finding.clause_id" class="clause-item">
            <div class="clause-meta">{{ finding.clause_id }} &middot; {{ finding.clause_type || 'clause' }} &middot; {{ finding.source_citation }}</div>
            <div>{{ finding.text }}</div>
          </li>
        </ul>
      </div>

      <div v-if="nextSteps.length" class="nested-card">
        <strong>Next steps</strong>
        <ul>
          <li v-for="(step, index) in nextSteps" :key="index">{{ step }}</li>
        </ul>
      </div>

      <p v-if="analysis.explanation" class="muted">{{ analysis.explanation }}</p>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import Tag from 'primevue/tag';

const props = defineProps({
  analysis: {
    type: Object,
    default: null,
  },
});

const report = computed(() => props.analysis?.report_json || {});
const damageTable = computed(() => report.value.damage_table || []);
const nextSteps = computed(() => report.value.next_steps || []);
const clauseFindings = computed(() => (props.analysis?.policy_findings || []).filter((f) => f.clause_id !== 'POLICY-LIMIT-CHECK'));
const fraudReason = computed(() => report.value.fraud_assessment?.reason || '');
const fraudSignals = computed(() => report.value.fraud_assessment?.signals || []);
const isFallbackReport = computed(() => !!report.value.is_fallback);
const fallbackReason = computed(() => report.value.fallback_reason || '');
const isRecommendationOverridden = computed(() => !!report.value.original_recommendation);
const recommendationOverrideReason = computed(() => report.value.recommendation_override_reason || '');
const recommendationTagText = computed(() => {
  if (isRecommendationOverridden.value) {
    return `Recommendation: ${props.analysis?.recommendation || 'Pending'} (AI said ${report.value.original_recommendation})`;
  }
  return `Recommendation: ${props.analysis?.recommendation || 'Pending'}`;
});

// needs_human_review can be forced by either a low confidence score or a
// fraud hard-rule signal (see _should_escalate_to_human in the backend
// orchestrator) -- these are independent triggers, so the banner text must
// reflect which one actually fired instead of always blaming confidence.
const escalationReason = computed(() => {
  const fraudFlagged = !!report.value.fraud_assessment?.needs_investigation;
  const lowConfidence = props.analysis?.confidence_score != null && Number(props.analysis.confidence_score) < 0.6;
  if (fraudFlagged && lowConfidence) {
    return 'This claim was flagged for potential fraud and its confidence score fell below the automated-decision threshold, so it was routed for manual review instead of an AI recommendation.';
  }
  if (fraudFlagged) {
    return 'This claim was flagged for potential fraud, so it was routed for manual review instead of an AI recommendation.';
  }
  return 'Confidence fell below the automated-decision threshold, so this claim was routed for manual review instead of an AI recommendation.';
});

const percent = (value) => `${Math.round(Number(value) * 100)}%`;

const severityTagColor = (label) => ({ Minor: 'success', Moderate: 'warn', Severe: 'danger' }[label] || 'secondary');
const recommendationTagColor = (recommendation) => ({ Approve: 'success', Investigate: 'warn', Deny: 'danger' }[recommendation] || 'secondary');
const fraudTagColor = (score) => (Number(score) >= 0.65 ? 'danger' : Number(score) >= 0.4 ? 'warn' : 'success');
</script>
