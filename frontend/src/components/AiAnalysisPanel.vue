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

      <div v-if="recommendationReason" class="nested-card">
        <strong>Why this recommendation</strong>
        <p>{{ recommendationReason }}</p>
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
            <div v-if="finding.retrieval_breakdown" class="muted retrieval-caption">
              Retrieved by {{ retrievalSourceLabel(finding.retrieval_breakdown) }}
            </div>
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

      <div v-if="hasExplainability" class="nested-card explainability-card">
        <div class="explainability-header">
          <i class="pi pi-sitemap" />
          <strong>AI explainability</strong>
        </div>

        <details v-if="saliencyDetections.length" class="explain-details">
          <summary>Damage classification &mdash; why this region ({{ saliencyDetections.length }})</summary>
          <div class="explain-body">
            <p class="muted saliency-intro">
              Occlusion-sensitivity map: each cell of the detected region was masked in turn and re-run
              through the model &mdash; darker cells are the pixels it relied on most for this classification.
            </p>
            <div v-for="(detection, index) in saliencyDetections" :key="index" class="saliency-block">
              <div class="clause-meta">{{ detection.class_name }} &middot; confidence {{ percent(detection.confidence) }}</div>
              <div
                class="saliency-grid"
                :style="{ gridTemplateColumns: `repeat(${detection.saliency.grid_size}, 1fr)` }"
              >
                <div
                  v-for="(cell, cellIndex) in flattenGrid(detection.saliency.importance)"
                  :key="cellIndex"
                  class="saliency-cell"
                  :style="{ backgroundColor: heatColor(cell) }"
                />
              </div>
              <p v-if="detection.saliency.peak_cell" class="muted saliency-caption">
                Most influential region: row {{ detection.saliency.peak_cell.row + 1 }}, column
                {{ detection.saliency.peak_cell.col + 1 }} &mdash; masking it dropped confidence by
                {{ percent(detection.saliency.peak_confidence_drop) }}.
              </p>
            </div>
          </div>
        </details>

        <details v-if="severityPerRegion.length" class="explain-details">
          <summary>Severity scoring &mdash; how this was determined ({{ percent(severitySummary.severity_score) }} of photo area)</summary>
          <div class="explain-body step-list">
            <p class="muted saliency-intro">
              Severity is a deterministic ratio of total damaged area to the photo's own area, not the
              model's raw detection confidence. Each region below shows the running severity label after
              it was added, in the order counted &mdash; thresholds: &lt;8% Minor, 8&ndash;20% Moderate, &ge;20% Severe.
            </p>
            <div v-for="(region, index) in severityPerRegion" :key="index" class="step-row">
              <div class="step-row-header">
                <span>{{ region.class_name }} &middot; confidence {{ percent(region.confidence) }}</span>
                <Tag :value="region.severity" :severity="severityTagColor(region.severity)" />
              </div>
            </div>
          </div>
        </details>

        <details v-if="fraudBreakdown.length" class="explain-details">
          <summary>Fraud score &mdash; how it was calculated ({{ percent(analysis.fraud_score) }})</summary>
          <div class="explain-body step-list">
            <div
              v-for="(step, index) in fraudBreakdown"
              :key="index"
              class="step-row"
              :class="{ 'step-row-inactive': !step.triggered }"
            >
              <div class="step-row-header">
                <span>{{ step.label }}</span>
                <span class="step-row-value">{{ step.triggered ? formatDelta(step.delta) : 'not triggered' }}</span>
              </div>
              <div class="bar-track">
                <div class="bar-fill" :style="{ width: percent(step.running_total), background: fraudStepColor(step.running_total) }" />
              </div>
              <p class="muted saliency-caption">{{ step.detail }}</p>
            </div>
          </div>
        </details>

        <details v-if="narrativeAnalysis" class="explain-details">
          <summary>Narrative risk analysis</summary>
          <div class="explain-body">
            <blockquote v-for="(flag, index) in narrativeAnalysis.flags" :key="index" class="narrative-flag">
              <span>&ldquo;{{ flag.quote }}&rdquo;</span>
              <span class="narrative-concern">{{ flag.concern }}</span>
            </blockquote>
            <p v-if="!narrativeAnalysis.flags.length && narrativeAnalysis.available" class="muted">
              No red-flag phrases found in the incident narrative.
            </p>
            <p v-else-if="!narrativeAnalysis.available" class="muted">
              Narrative analysis unavailable{{ narrativeAnalysis.reason ? `: ${narrativeAnalysis.reason}` : '' }}.
            </p>
          </div>
        </details>

        <details v-if="consistencyCheck" class="explain-details">
          <summary>
            <span>Recommendation consistency check</span>
            <span :class="consistencyCheck.all_passed ? 'consistency-pass' : 'consistency-fail'">
              {{ consistencyCheck.all_passed ? 'All checks passed' : 'Issue found' }}
            </span>
          </summary>
          <ul class="explain-body clause-list">
            <li v-for="check in consistencyCheck.checks" :key="check.rule" class="consistency-item">
              <i :class="check.passed ? 'pi pi-check-circle consistency-pass' : 'pi pi-times-circle consistency-fail'" />
              <div>
                <div class="clause-meta">{{ consistencyRuleLabel(check.rule) }}</div>
                <div>{{ check.detail }}</div>
              </div>
            </li>
          </ul>
        </details>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import Tag from 'primevue/tag';

const CONSISTENCY_RULE_LABELS = {
  citations_grounded: 'Citations are grounded in retrieved clauses',
  sub_limit_respected: 'Policy-limit violations are not approved',
  coverage_basis_present: 'Approval has a coverage basis',
  fraud_signals_reflected: 'Fraud signals are reflected in the recommendation',
};

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

const saliencyDetections = computed(() => (props.analysis?.detections || []).filter((d) => d && d.saliency));
const severitySummary = computed(() => report.value.severity_summary || null);
const severityPerRegion = computed(() => severitySummary.value?.per_region || []);
const fraudBreakdown = computed(() => report.value.fraud_assessment?.score_breakdown || []);
const narrativeAnalysis = computed(() => report.value.fraud_assessment?.narrative_analysis || null);
const consistencyCheck = computed(() => report.value.consistency_check || null);
const hasExplainability = computed(() => (
  saliencyDetections.value.length > 0
  || severityPerRegion.value.length > 0
  || fraudBreakdown.value.length > 0
  || !!narrativeAnalysis.value
  || !!consistencyCheck.value
));
const isFallbackReport = computed(() => !!report.value.is_fallback);
const fallbackReason = computed(() => report.value.fallback_reason || '');
const recommendationReason = computed(() => report.value.recommendation_reason || '');
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

const flattenGrid = (grid) => (grid || []).flat();
// Floored at 0.08 (not 0) so a zero-importance cell still reads as part of
// the grid rather than disappearing into the card background -- the grid
// structure itself is part of what's being communicated.
const heatColor = (value) => `rgba(185, 28, 28, ${Math.max(0.08, Math.min(1, Number(value) || 0))})`;
const formatDelta = (delta) => (delta > 0 ? `+${delta}` : delta < 0 ? `${delta}` : '±0');
const fraudStepColor = (value) => (Number(value) >= 0.65 ? 'var(--status-danger)' : Number(value) >= 0.4 ? 'var(--status-warning)' : 'var(--status-good)');
const consistencyRuleLabel = (rule) => CONSISTENCY_RULE_LABELS[rule] || rule;

const retrievalSourceLabel = (breakdown) => {
  const parts = [];
  if (breakdown.dense_rank) parts.push(`semantic match #${breakdown.dense_rank}`);
  if (breakdown.sparse_rank) parts.push(`keyword match #${breakdown.sparse_rank}`);
  return parts.length ? parts.join(' + ') : 'not closely ranked by either signal';
};
</script>
