<template>
  <div class="page">
    <div class="hero-card">
      <div>
        <p class="eyebrow">Supervisor dashboard</p>
        <h1>Monitor portfolio health</h1>
        <p>Track claim throughput, fraud risk concentration, and AI assessment mix from one executive view.</p>
      </div>
      <div class="hero-badge">Executive view</div>
    </div>

    <div v-if="loading" class="card"><i class="pi pi-spin pi-spinner" /> Loading analytics…</div>
    <template v-else>
      <div class="card">
        <div class="stats-grid">
          <div class="stat-card">
            <h3>Total claims</h3>
            <p>{{ summary.total_claims }}</p>
          </div>
          <div class="stat-card">
            <h3>Pending</h3>
            <p>{{ summary.pending_count }}</p>
          </div>
          <div class="stat-card">
            <h3>Approved</h3>
            <p>{{ summary.approved_count }}</p>
          </div>
          <div class="stat-card">
            <h3>Denied</h3>
            <p>{{ summary.denied_count }}</p>
          </div>
          <div class="stat-card">
            <h3>Average fraud score</h3>
            <p>{{ summary.average_fraud_score?.toFixed(2) }}</p>
          </div>
          <div class="stat-card">
            <h3>Claims processed today</h3>
            <p>{{ summary.claims_processed_today }}</p>
          </div>
        </div>
      </div>

      <div v-if="hasSeverityData" class="card">
        <h2 class="section-title">Severity distribution</h2>
        <div class="bar-chart">
          <div v-for="row in severityRows" :key="row.label" class="bar-chart-row">
            <span>{{ row.label }}</span>
            <div class="bar-track"><div class="bar-fill" :style="{ width: row.pct + '%', background: row.color }" /></div>
            <span>{{ row.value }}</span>
          </div>
        </div>
      </div>
      <div v-else class="card">
        <h2 class="section-title">Severity distribution</h2>
        <p class="muted">No completed AI analyses yet, so there is no severity mix to chart.</p>
      </div>

      <div class="card">
        <h2 class="section-title">Coverage flags</h2>
        <p>{{ (summary.coverage_flag_rate * 100).toFixed(1) }}% of claims show coverage-limit concerns.</p>
      </div>

      <div class="card">
        <h2 class="section-title">AI pipeline system status</h2>
        <div class="inline-group">
          <Tag :value="systemStatus.pipeline_status || 'unknown'" :severity="systemStatus.pipeline_status === 'operational' ? 'success' : 'danger'" />
          <Tag :value="`Avg analysis time: ${systemStatus.avg_analysis_time_seconds != null ? systemStatus.avg_analysis_time_seconds + 's' : 'n/a'}`" severity="secondary" />
          <Tag :value="`Awaiting analysis: ${systemStatus.claims_awaiting_analysis ?? 0}`" severity="secondary" />
          <Tag v-if="systemStatus.recent_failure_count" :value="`Failures (last hour): ${systemStatus.recent_failure_count}`" severity="danger" />
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue';
import Tag from 'primevue/tag';
import { getSupervisorAnalytics } from '../services/api';

const summary = ref({ total_claims: 0, pending_count: 0, average_fraud_score: 0, severity_counts: {}, coverage_flag_rate: 0 });
const systemStatus = ref({});
const loading = ref(true);

const severityColors = { Minor: '#16a34a', Moderate: '#d97706', Severe: '#dc2626' };

const hasSeverityData = computed(() => Object.values(summary.value.severity_counts || {}).some((count) => count > 0));

const severityRows = computed(() => {
  const counts = summary.value.severity_counts || {};
  const max = Math.max(1, ...Object.values(counts));
  return Object.entries(counts).map(([label, value]) => ({
    label,
    value,
    pct: Math.round((value / max) * 100),
    color: severityColors[label] || '#64748b',
  }));
});

onMounted(async () => {
  try {
    const response = await getSupervisorAnalytics();
    summary.value = response.summary || summary.value;
    systemStatus.value = response.summary?.system_status || {};
  } finally {
    loading.value = false;
  }
});
</script>
