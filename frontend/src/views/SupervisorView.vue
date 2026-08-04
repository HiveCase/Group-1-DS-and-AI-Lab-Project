<template>
  <div class="page">
    <div class="hero-card accent-4">
      <div>
        <p class="eyebrow">Supervisor dashboard</p>
        <h1>Monitor portfolio health</h1>
        <p>Track claim throughput, fraud risk concentration, and AI assessment mix from one executive view.</p>
      </div>
      <div class="hero-badge">Executive view</div>
    </div>

    <div v-if="loading" class="card">Loading analytics…</div>
    <div v-else class="card">
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
          <h3>Average fraud score</h3>
          <p>{{ summary.average_fraud_score }}</p>
        </div>
      </div>
      <div class="card nested-card">
        <h3>Severity mix</h3>
        <p>Minor: {{ summary.severity_counts.Minor }} | Moderate: {{ summary.severity_counts.Moderate }} | Severe: {{ summary.severity_counts.Severe }}</p>
      </div>
      <div class="card nested-card">
        <h3>Coverage flags</h3>
        <p>{{ summary.coverage_flag_rate }} of claims show coverage-limit concerns.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import { getSupervisorAnalytics } from '../services/api';

const summary = ref({ total_claims: 0, pending_count: 0, average_fraud_score: 0, severity_counts: {} });
const loading = ref(true);

onMounted(async () => {
  try {
    const response = await getSupervisorAnalytics();
    summary.value = response.summary || summary.value;
  } finally {
    loading.value = false;
  }
});
</script>
