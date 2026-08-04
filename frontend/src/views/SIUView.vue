<template>
  <div class="page">
    <div class="hero-card accent-3">
      <div>
        <p class="eyebrow">SIU portal</p>
        <h1>Investigate suspicious patterns</h1>
        <p>Surface claims flagged by the AI analysis and prioritize the highest-risk cases.</p>
      </div>
      <div class="hero-badge">Fraud focused</div>
    </div>

    <div v-if="loading" class="card">Loading investigations…</div>
    <div v-else class="card">
      <div class="stats-grid">
        <div class="stat-card">
          <h3>High-risk</h3>
          <p>{{ summary.high_risk_count }}</p>
        </div>
        <div class="stat-card">
          <h3>Under investigation</h3>
          <p>{{ summary.under_investigation_count }}</p>
        </div>
        <div class="stat-card">
          <h3>Confirmed fraud</h3>
          <p>{{ summary.confirmed_fraud_count }}</p>
        </div>
      </div>
      <div v-for="claim in claims" :key="claim.claim_id" class="card nested-card">
        <div class="inline-group">
          <strong>{{ claim.claim_id }}</strong>
          <span>{{ claim.claimant_name }}</span>
          <span>Fraud score: {{ claim.fraud_score }}</span>
        </div>
        <p>Amount: {{ claim.claimed_amount }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import { getSIUDashboard } from '../services/api';

const claims = ref([]);
const loading = ref(true);

onMounted(async () => {
  try {
    const response = await getSIUDashboard();
    claims.value = response.claims || [];
  } finally {
    loading.value = false;
  }
});
</script>
