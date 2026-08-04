<template>
  <div class="page">
    <div class="hero-card accent-2">
      <div>
        <p class="eyebrow">Adjuster portal</p>
        <h1>Review and decide with context</h1>
        <p>Inspect incoming claims, review AI findings, and submit an approval or denial decision.</p>
      </div>
      <div class="hero-badge">Decision-ready</div>
    </div>

    <div v-if="loading" class="card">Loading dashboard…</div>
    <div v-else class="card">
      <div class="stats-grid">
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
      </div>
      <div v-for="claim in claims" :key="claim.claim_id" class="card nested-card">
        <div class="inline-group">
          <strong>{{ claim.claim_id }}</strong>
          <span>{{ claim.claimant_name }}</span>
          <span>{{ claim.status }}</span>
        </div>
        <p>Claimed amount: {{ claim.claimed_amount }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import { getAdjusterDashboard } from '../services/api';

const loading = ref(true);
const summary = ref({ pending_count: 0, approved_count: 0, denied_count: 0 });
const claims = ref([]);

onMounted(async () => {
  try {
    const response = await getAdjusterDashboard();
    summary.value = response.summary || summary.value;
    claims.value = response.claims || [];
  } finally {
    loading.value = false;
  }
});
</script>
