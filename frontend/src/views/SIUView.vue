<template>
  <div class="page">
    <div class="hero-card">
      <div>
        <p class="eyebrow">SIU portal</p>
        <h1>Investigate suspicious patterns</h1>
        <p>Surface claims flagged by the AI analysis or referred by an adjuster, and prioritize the highest-risk cases.</p>
      </div>
      <div class="hero-badge">Fraud focused</div>
    </div>

    <div v-if="loading" class="card"><i class="pi pi-spin pi-spinner" /> Loading investigations…</div>
    <div v-else class="card">
      <div class="stats-grid">
        <div class="stat-card">
          <h3>Flagged</h3>
          <p>{{ summary.flagged_count }}</p>
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
        <div class="inline-group" style="justify-content: space-between;">
          <div class="inline-group">
            <strong>{{ claim.claim_id }}</strong>
            <span>{{ claim.claimant_name }}</span>
            <Tag :value="`Fraud score: ${claim.fraud_score}`" :severity="claim.referral_reason === 'fraud_score' ? 'danger' : 'secondary'" />
            <Tag v-if="claim.referral_reason === 'adjuster_referral'" value="Adjuster referral" severity="warn" />
            <Tag v-if="claim.needs_human_review" value="Needs human review" severity="warn" />
            <Tag :value="claim.investigation_status" severity="secondary" />
          </div>
          <Button label="Investigate" size="small" text @click="selectClaim(claim.claim_id)" />
        </div>
        <p>Amount: {{ claim.claimed_amount }}</p>
      </div>
      <p v-if="!claims.length" class="muted">No claims are currently flagged for investigation.</p>
    </div>

    <div v-if="selectedClaimId" class="card">
      <div class="inline-group" style="justify-content: space-between;">
        <h2 class="section-title">Claim {{ selectedClaimId }}</h2>
        <Button label="Close" text size="small" @click="clearSelection" />
      </div>

      <AiAnalysisPanel :analysis="selectedDetail?.analysis_result" />

      <div class="nested-card">
        <strong>Investigation action</strong>
        <label>
          Investigator ID
          <InputText v-model="investigatorId" placeholder="Your analyst ID" />
        </label>
        <label>
          Notes
          <Textarea v-model="notes" rows="3" placeholder="Investigation notes" />
        </label>
        <div class="inline-group">
          <Button label="Open investigation" severity="warn" :loading="submitting" @click="recordAction('under_investigation')" />
          <Button label="Confirm fraud" severity="danger" :loading="submitting" @click="recordAction('confirmed_fraud')" />
          <Button label="Clear flag" severity="success" outlined :loading="submitting" @click="recordAction('cleared')" />
        </div>
        <Message v-if="actionMessage" severity="success">{{ actionMessage }}</Message>
        <Message v-if="actionError" severity="error">{{ actionError }}</Message>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue';
import Tag from 'primevue/tag';
import Button from 'primevue/button';
import InputText from 'primevue/inputtext';
import Textarea from 'primevue/textarea';
import Message from 'primevue/message';
import AiAnalysisPanel from '../components/AiAnalysisPanel.vue';
import { getClaimDetail, getSIUDashboard, submitSIUAction } from '../services/api';

const claims = ref([]);
const summary = ref({ flagged_count: 0, under_investigation_count: 0, confirmed_fraud_count: 0 });
const loading = ref(true);

const selectedClaimId = ref('');
const selectedDetail = ref(null);
const investigatorId = ref('');
const notes = ref('');
const submitting = ref(false);
const actionMessage = ref('');
const actionError = ref('');

let pollTimer = null;

const loadDashboard = async () => {
  try {
    const response = await getSIUDashboard();
    summary.value = response.summary || summary.value;
    claims.value = response.claims || [];
  } finally {
    loading.value = false;
  }
};

const selectClaim = async (claimId) => {
  selectedClaimId.value = claimId;
  actionMessage.value = '';
  actionError.value = '';
  await loadDetail();
};

const loadDetail = async () => {
  if (!selectedClaimId.value) return;
  selectedDetail.value = await getClaimDetail(selectedClaimId.value);
  if (selectedDetail.value?.analysis_result?.status === 'pending') {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(loadDetail, 3000);
  }
};

const clearSelection = () => {
  selectedClaimId.value = '';
  selectedDetail.value = null;
  clearTimeout(pollTimer);
};

const recordAction = async (status) => {
  submitting.value = true;
  actionMessage.value = '';
  actionError.value = '';
  try {
    await submitSIUAction(selectedClaimId.value, {
      investigator_id: investigatorId.value,
      status,
      notes: notes.value,
    });
    actionMessage.value = 'Investigation action recorded.';
    await loadDashboard();
  } catch (error) {
    actionError.value = error.response?.data?.detail || 'Unable to record investigation action';
  } finally {
    submitting.value = false;
  }
};

onMounted(loadDashboard);
onBeforeUnmount(() => clearTimeout(pollTimer));
</script>
