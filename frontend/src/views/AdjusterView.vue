<template>
  <div class="page">
    <div class="hero-card">
      <div>
        <p class="eyebrow">Adjuster portal</p>
        <h1>Review and decide with context</h1>
        <p>Inspect incoming claims, review AI findings, and submit an approval or denial decision.</p>
      </div>
      <div class="hero-badge">Decision-ready</div>
    </div>

    <div v-if="loading" class="card"><i class="pi pi-spin pi-spinner" /> Loading dashboard…</div>
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

      <DataTable :value="claims" size="small" selectionMode="single" @row-click="(event) => selectClaim(event.data.claim_id)">
        <Column field="claim_id" header="Claim ID" />
        <Column field="claimant_name" header="Claimant" />
        <Column header="Predicted damage">
          <template #body="{ data }">
            <a :href="annotatedPhotoUrl(data.claim_id)" target="_blank" rel="noopener" @click.stop>
              <img
                :src="annotatedPhotoUrl(data.claim_id)"
                alt="Predicted damage regions"
                class="predicted-damage-thumb"
                @error="(event) => { event.target.style.visibility = 'hidden'; }"
              />
            </a>
          </template>
        </Column>
        <Column field="status" header="Status">
          <template #body="{ data }"><Tag :value="data.status" :severity="statusTagColor(data.status)" /></template>
        </Column>
        <Column field="claimed_amount" header="Claimed amount" />
      </DataTable>
      <p v-if="!claims.length" class="muted">No claims are waiting for review right now.</p>
    </div>

    <div v-if="selectedClaimId" class="card">
      <div class="inline-group" style="justify-content: space-between;">
        <h2 class="section-title">Claim {{ selectedClaimId }}</h2>
        <Button label="Close" text size="small" @click="clearSelection" />
      </div>

      <AiAnalysisPanel :analysis="selectedDetail?.analysis_result" />

      <div class="nested-card">
        <strong>Decision</strong>
        <div class="grid two-col">
          <label>
            Decision
            <Select v-model="decision" :options="decisionOptions" optionLabel="label" optionValue="value" placeholder="Choose a decision" />
          </label>
          <label v-if="decision === 'approved'">
            Settlement amount
            <InputNumber v-model="settlementAmount" mode="currency" currency="USD" locale="en-US" />
          </label>
        </div>
        <label>
          Reasoning note
          <Textarea v-model="reasoningNote" rows="3" placeholder="Explain the decision" />
        </label>
        <Button label="Submit decision" :disabled="!decision || !reasoningNote || submittingDecision" :loading="submittingDecision" @click="submitDecisionForClaim" />
        <Message v-if="decisionMessage" severity="success">{{ decisionMessage }}</Message>
        <Message v-if="decisionError" severity="error">{{ decisionError }}</Message>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import Tag from 'primevue/tag';
import Button from 'primevue/button';
import Select from 'primevue/select';
import InputNumber from 'primevue/inputnumber';
import Textarea from 'primevue/textarea';
import Message from 'primevue/message';
import AiAnalysisPanel from '../components/AiAnalysisPanel.vue';
import { annotatedPhotoUrl, getAdjusterDashboard, getClaimDetail, submitDecision } from '../services/api';

const loading = ref(true);
const summary = ref({ pending_count: 0, approved_count: 0, denied_count: 0 });
const claims = ref([]);

const selectedClaimId = ref('');
const selectedDetail = ref(null);
const decision = ref('');
const settlementAmount = ref(null);
const reasoningNote = ref('');
const submittingDecision = ref(false);
const decisionMessage = ref('');
const decisionError = ref('');

const decisionOptions = [
  { label: 'Approve', value: 'approved' },
  { label: 'Deny', value: 'denied' },
  { label: 'Request more information', value: 'request_more_info' },
];

let pollTimer = null;

const statusTagColor = (status) => ({ submitted: 'secondary', 'under review': 'warn', approved: 'success', denied: 'danger' }[status] || 'secondary');

const loadDashboard = async () => {
  try {
    const response = await getAdjusterDashboard();
    summary.value = response.summary || summary.value;
    claims.value = response.claims || [];
  } finally {
    loading.value = false;
  }
};

const selectClaim = async (claimId) => {
  selectedClaimId.value = claimId;
  decision.value = '';
  settlementAmount.value = null;
  reasoningNote.value = '';
  decisionMessage.value = '';
  decisionError.value = '';
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

const submitDecisionForClaim = async () => {
  submittingDecision.value = true;
  decisionMessage.value = '';
  decisionError.value = '';
  try {
    await submitDecision(selectedClaimId.value, {
      decision: decision.value,
      reasoning_note: reasoningNote.value,
      settlement_amount: decision.value === 'approved' ? settlementAmount.value : null,
    });
    decisionMessage.value = decision.value === 'request_more_info'
      ? 'Decision recorded. Claim routed to SIU for further verification.'
      : 'Decision recorded.';
    await Promise.all([loadDashboard(), loadDetail()]);
  } catch (error) {
    decisionError.value = error.response?.data?.detail || 'Unable to submit decision';
  } finally {
    submittingDecision.value = false;
  }
};

onMounted(loadDashboard);
onBeforeUnmount(() => clearTimeout(pollTimer));
</script>

<style scoped>
.predicted-damage-thumb {
  height: 72px;
  width: 108px;
  object-fit: cover;
  border-radius: 4px;
  cursor: zoom-in;
}
</style>
