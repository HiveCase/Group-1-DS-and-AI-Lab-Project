<template>
  <div class="page">
    <div class="hero-card">
      <div>
        <p class="eyebrow">Claimant portal</p>
        <h1>File a claim in minutes</h1>
        <p>Look up your policy, submit incident details, and track the claim status from one place.</p>
      </div>
      <div class="hero-badge">Fast and guided</div>
    </div>

    <div class="grid two-col">
      <div class="card">
        <h2 class="section-title">Policy lookup</h2>
        <div class="inline-group">
          <InputText v-model="policyNumber" placeholder="Enter policy number" />
          <Button label="Lookup policy" @click="lookUpPolicy" />
        </div>
        <p v-if="policyDetails" class="status">Policy {{ policyDetails.policy_number || policyDetails.policyNumber }} is {{ policyDetails.status || 'active' }} with limit {{ policyDetails.policy_limit || policyDetails.limit }}</p>
        <Message v-if="policyError" severity="error">{{ policyError }}</Message>
      </div>

      <div class="card">
        <h2 class="section-title">Claim status</h2>
        <div class="inline-group">
          <InputText v-model="lookupClaimId" placeholder="Enter claim ID" />
          <Button label="Check status" @click="lookUpClaim" />
        </div>
        <p v-if="claimStatus" class="status">{{ claimStatus }}</p>
        <Message v-if="claimError" severity="error">{{ claimError }}</Message>
      </div>
    </div>

    <form class="card" @submit.prevent="submitClaim">
      <h2 class="section-title">Submit a new claim</h2>
      <div class="grid two-col">
        <label>
          Policy Number
          <InputText v-model="policyNumber" placeholder="Enter policy number" />
        </label>
        <label>
          Claimant Name
          <InputText v-model="claimantName" placeholder="Full name" />
        </label>
        <label>
          Contact Information
          <InputText v-model="contactInfo" placeholder="Email or phone" />
        </label>
        <label>
          Incident Date
          <DatePicker v-model="incidentDate" dateFormat="yy-mm-dd" showIcon :minDate="minIncidentDate" :maxDate="maxIncidentDate" />
        </label>
        <label>
          Vehicle No
          <InputText v-model="vehicleNo" placeholder="Vehicle registration number" />
        </label>
      </div>
      <label>
        Incident Description
        <Textarea v-model="incidentDescription" rows="3" placeholder="Describe the incident" />
      </label>
      <label>
        Claimed Amount
        <InputNumber v-model="claimedAmount" mode="currency" currency="USD" locale="en-US" />
      </label>
      <label>
        Damage Photos
        <input type="file" accept="image/*" multiple @change="selectPhotos" />
      </label>
      <p class="muted">{{ photos.length }} photo(s) selected</p>
      <Button type="submit" label="Submit Claim" :loading="submitting" />
    </form>

    <div v-if="message" class="card success-panel">
      <h2 class="section-title">Confirmation</h2>
      <p class="status">{{ message }}</p>
      <p v-if="submittedClaimId">Status: submitted</p>
    </div>
    <Message v-if="submitError" severity="error">{{ submitError }}</Message>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import InputText from 'primevue/inputtext';
import InputNumber from 'primevue/inputnumber';
import Textarea from 'primevue/textarea';
import DatePicker from 'primevue/datepicker';
import Button from 'primevue/button';
import Message from 'primevue/message';
import { getClaim, lookupPolicy, submitClaim as submitClaimApi } from '../services/api';

const policyNumber = ref('');
const claimantName = ref('');
const contactInfo = ref('');
const incidentDate = ref(null);
const vehicleNo = ref('');
const incidentDescription = ref('');
const claimedAmount = ref(null);
const photos = ref([]);
const message = ref('');
const submittedClaimId = ref('');
const lookupClaimId = ref('');
const claimStatus = ref('');
const claimError = ref('');
const policyDetails = ref(null);
const policyError = ref('');
const submitError = ref('');
const submitting = ref(false);

// Incident date picker: only the last 1 year up to today is a valid
// incident date -- future dates are never valid, and anything older is
// outside the window this portal accepts claims for.
const maxIncidentDate = new Date();
const minIncidentDate = new Date(maxIncidentDate);
minIncidentDate.setFullYear(minIncidentDate.getFullYear() - 1);

const lookUpPolicy = async () => {
  policyError.value = '';
  try {
    policyDetails.value = await lookupPolicy(policyNumber.value);
  } catch (error) {
    policyDetails.value = null;
    policyError.value = error.response?.data?.detail || 'Unable to lookup policy';
  }
};

const selectPhotos = (event) => {
  photos.value = Array.from(event.target.files || []);
};

const toIsoDate = (value) => {
  if (!value) return '';
  const date = value instanceof Date ? value : new Date(value);
  return date.toISOString().slice(0, 10);
};

const submitClaim = async () => {
  submitError.value = '';
  message.value = '';

  if (photos.value.length < 1 || photos.value.length > 5) {
    submitError.value = 'Upload between 1 and 5 photos';
    return;
  }

  submitting.value = true;
  try {
    const payload = {
      policy_number: policyNumber.value,
      claimant_name: claimantName.value,
      contact_info: contactInfo.value,
      incident_date: toIsoDate(incidentDate.value),
      vehicle_no: vehicleNo.value,
      incident_description: incidentDescription.value,
      claimed_amount: Number(claimedAmount.value),
      photos: photos.value,
    };

    const response = await submitClaimApi(payload);
    submittedClaimId.value = response.claim_id || response.id || '';
    lookupClaimId.value = submittedClaimId.value;
    message.value = `Claim submitted successfully with id ${submittedClaimId.value}`;
  } catch (error) {
    submitError.value = error.response?.data?.detail || 'Unable to submit claim';
  } finally {
    submitting.value = false;
  }
};

const lookUpClaim = async () => {
  claimError.value = '';
  claimStatus.value = '';
  try {
    const response = await getClaim(lookupClaimId.value);
    claimStatus.value = `Claim ${response.claim_id || response.id || lookupClaimId.value} is ${response.status || 'unknown'}`;
  } catch (error) {
    claimError.value = error.response?.data?.detail || 'Unable to find claim';
  }
};
</script>
