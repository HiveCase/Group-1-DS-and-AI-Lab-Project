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
        <h2>Policy lookup</h2>
        <div class="inline-group">
          <input v-model="policyNumber" placeholder="Enter policy number" />
          <button type="button" @click="lookUpPolicy">Lookup policy</button>
        </div>
        <p v-if="policyDetails" class="status">Policy {{ policyDetails.policy_number }} is {{ policyDetails.status }} with limit {{ policyDetails.policy_limit }}</p>
        <p v-if="policyError" class="error">{{ policyError }}</p>
      </div>

      <div class="card">
        <h2>Claim status</h2>
        <div class="inline-group">
          <input v-model="lookupClaimId" placeholder="Enter claim ID" />
          <button type="button" @click="lookUpClaim">Check status</button>
        </div>
        <p v-if="claimStatus" class="status">{{ claimStatus }}</p>
        <p v-if="claimError" class="error">{{ claimError }}</p>
      </div>
    </div>

    <form class="card" @submit.prevent="submitClaim">
      <h2>Submit a new claim</h2>
      <div class="grid two-col">
        <label>
          Policy Number
          <input v-model="policyNumber" placeholder="Enter policy number" />
        </label>
        <label>
          Claimant Name
          <input v-model="claimantName" placeholder="Full name" />
        </label>
        <label>
          Contact Information
          <input v-model="contactInfo" placeholder="Email or phone" />
        </label>
        <label>
          Incident Date
          <input v-model="incidentDate" type="date" />
        </label>
      </div>
      <label>
        Incident Description
        <textarea v-model="incidentDescription" placeholder="Describe the incident"></textarea>
      </label>
      <label>
        Claimed Amount
        <input v-model="claimedAmount" type="number" />
      </label>
      <label>
        Damage Photos
        <input type="file" accept="image/*" multiple @change="selectPhotos" />
      </label>
      <p class="muted">{{ photos.length }} photo(s) selected</p>
      <button type="submit">Submit Claim</button>
    </form>

    <div v-if="message" class="card success-panel">
      <h2>Confirmation</h2>
      <p class="status">{{ message }}</p>
      <p v-if="submittedClaimId">Status: submitted</p>
    </div>
    <p v-if="submitError" class="error">{{ submitError }}</p>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { getClaim, lookupPolicy, submitClaim as submitClaimApi } from '../services/api';

const policyNumber = ref('');
const claimantName = ref('');
const contactInfo = ref('');
const incidentDate = ref('');
const incidentDescription = ref('');
const claimedAmount = ref('');
const photos = ref([]);
const message = ref('');
const submittedClaimId = ref('');
const lookupClaimId = ref('');
const claimStatus = ref('');
const claimError = ref('');
const policyDetails = ref(null);
const policyError = ref('');
const submitError = ref('');

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

const submitClaim = async () => {
  submitError.value = '';
  message.value = '';

  try {
    if (photos.value.length < 1 || photos.value.length > 5) {
      submitError.value = 'Upload between 1 and 5 photos';
      return;
    }

    const payload = {
      policy_number: policyNumber.value,
      claimant_name: claimantName.value,
      contact_info: contactInfo.value,
      incident_date: incidentDate.value,
      incident_description: incidentDescription.value,
      claimed_amount: Number(claimedAmount.value),
      photos: photos.value,
    };

    const response = await submitClaimApi(payload);
    submittedClaimId.value = response.claim_id;
    lookupClaimId.value = response.claim_id;
    message.value = `Claim submitted successfully with id ${response.claim_id}`;
  } catch (error) {
    submitError.value = error.response?.data?.detail || 'Unable to submit claim';
  }
};

const lookUpClaim = async () => {
  claimError.value = '';
  claimStatus.value = '';
  try {
    const response = await getClaim(lookupClaimId.value);
    claimStatus.value = `Claim ${response.claim_id} is ${response.status}`;
  } catch (error) {
    claimError.value = error.response?.data?.detail || 'Unable to find claim';
  }
};
</script>
