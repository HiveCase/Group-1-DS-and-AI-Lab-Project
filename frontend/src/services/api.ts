import axios from 'axios';

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 10000,
});

export async function lookupPolicy(policyNumber: string) {
  const response = await client.post('/policies/lookup', { policy_number: policyNumber });
  return response.data;
}

export async function submitClaim(payload: Record<string, unknown>) {
  const form = new FormData();
  Object.entries(payload).forEach(([key, value]) => {
    if (key === 'photos' && Array.isArray(value)) {
      value.forEach((photo) => form.append('photos', photo));
    } else if (value !== undefined && value !== null) {
      form.append(key, String(value));
    }
  });
  const response = await client.post('/claims', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function getClaim(claimId: string) {
  const response = await client.get(`/claims/${claimId}`);
  return response.data;
}

export async function listClaims(status?: string) {
  const response = await client.get('/claims', { params: status ? { status } : {} });
  return response.data;
}

export async function getAdjusterDashboard() {
  const response = await client.get('/claims/adjuster-dashboard');
  return response.data;
}

export async function getClaimDetail(claimId: string) {
  const response = await client.get(`/claims/${claimId}/detail`);
  return response.data;
}

export async function submitDecision(claimId: string, payload: Record<string, unknown>) {
  const response = await client.post(`/claims/${claimId}/decision`, payload);
  return response.data;
}

export async function getSIUDashboard() {
  const response = await client.get('/claims/siu-dashboard');
  return response.data;
}

export async function getSupervisorAnalytics() {
  const response = await client.get('/analytics/summary');
  return response.data;
}
