import axios from 'axios';
import { accessToken, logout } from './auth';

const defaultBaseUrl = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? '/api' : '');

const client = axios.create({
  baseURL: defaultBaseUrl,
  timeout: 10000,
});

// Every route except /auth/*, /health, and the annotated-photo image now
// requires a bearer token server-side -- attach it here rather than at
// each call site, so it can never be forgotten on a new endpoint.
client.interceptors.request.use((config) => {
  if (accessToken.value) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${accessToken.value}`;
  }
  return config;
});

// A 401 means the token is missing/expired/invalid -- the stored session
// is no longer good for anything, so clear it and send the user back to
// log in rather than leaving them stuck on a page that will 401 forever.
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && window.location.pathname !== '/login') {
      logout();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  },
);

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

export function annotatedPhotoUrl(claimId: string) {
  return `${defaultBaseUrl}/claims/${claimId}/annotated-photo`;
}

export async function submitDecision(claimId: string, payload: Record<string, unknown>) {
  const response = await client.post(`/claims/${claimId}/decision`, payload);
  return response.data;
}

export async function getSIUDashboard() {
  const response = await client.get('/claims/siu-dashboard');
  return response.data;
}

export async function submitSIUAction(claimId: string, payload: Record<string, unknown>) {
  const response = await client.post(`/claims/${claimId}/siu-action`, payload);
  return response.data;
}

export async function getSupervisorAnalytics() {
  const response = await client.get('/analytics/summary');
  return response.data;
}

export async function signup(payload: { email: string; password: string; role: string }) {
  const response = await client.post('/auth/signup', payload);
  return response.data;
}

export async function login(payload: { email: string; password: string }) {
  const response = await client.post('/auth/login', payload);
  return response.data;
}
