import axios from 'axios';

const defaultBaseUrl = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? '/api' : '');

const client = axios.create({
  baseURL: defaultBaseUrl,
  timeout: 10000,
});

// ── Attach JWT to every request ────────────────────────────────────
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ── Redirect to /login on 401 ──────────────────────────────────────
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      localStorage.removeItem('user');
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// ── Auth endpoints ─────────────────────────────────────────────────

export async function login(username: string, password: string) {
  const response = await client.post('/auth/login', { username, password });
  return response.data;
}

export async function signup(data: Record<string, unknown>) {
  const response = await client.post('/auth/signup', data);
  return response.data;
}

export async function refreshToken(refresh_token: string) {
  const response = await client.post('/auth/refresh', { refresh_token });
  return response.data;
}

export async function getMe() {
  const response = await client.get('/auth/me');
  return response.data;
}

// ── Existing endpoints ─────────────────────────────────────────────

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
