import { ref } from 'vue';
import { login as loginApi, signup as signupApi } from './api';

const STORAGE_KEY = 'claims_portal_auth';

function loadStoredAuth() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

const stored = loadStoredAuth();
export const currentUser = ref(stored?.user || null);
export const accessToken = ref(stored?.access_token || null);

export async function signup(email, password, role) {
  return signupApi({ email, password, role });
}

export async function login(email, password) {
  const response = await loginApi({ email, password });
  currentUser.value = response.user;
  accessToken.value = response.access_token;
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ user: response.user, access_token: response.access_token }));
  return response;
}

export function logout() {
  currentUser.value = null;
  accessToken.value = null;
  localStorage.removeItem(STORAGE_KEY);
}
