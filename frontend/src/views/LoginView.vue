<template>
  <div class="auth-page">
    <div class="auth-card">
      <div class="auth-brand">
        <span class="pi pi-shield"></span>
        <h1>Claims Portal</h1>
      </div>
      <p class="auth-subtitle">Sign in to your account</p>

      <div v-if="errorMsg" class="auth-error">
        <span class="pi pi-exclamation-circle"></span>
        {{ errorMsg }}
      </div>

      <div v-if="successMsg" class="auth-success">
        <span class="pi pi-check-circle"></span>
        {{ successMsg }}
      </div>

      <form @submit.prevent="handleLogin" class="auth-form">
        <label>
          Username
          <InputText v-model="username" placeholder="Enter your username" id="login-username" />
        </label>
        <label>
          Password
          <InputText v-model="password" type="password" placeholder="Enter your password" id="login-password" />
        </label>
        <Button type="submit" :label="loading ? 'Signing in...' : 'Sign In'" :disabled="loading" id="login-submit" class="auth-btn" />
      </form>

      <p class="auth-link">
        Don't have an account?
        <router-link to="/signup">Sign up</router-link>
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import InputText from 'primevue/inputtext';
import Button from 'primevue/button';
import { login } from '../services/api.ts';

const router = useRouter();
const route = useRoute();
const username = ref('');
const password = ref('');
const errorMsg = ref('');
const successMsg = ref('');
const loading = ref(false);

onMounted(() => {
  if (route.query.registered) {
    successMsg.value = 'Account created! Please sign in.';
  }
});

async function handleLogin() {
  errorMsg.value = '';
  successMsg.value = '';
  if (!username.value || !password.value) {
    errorMsg.value = 'Please enter both username and password.';
    return;
  }
  loading.value = true;
  try {
    const data = await login(username.value, password.value);
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    localStorage.setItem('user', JSON.stringify(data.user));
    router.push('/');
  } catch (err) {
    errorMsg.value = err.response?.data?.detail || 'Login failed. Please try again.';
  } finally {
    loading.value = false;
  }
}
</script>

<style scoped>
.auth-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-bg);
  padding: var(--space-4);
}
.auth-card {
  background: var(--color-surface);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-card);
  padding: var(--space-8);
  width: 100%;
  max-width: 420px;
}
.auth-brand {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  justify-content: center;
  margin-bottom: var(--space-2);
}
.auth-brand .pi { font-size: 1.6rem; color: var(--portal-claimant); }
.auth-brand h1 { margin: 0; font-size: var(--text-lg); color: var(--color-text); }
.auth-subtitle { text-align: center; color: var(--color-text-muted); margin: 0 0 var(--space-6); font-size: var(--text-sm); }
.auth-form { display: flex; flex-direction: column; gap: var(--space-4); }
.auth-form label { font-size: var(--text-sm); font-weight: 600; color: var(--color-text); display: flex; flex-direction: column; gap: var(--space-1); }
.auth-form :deep(input) {
  width: 100%;
  padding: 0.65rem 0.8rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: var(--text-sm);
  font-family: var(--font-family);
  background: var(--color-surface);
  color: var(--color-text);
  transition: border-color 0.2s;
  box-sizing: border-box;
}
.auth-form :deep(input:focus) { border-color: var(--portal-claimant); outline: none; box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
.auth-btn {
  width: 100%;
  padding: 0.7rem;
  background: var(--portal-claimant) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  color: white !important;
  font-weight: 600;
  font-size: var(--text-sm);
  cursor: pointer;
  transition: background 0.2s, transform 0.1s;
}
.auth-btn:hover:not(:disabled) { background: #1d4ed8 !important; transform: translateY(-1px); }
.auth-btn:disabled { opacity: 0.6; cursor: not-allowed; }
.auth-error {
  background: var(--status-danger-bg);
  color: var(--status-danger);
  padding: var(--space-3);
  border-radius: var(--radius-sm);
  font-size: var(--text-sm);
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-3);
}
.auth-success {
  background: var(--status-good-bg);
  color: var(--status-good);
  padding: var(--space-3);
  border-radius: var(--radius-sm);
  font-size: var(--text-sm);
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin-bottom: var(--space-3);
}
.auth-link { text-align: center; font-size: var(--text-sm); color: var(--color-text-muted); margin-top: var(--space-4); }
.auth-link a { color: var(--portal-claimant); text-decoration: none; font-weight: 600; }
.auth-link a:hover { text-decoration: underline; }
</style>
