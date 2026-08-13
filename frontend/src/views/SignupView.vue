<template>
  <div class="auth-page">
    <div class="auth-card">
      <div class="auth-brand">
        <span class="pi pi-shield"></span>
        <h1>Claims Portal</h1>
      </div>
      <p class="auth-subtitle">Create your account</p>

      <div v-if="errorMsg" class="auth-error">
        <span class="pi pi-exclamation-circle"></span>
        {{ errorMsg }}
      </div>

      <form @submit.prevent="handleSignup" class="auth-form">
        <label>
          Full Name
          <InputText v-model="form.full_name" placeholder="Enter your full name" id="signup-fullname" />
        </label>
        <label>
          Email
          <InputText v-model="form.email" type="email" placeholder="Enter your email" id="signup-email" />
        </label>
        <label>
          Username
          <InputText v-model="form.username" placeholder="Choose a username" id="signup-username" />
        </label>
        <div class="auth-row">
          <label>
            Password
            <InputText v-model="form.password" type="password" placeholder="Min 6 characters" id="signup-password" />
          </label>
          <label>
            Confirm Password
            <InputText v-model="confirmPassword" type="password" placeholder="Re-enter password" id="signup-confirm" />
          </label>
        </div>
        <Button type="submit" :label="loading ? 'Creating account...' : 'Create Account'" :disabled="loading" id="signup-submit" class="auth-btn" />
      </form>

      <p class="auth-link">
        Already have an account?
        <router-link to="/login">Sign in</router-link>
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue';
import { useRouter } from 'vue-router';
import InputText from 'primevue/inputtext';
import Button from 'primevue/button';
import { signup } from '../services/api.ts';

const router = useRouter();
const form = reactive({
  full_name: '',
  email: '',
  username: '',
  password: '',
});
const confirmPassword = ref('');
const errorMsg = ref('');
const loading = ref(false);

async function handleSignup() {
  errorMsg.value = '';
  if (!form.full_name || !form.email || !form.username || !form.password) {
    errorMsg.value = 'Please fill in all fields.';
    return;
  }
  if (form.password.length < 6) {
    errorMsg.value = 'Password must be at least 6 characters.';
    return;
  }
  if (form.password !== confirmPassword.value) {
    errorMsg.value = 'Passwords do not match.';
    return;
  }
  loading.value = true;
  try {
    await signup(form);
    router.push({ path: '/login', query: { registered: '1' } });
  } catch (err) {
    errorMsg.value = err.response?.data?.detail || 'Signup failed. Please try again.';
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
  max-width: 480px;
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
.auth-row { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3); }
@media (max-width: 480px) { .auth-row { grid-template-columns: 1fr; } }
.auth-form :deep(input), .auth-select {
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
.auth-form :deep(input:focus), .auth-select:focus { border-color: var(--portal-claimant); outline: none; box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
.auth-select { appearance: auto; cursor: pointer; }
.role-picker {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-2);
}
.role-option {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: 0.6rem 0.8rem;
  border: 2px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  color: var(--color-text);
  font-size: var(--text-sm);
  font-family: var(--font-family);
  font-weight: 500;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
}
.role-option:hover { border-color: var(--portal-claimant); background: rgba(37,99,235,0.04); }
.role-option.active { border-color: var(--portal-claimant); background: rgba(37,99,235,0.08); color: var(--portal-claimant); font-weight: 600; }
.role-option .pi { font-size: 0.9rem; }
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
.auth-link { text-align: center; font-size: var(--text-sm); color: var(--color-text-muted); margin-top: var(--space-4); }
.auth-link a { color: var(--portal-claimant); text-decoration: none; font-weight: 600; }
.auth-link a:hover { text-decoration: underline; }
</style>
