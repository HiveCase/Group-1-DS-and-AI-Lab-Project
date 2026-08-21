<template>
  <div class="page auth-page">
    <div class="card auth-card">
      <h1 class="section-title">Sign up</h1>
      <p class="muted">Create a Claims Portal account.</p>

      <form @submit.prevent="handleSignup">
        <label>
          Email
          <InputText v-model="email" type="email" placeholder="you@example.com" required />
        </label>
        <label>
          Password
          <Password v-model="password" toggleMask placeholder="At least 8 characters" required />
        </label>
        <p class="muted">New accounts can file and track claims (the Claimant portal). Adjuster/SIU/Supervisor access is granted separately by an administrator.</p>
        <Button type="submit" label="Sign up" :loading="submitting" />
      </form>

      <Message v-if="error" severity="error">{{ error }}</Message>
      <Message v-if="success" severity="success">{{ success }}</Message>

      <p class="muted">
        Already have an account?
        <router-link to="/login">Log in</router-link>
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import InputText from 'primevue/inputtext';
import Password from 'primevue/password';
import Button from 'primevue/button';
import Message from 'primevue/message';
import { signup } from '../services/auth';

const router = useRouter();

const email = ref('');
const password = ref('');
const error = ref('');
const success = ref('');
const submitting = ref(false);

const handleSignup = async () => {
  error.value = '';
  success.value = '';
  submitting.value = true;
  try {
    await signup(email.value, password.value);
    success.value = 'Account created. You can now log in.';
    setTimeout(() => router.push('/login'), 1000);
  } catch (err) {
    error.value = err.response?.data?.detail || 'Unable to sign up';
  } finally {
    submitting.value = false;
  }
};
</script>
