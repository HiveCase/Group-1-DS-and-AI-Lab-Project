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
        <label>
          Role
          <Select v-model="role" :options="roleOptions" optionLabel="label" optionValue="value" placeholder="Select a role" />
        </label>
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
import Select from 'primevue/select';
import Button from 'primevue/button';
import Message from 'primevue/message';
import { signup } from '../services/auth';

const router = useRouter();

const roleOptions = [
  { label: 'User', value: 'user' },
  { label: 'Admin', value: 'admin' },
];

const email = ref('');
const password = ref('');
const role = ref('user');
const error = ref('');
const success = ref('');
const submitting = ref(false);

const handleSignup = async () => {
  error.value = '';
  success.value = '';
  submitting.value = true;
  try {
    await signup(email.value, password.value, role.value);
    success.value = 'Account created. You can now log in.';
    setTimeout(() => router.push('/login'), 1000);
  } catch (err) {
    error.value = err.response?.data?.detail || 'Unable to sign up';
  } finally {
    submitting.value = false;
  }
};
</script>
