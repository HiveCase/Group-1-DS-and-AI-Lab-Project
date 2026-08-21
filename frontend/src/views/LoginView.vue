<template>
  <div class="page auth-page">
    <div class="card auth-card">
      <h1 class="section-title">Log in</h1>
      <p class="muted">Access your Claims Portal account.</p>

      <form @submit.prevent="handleLogin">
        <label>
          Email
          <InputText v-model="email" type="email" placeholder="you@example.com" required />
        </label>
        <label>
          Password
          <Password v-model="password" :feedback="false" toggleMask placeholder="Password" required />
        </label>
        <Button type="submit" label="Log in" :loading="submitting" />
      </form>

      <Message v-if="error" severity="error">{{ error }}</Message>

      <p class="muted">
        No account yet?
        <router-link to="/signup">Sign up</router-link>
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import InputText from 'primevue/inputtext';
import Password from 'primevue/password';
import Button from 'primevue/button';
import Message from 'primevue/message';
import { login } from '../services/auth';

const route = useRoute();
const router = useRouter();

const email = ref('');
const password = ref('');
const error = ref('');
const submitting = ref(false);

const handleLogin = async () => {
  error.value = '';
  submitting.value = true;
  try {
    await login(email.value, password.value);
    router.push(typeof route.query.redirect === 'string' ? route.query.redirect : '/');
  } catch (err) {
    error.value = err.response?.data?.detail || 'Unable to log in';
  } finally {
    submitting.value = false;
  }
};
</script>
