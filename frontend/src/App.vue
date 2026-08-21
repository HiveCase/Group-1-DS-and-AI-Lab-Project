<template>
  <div v-if="route.meta.public" class="auth-shell">
    <router-view />
  </div>
  <div v-else class="app-shell" :style="{ '--accent': accentColor }">
    <aside class="sidebar">
      <div class="brand">
        <span class="pi pi-shield" />
        <h2>Claims Portal</h2>
      </div>
      <nav>
        <router-link to="/" class="nav-link"><span class="pi pi-home" /> Home</router-link>
        <router-link to="/claimant" class="nav-link"><span class="pi pi-file" /> Claimant</router-link>
        <template v-if="isAdmin">
          <router-link to="/adjuster" class="nav-link"><span class="pi pi-verified" /> Adjuster</router-link>
          <router-link to="/siu" class="nav-link"><span class="pi pi-search" /> SIU</router-link>
          <router-link to="/supervisor" class="nav-link"><span class="pi pi-chart-bar" /> Supervisor</router-link>
        </template>
      </nav>
    </aside>
    <div class="shell-body">
      <header class="topbar">
        <span class="topbar-label">{{ portalLabel }}</span>
        <div class="topbar-auth">
          <span class="topbar-user">{{ currentUser?.email }} ({{ currentUser?.role }})</span>
          <button class="link-button" @click="handleLogout">Log out</button>
        </div>
      </header>
      <main class="main-content">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { currentUser, isAdmin, logout } from './services/auth';

const route = useRoute();
const router = useRouter();

const accentMap = {
  claimant: 'var(--portal-claimant)',
  adjuster: 'var(--portal-adjuster)',
  siu: 'var(--portal-siu)',
  supervisor: 'var(--portal-supervisor)',
};

const accentColor = computed(() => accentMap[route.meta.accent] || 'var(--portal-claimant)');
const portalLabel = computed(() => route.meta.label || 'Claims Portal');

const handleLogout = () => {
  logout();
  router.push('/login');
};
</script>
