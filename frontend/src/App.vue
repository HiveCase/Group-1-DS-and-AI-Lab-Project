<template>
  <!-- Auth pages: full-page, no sidebar -->
  <router-view v-if="isAuthPage" />

  <!-- Portal pages: sidebar shell -->
  <div v-else class="app-shell" :style="{ '--accent': accentColor }">
    <aside class="sidebar">
      <div class="brand">
        <span class="pi pi-shield" />
        <h2>Claims Portal</h2>
      </div>
      <nav>
        <router-link to="/" class="nav-link"><span class="pi pi-home" /> Home</router-link>
        <router-link to="/claimant" class="nav-link"><span class="pi pi-file" /> Claimant</router-link>
        <router-link to="/adjuster" class="nav-link"><span class="pi pi-verified" /> Adjuster</router-link>
        <router-link to="/siu" class="nav-link"><span class="pi pi-search" /> SIU</router-link>
        <router-link to="/supervisor" class="nav-link"><span class="pi pi-chart-bar" /> Supervisor</router-link>
      </nav>
      <div class="sidebar-footer">
        <div v-if="currentUser" class="user-info">
          <span class="pi pi-user" />
          <span class="user-name">{{ currentUser.full_name }}</span>
        </div>
        <button class="logout-btn" @click="handleLogout">
          <span class="pi pi-sign-out" />
          Sign Out
        </button>
      </div>
    </aside>
    <div class="shell-body">
      <header class="topbar">
        <span class="topbar-label">{{ portalLabel }}</span>
      </header>
      <main class="main-content">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watchEffect } from 'vue';
import { useRoute, useRouter } from 'vue-router';

const route = useRoute();
const router = useRouter();

const accentMap = {
  claimant: 'var(--portal-claimant)',
  adjuster: 'var(--portal-adjuster)',
  siu: 'var(--portal-siu)',
  supervisor: 'var(--portal-supervisor)',
};

const isAuthPage = computed(() => route.meta.auth === false);
const accentColor = computed(() => accentMap[route.meta.accent] || 'var(--portal-claimant)');
const portalLabel = computed(() => route.meta.label || 'Claims Portal');

const currentUser = ref(null);
watchEffect(() => {
  try {
    const stored = localStorage.getItem('user');
    currentUser.value = stored ? JSON.parse(stored) : null;
  } catch { currentUser.value = null; }
});

function handleLogout() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
  localStorage.removeItem('user');
  router.push('/login');
}
</script>

<style scoped>
.sidebar-footer {
  margin-top: auto;
  padding-top: var(--space-6);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}
.user-info {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  color: rgba(255,255,255,0.7);
  font-size: var(--text-sm);
  padding: var(--space-2) var(--space-3);
}
.user-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.logout-btn {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  background: rgba(255,255,255,0.08);
  border: none;
  color: rgba(255,255,255,0.8);
  padding: var(--space-3);
  border-radius: var(--radius-sm);
  font-size: var(--text-sm);
  font-family: var(--font-family);
  cursor: pointer;
  transition: background 0.2s;
}
.logout-btn:hover { background: rgba(255,255,255,0.14); color: white; }
</style>
