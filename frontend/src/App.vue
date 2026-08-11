<template>
  <div class="app-shell" :style="{ '--accent': accentColor }">
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
import { computed } from 'vue';
import { useRoute } from 'vue-router';

const route = useRoute();

const accentMap = {
  claimant: 'var(--portal-claimant)',
  adjuster: 'var(--portal-adjuster)',
  siu: 'var(--portal-siu)',
  supervisor: 'var(--portal-supervisor)',
};

const accentColor = computed(() => accentMap[route.meta.accent] || 'var(--portal-claimant)');
const portalLabel = computed(() => route.meta.label || 'Claims Portal');
</script>
