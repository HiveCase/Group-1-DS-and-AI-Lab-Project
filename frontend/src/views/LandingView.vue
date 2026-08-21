<template>
  <div class="page">
    <div class="hero-card">
      <div>
        <p class="eyebrow">Claims platform</p>
        <h1>Choose a portal</h1>
        <p>Move from claim intake to review, investigation, and oversight with a consistent experience.</p>
      </div>
      <div class="hero-badge">Four-role workflow</div>
    </div>

    <div class="portal-grid">
      <router-link class="portal-card accent-claimant" to="/claimant">
        <Tag value="Claimant" severity="info" />
        <h2>Claimant</h2>
        <p>File claims, track status, and manage evidence.</p>
      </router-link>
      <router-link class="portal-card accent-adjuster" to="/adjuster">
        <Tag value="Adjuster" severity="success" />
        <h2>Adjuster</h2>
        <p>Review submissions, inspect AI findings, and decide.</p>
      </router-link>
      <router-link class="portal-card accent-siu" to="/siu">
        <Tag value="SIU" severity="warn" />
        <h2>SIU</h2>
        <p>Investigate high-risk claims and suspicious trends.</p>
      </router-link>
      <router-link class="portal-card accent-supervisor" to="/supervisor">
        <Tag value="Supervisor" severity="warn" />
        <h2>Supervisor</h2>
        <p>Monitor portfolio health and AI performance.</p>
      </router-link>
    </div>
  </div>
</template>

<script setup>
import Tag from 'primevue/tag';
</script>
