<template>
  <div class="app-shell">
    <aside class="sidebar">
      <h2>Claims Portal</h2>
      <nav>
        <router-link to="/">Home</router-link>
        <router-link to="/claimant">Claimant</router-link>
        <router-link to="/adjuster">Adjuster</router-link>
        <router-link to="/siu">SIU</router-link>
        <router-link to="/supervisor">Supervisor</router-link>
      </nav>
    </aside>
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>
