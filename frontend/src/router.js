import { createRouter, createWebHistory } from 'vue-router';
import LandingView from './views/LandingView.vue';
import ClaimantView from './views/ClaimantView.vue';
import AdjusterView from './views/AdjusterView.vue';
import SIUView from './views/SIUView.vue';
import SupervisorView from './views/SupervisorView.vue';

export const routes = [
  { path: '/', name: 'landing', component: LandingView },
  { path: '/claimant', name: 'claimant', component: ClaimantView },
  { path: '/adjuster', name: 'adjuster', component: AdjusterView },
  { path: '/siu', name: 'siu', component: SIUView },
  { path: '/supervisor', name: 'supervisor', component: SupervisorView },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
