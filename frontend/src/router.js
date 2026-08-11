import { createRouter, createWebHistory } from 'vue-router';
import LandingView from './views/LandingView.vue';
import ClaimantView from './views/ClaimantView.vue';
import AdjusterView from './views/AdjusterView.vue';
import SIUView from './views/SIUView.vue';
import SupervisorView from './views/SupervisorView.vue';

export const routes = [
  { path: '/', name: 'landing', component: LandingView, meta: { label: 'Portal selection' } },
  { path: '/claimant', name: 'claimant', component: ClaimantView, meta: { label: 'Claimant', accent: 'claimant' } },
  { path: '/adjuster', name: 'adjuster', component: AdjusterView, meta: { label: 'Adjuster', accent: 'adjuster' } },
  { path: '/siu', name: 'siu', component: SIUView, meta: { label: 'SIU', accent: 'siu' } },
  { path: '/supervisor', name: 'supervisor', component: SupervisorView, meta: { label: 'Supervisor', accent: 'supervisor' } },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
