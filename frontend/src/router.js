import { createRouter, createWebHistory } from 'vue-router';
import LandingView from './views/LandingView.vue';
import ClaimantView from './views/ClaimantView.vue';
import AdjusterView from './views/AdjusterView.vue';
import SIUView from './views/SIUView.vue';
import SupervisorView from './views/SupervisorView.vue';
import LoginView from './views/LoginView.vue';
import SignupView from './views/SignupView.vue';

export const routes = [
  { path: '/login', name: 'login', component: LoginView, meta: { label: 'Login', auth: false } },
  { path: '/signup', name: 'signup', component: SignupView, meta: { label: 'Sign Up', auth: false } },
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

router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('access_token');
  if (to.meta.auth === false) {
    // Auth pages (login/signup) — accessible without token
    next();
  } else if (!token) {
    // Protected pages — redirect to login if no token
    next({ name: 'login' });
  } else {
    next();
  }
});

export default router;
