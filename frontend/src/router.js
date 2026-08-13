import { createRouter, createWebHistory } from 'vue-router';
import LandingView from './views/LandingView.vue';
import ClaimantView from './views/ClaimantView.vue';
import AdjusterView from './views/AdjusterView.vue';
import SIUView from './views/SIUView.vue';
import SupervisorView from './views/SupervisorView.vue';
import LoginView from './views/LoginView.vue';
import SignupView from './views/SignupView.vue';
import { currentUser } from './services/auth';

export const routes = [
  { path: '/', name: 'landing', component: LandingView, meta: { label: 'Portal selection' } },
  { path: '/claimant', name: 'claimant', component: ClaimantView, meta: { label: 'Claimant', accent: 'claimant' } },
  { path: '/adjuster', name: 'adjuster', component: AdjusterView, meta: { label: 'Adjuster', accent: 'adjuster' } },
  { path: '/siu', name: 'siu', component: SIUView, meta: { label: 'SIU', accent: 'siu' } },
  { path: '/supervisor', name: 'supervisor', component: SupervisorView, meta: { label: 'Supervisor', accent: 'supervisor' } },
  { path: '/login', name: 'login', component: LoginView, meta: { label: 'Log in', public: true } },
  { path: '/signup', name: 'signup', component: SignupView, meta: { label: 'Sign up', public: true } },
];

function authGuard(to) {
  const isPublic = to.meta.public === true;
  if (!isPublic && !currentUser.value) {
    return { path: '/login', query: { redirect: to.fullPath } };
  }
  if (isPublic && currentUser.value) {
    return { path: '/' };
  }
  return true;
}

export function createAppRouter(history) {
  const router = createRouter({ history, routes });
  router.beforeEach(authGuard);
  return router;
}

const router = createAppRouter(createWebHistory());

export default router;
