import { mount } from '@vue/test-utils';
import { createMemoryHistory } from 'vue-router';
import App from '../src/App.vue';
import { createAppRouter } from '../src/router';
import { currentUser } from '../src/services/auth';

describe('App shell', () => {
  afterEach(() => {
    currentUser.value = null;
  });

  it('redirects an anonymous visitor to the login page', async () => {
    currentUser.value = null;
    const router = createAppRouter(createMemoryHistory());
    const wrapper = mount(App, { global: { plugins: [router] } });
    await router.push('/');
    await router.isReady();
    expect(router.currentRoute.value.path).toBe('/login');
    expect(wrapper.text()).toContain('Log in');
  });

  it('renders the portal entry points for a logged-in user', async () => {
    currentUser.value = { email: 'demo@example.com', role: 'user' };
    const router = createAppRouter(createMemoryHistory());
    const wrapper = mount(App, { global: { plugins: [router] } });
    await router.push('/');
    await router.isReady();
    expect(wrapper.text()).toContain('Claims Portal');
    expect(wrapper.text()).toContain('demo@example.com');
  });
});
