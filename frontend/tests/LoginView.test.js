import { mount } from '@vue/test-utils';
import { createMemoryHistory } from 'vue-router';
import LoginView from '../src/views/LoginView.vue';
import { createAppRouter } from '../src/router';
import { currentUser } from '../src/services/auth';

describe('LoginView', () => {
  afterEach(() => {
    currentUser.value = null;
  });

  it('renders the login form', async () => {
    const router = createAppRouter(createMemoryHistory());
    router.push('/login');
    await router.isReady();
    const wrapper = mount(LoginView, { global: { plugins: [router] } });
    expect(wrapper.text()).toContain('Log in');
  });
});
