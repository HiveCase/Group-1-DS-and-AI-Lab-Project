import { mount } from '@vue/test-utils';
import { createMemoryHistory } from 'vue-router';
import SignupView from '../src/views/SignupView.vue';
import { createAppRouter } from '../src/router';
import { currentUser } from '../src/services/auth';

describe('SignupView', () => {
  afterEach(() => {
    currentUser.value = null;
  });

  it('renders the signup form', async () => {
    const router = createAppRouter(createMemoryHistory());
    router.push('/signup');
    await router.isReady();
    const wrapper = mount(SignupView, { global: { plugins: [router] } });
    expect(wrapper.text()).toContain('Sign up');
  });
});
