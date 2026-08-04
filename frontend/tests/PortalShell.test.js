import { mount } from '@vue/test-utils';
import App from '../src/App.vue';
import { createRouter, createMemoryHistory } from 'vue-router';
import { routes } from '../src/router';

const router = createRouter({
  history: createMemoryHistory(),
  routes,
});

describe('App shell', () => {
  it('renders the portal entry points', async () => {
    const wrapper = mount(App, {
      global: {
        plugins: [router],
      },
    });
    await router.push('/');
    await router.isReady();
    expect(wrapper.text()).toContain('Claims Portal');
  });
});
