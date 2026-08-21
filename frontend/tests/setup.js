import { config } from '@vue/test-utils';
import PrimeVue from 'primevue/config';
import Aura from '@primevue/themes/aura';

// Registered globally so every mount() in the suite has PrimeVue's
// injected config available, matching what main.js sets up for the real app.
config.global.plugins.push([PrimeVue, { theme: { preset: Aura, options: { darkModeSelector: false } } }]);

// jsdom has no matchMedia implementation; PrimeVue's DatePicker/overlay
// components use it for responsive breakpoints during mount.
if (!window.matchMedia) {
  window.matchMedia = (query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  });
}
