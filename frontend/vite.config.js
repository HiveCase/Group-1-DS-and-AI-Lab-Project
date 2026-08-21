import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

// 127.0.0.1 is correct for bare-metal dev, where the frontend and backend
// are sibling processes sharing one loopback interface. It's wrong inside
// Docker Compose, where each service is a separate container with its own
// network namespace -- there, the backend must be reached by its Compose
// service name (Docker's internal DNS resolves it), which docker-compose.yml
// supplies via VITE_BACKEND_PROXY_TARGET.
const backendTarget = process.env.VITE_BACKEND_PROXY_TARGET || 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.js'],
  },
});
