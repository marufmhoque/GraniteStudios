import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base: '/tools/granite-studios/' — the site proxies this path to the
// deployed app, so every asset URL has to be built with that prefix.
export default defineConfig({
  base: '/tools/granite-studios/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
