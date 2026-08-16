import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Dev server proxies the WebSocket endpoint to the Rust gateway; the
// built bundle is served by the gateway itself (ServeDir on /).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/ws': {
        target: 'ws://127.0.0.1:8080',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 600,
  },
});
