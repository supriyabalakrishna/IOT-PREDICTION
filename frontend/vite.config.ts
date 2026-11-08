import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          const normalized = id.replaceAll("\\", "/");
          if (normalized.includes("/node_modules/recharts/") || /\/node_modules\/d3-/.test(normalized)) {
            return "charts";
          }
          if (/\/node_modules\/(react|react-dom|scheduler)\//.test(normalized)) {
            return "framework";
          }
        },
      },
    },
  },
});
