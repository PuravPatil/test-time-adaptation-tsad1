import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The app calls the API at /api. In development Vite forwards /api (HTTP and
// WebSocket) to the FastAPI server, so no CORS setup is needed; in Docker, nginx
// does the same (see nginx.conf). Point it elsewhere with API_TARGET=http://host:port.
const target = process.env.API_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target, changeOrigin: true, ws: true, rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
  preview: {
    port: 4173,
    proxy: {
      "/api": { target, changeOrigin: true, ws: true, rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
});
