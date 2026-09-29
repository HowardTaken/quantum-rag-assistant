/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves this as a project site at /<repo-name>/, so asset URLs need
// that prefix in production; the local dev server still runs at the root.
export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: command === "build" ? "/quantum-rag-assistant/" : "/",
  server: {
    port: 5173,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/setupTests.ts"],
  },
}));
