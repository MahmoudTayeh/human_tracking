/* Design reminder — use a relative base so the built showcase works under a GitHub Pages project path. */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
});
