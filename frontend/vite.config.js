import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves from https://<user>.github.io/<repo>/ - base must
// match the repo name so built asset URLs resolve correctly there. Root
// "/" is used for local dev/preview (Vite dev server ignores base by
// default in a way that still works either way, but build output needs
// this set explicitly).
export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE_PATH || "/Fantasy-Premier-League-Manager/",
});
