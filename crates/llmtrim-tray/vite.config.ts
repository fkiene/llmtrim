import { defineConfig, type Plugin } from "vite";

// Strip the `crossorigin` attribute Vite stamps onto the emitted <script> and
// <link rel=stylesheet> tags. On Windows the webview serves the app from
// `http://tauri.localhost`, and the asset-protocol response carries no
// `Access-Control-Allow-Origin`, so a `crossorigin` tag fails the CORS check and
// the JS/CSS never load — a black window. macOS/Linux use the `tauri://`
// opaque-origin scheme, which skips the check, so the attribute only breaks
// Windows. The assets are same-origin and local, so dropping it is safe.
function stripCrossorigin(): Plugin {
  return {
    name: "strip-crossorigin",
    transformIndexHtml(html) {
      return html.replace(/ crossorigin/g, "");
    },
  };
}

// Frontend for the llmtrim tray popover.
//
// CSP-hardening choices (the app CSP is `script-src 'self'`, `connect-src 'none'`):
//   - `base: "./"` emits relative asset URLs so the Tauri asset protocol resolves
//     them; no absolute http(s) origins land in the HTML.
//   - `modulePreload.polyfill: false` stops Vite injecting its inline
//     `<script type="module">` polyfill — an inline script would violate
//     `script-src 'self'`. Tauri's webviews support modulepreload natively.
//   - `assetsInlineLimit: 0` keeps assets as real files rather than data: URIs.
export default defineConfig({
  root: __dirname,
  base: "./",
  plugins: [stripCrossorigin()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2022",
    modulePreload: { polyfill: false },
    assetsInlineLimit: 0,
    sourcemap: false,
    // Stable, content-hash-free output names. dist/ is committed (Homebrew builds
    // the tray from the source tarball with no Node step, and tauri-build embeds
    // dist/ at compile time), so unhashed names keep the committed diff to the
    // actual content change instead of a renamed file every build.
    rollupOptions: {
      output: {
        entryFileNames: "assets/index.js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/index[extname]",
      },
    },
  },
  clearScreen: false,
});
