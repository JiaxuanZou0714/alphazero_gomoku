/* Service worker for the AlphaZero Gomoku static app.
 *
 * Caching strategy (intentionally split so the sha256-keyed IndexedDB model
 * cache in engine.worker.js stays authoritative):
 *   - app shell + vendored libs (onnxruntime wasm/js, d3): cache-first (immutable).
 *   - model manifest.json / catalog.json: network-first with cache fallback, so a
 *     re-exported model's new sha256 is always observed when online (a cache-first
 *     copy would pin the old hash and silently defeat invalidation), while still
 *     working offline.
 *   - model chunk parts (*.onnx.partNN): NOT cached here — engine.worker.js verifies
 *     them against the manifest sha256 and persists the assembled bytes in IndexedDB,
 *     so SW-caching them too would just double-store ~tens of MB per model.
 * Bump CACHE_VERSION to roll the shell/vendor cache when those assets change. */
const CACHE_VERSION = "az-gomoku-v40";

// Resolve relative to the worker scope so it works under the GitHub Pages
// subpath (e.g. /alphazero_gomoku/).
const SCOPE = self.registration ? self.registration.scope : self.location.href;
const rel = (path) => new URL(path, SCOPE).href;

const APP_SHELL = [
  rel("./"),
  rel("./index.html"),
  rel("./app.js"),
  rel("./hero.js"),
  rel("./engine.worker.js"),
  rel("./styles.css"),
];

self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches
      .open(CACHE_VERSION)
      .then((cache) => Promise.allSettled(APP_SHELL.map((url) => cache.add(url))))
      .catch(() => undefined),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((key) => key !== CACHE_VERSION).map((key) => caches.delete(key))),
      )
      .then(() => self.clients.claim())
      .catch(() => undefined),
  );
});

// Decide how (and whether) to serve a request: "cache-first", "network-first",
// or "passthrough" (let the browser handle it, no SW involvement).
function strategyFor(request) {
  if (request.method !== "GET") return "passthrough";
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return "passthrough";
  const path = url.pathname;
  // Content-addressed model chunks live only in IndexedDB (verified bytes).
  if (/\.onnx\.part\d+$/.test(path)) return "passthrough";
  // Model metadata must reflect the latest deploy/export.
  if (path.includes("/assets/models/") && path.endsWith(".json")) return "network-first";
  // Vendored libraries and pre-rendered diagrams are immutable for a deploy.
  if (path.includes("/assets/vendor/")) return "cache-first";
  if (path.includes("/assets/diagrams/")) return "cache-first";
  if (APP_SHELL.includes(url.href.split("?")[0])) return "cache-first";
  return "passthrough";
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_VERSION);
  const cached = await cache.match(request, { ignoreSearch: true });
  if (cached) return cached;
  const response = await fetch(request);
  if (response && response.ok) cache.put(request, response.clone()).catch(() => undefined);
  return response;
}

async function networkFirst(request) {
  const cache = await caches.open(CACHE_VERSION);
  try {
    const response = await fetch(request);
    if (response && response.ok) cache.put(request, response.clone()).catch(() => undefined);
    return response;
  } catch (err) {
    const cached = await cache.match(request, { ignoreSearch: true });
    if (cached) return cached;
    throw err;
  }
}

self.addEventListener("fetch", (event) => {
  const strategy = strategyFor(event.request);
  if (strategy === "passthrough") return;
  event.respondWith(
    strategy === "network-first" ? networkFirst(event.request) : cacheFirst(event.request),
  );
});
