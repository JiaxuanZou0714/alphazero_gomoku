/* Service worker for the AlphaZero Gomoku static app.
 * Cache-first for the app shell and everything under assets/ (onnxruntime
 * wasm/js, d3, and the chunked model parts). Bump CACHE_VERSION to roll the
 * cache when the deployed assets change. */
const CACHE_VERSION = "az-gomoku-v1";

// Resolve relative to the worker scope so it works under the GitHub Pages
// subpath (e.g. /alphazero_gomoku/).
const SCOPE = self.registration ? self.registration.scope : self.location.href;
const rel = (path) => new URL(path, SCOPE).href;

const APP_SHELL = [
  rel("./"),
  rel("./index.html"),
  rel("./app.js"),
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

function isCacheable(request) {
  if (request.method !== "GET") return false;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return false;
  if (url.pathname.includes("/assets/")) return true;
  return APP_SHELL.includes(url.href.split("?")[0]);
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (!isCacheable(request)) return;

  event.respondWith(
    caches.open(CACHE_VERSION).then(async (cache) => {
      const cached = await cache.match(request, { ignoreSearch: true });
      if (cached) return cached;
      const response = await fetch(request);
      if (response && response.ok) {
        cache.put(request, response.clone()).catch(() => undefined);
      }
      return response;
    }),
  );
});
