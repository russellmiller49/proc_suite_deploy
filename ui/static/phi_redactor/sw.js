// This service worker previously cached assets, but that can cause "stuck" behavior during rapid iteration.
// Self-unregister to ensure the UI always loads the latest scripts/styles.

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.map((k) => caches.delete(k)));
      await self.clients.claim();
      await self.registration.unregister();
    })()
  );
});

self.addEventListener("fetch", () => {
  // no-op
});
