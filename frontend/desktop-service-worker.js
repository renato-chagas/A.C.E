// Older desktop releases registered /sw.js and could serve a stale app shell.
// Activate this replacement without intercepting any requests.
self.addEventListener('install', (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.matchAll({ type: 'window' }).then((clients) => {
    for (const client of clients) {
      // Navigation must wait for activation to finish, so don't await it here.
      void client.navigate(client.url).catch(() => {});
    }
  }));
});
