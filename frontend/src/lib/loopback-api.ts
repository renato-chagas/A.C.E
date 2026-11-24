import { getBase, isTauri } from './api';

function loopbackUrl(value: string): URL | null {
  try {
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password ||
      url.port === '0' || url.search || url.hash) {
      return null;
    }
    const host = url.hostname;
    return host === 'localhost' || host === '[::1]' || /^127\.(?:\d+\.){2}\d+$/.test(host)
      ? url
      : null;
  } catch {
    return null;
  }
}

export function resolveLoopbackApiOrigin(): string | null {
  // A local Vite or HTTP proxy can forward /v1 to a remote service. Use a
  // direct verified loopback origin, even when the browser API setting is blank.
  if (!isTauri() && (typeof window === 'undefined' || !loopbackUrl(window.location.origin))) {
    return null;
  }
  const base = loopbackUrl(getBase() || 'http://127.0.0.1:8000');
  if (!base || base.pathname !== '/') return null;
  return base.origin;
}
