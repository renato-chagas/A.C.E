import { authHeaders } from './api';
import { resolveLoopbackApiOrigin } from './loopback-api';
import { safeExternalUrl } from './knowledge-graph';

export const WEB_LIBRARY_QUERY_LIMIT = 2048;

export type SavedWebSource = {
  url: string;
  title: string;
  saved_at: string;
  text: string;
  capture: 'page' | 'snippet' | 'unknown';
  revision: string;
};

type WebSearchResult = {
  sources: SavedWebSource[];
  offline_fallback: boolean;
};

export type WebGraphExtraction = {
  url: string;
  saved_at: string;
  entities_stored: number;
  relations_stored: number;
};

type ArchivedSource = Omit<SavedWebSource, 'revision'> & { revision?: string };

function isSource(value: unknown): value is ArchivedSource {
  if (!value || typeof value !== 'object') return false;
  const source = value as Partial<SavedWebSource>;
  return typeof source.url === 'string' &&
    typeof source.title === 'string' &&
    typeof source.saved_at === 'string' &&
    typeof source.text === 'string' &&
    (source.revision === undefined || typeof source.revision === 'string') &&
    (source.capture === 'page' || source.capture === 'snippet' || source.capture === 'unknown');
}

function isStoredCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

export function canExtractWebSource(source: SavedWebSource): boolean {
  return source.capture === 'page' && !!source.revision.trim() && !!safeExternalUrl(source.url);
}

export function createWebLibraryClient() {
  const origin = resolveLoopbackApiOrigin();
  if (!origin) return null;
  const endpoint = origin + '/v1/web-library';

  async function fetchLocal(path: string, signal: AbortSignal, body?: object) {
    try {
      return await fetch(endpoint + path, {
        method: body ? 'POST' : 'GET',
        headers: body ? authHeaders({ 'Content-Type': 'application/json' }) : authHeaders(),
        body: body ? JSON.stringify(body) : undefined,
        cache: 'no-store',
        credentials: 'omit',
        redirect: 'error',
        referrerPolicy: 'no-referrer',
        signal,
      });
    } catch (error) {
      if (signal.aborted) throw error;
      throw new Error('Could not reach the local ACE server.');
    }
  }

  async function request(query: string, online: boolean, signal: AbortSignal) {
    const term = query.trim();
    if (term.length > WEB_LIBRARY_QUERY_LIMIT || (online && !term)) {
      throw new Error('Enter a topic or a public page URL.');
    }
    const params = !online && term ? '?' + new URLSearchParams({ q: term }).toString() : '';
    const response = await fetchLocal(params, signal, online ? { query: term } : undefined);

    if (response.status === 401) throw new Error('The local ACE connection needs a valid API key.');
    if (response.status === 403) throw new Error('Web research requires direct access from this computer.');
    if (response.status === 409 || response.status === 429) throw new Error('Web research is busy. Try again in a moment.');
    if (response.status === 422) throw new Error('Check the topic or public page URL and try again.');
    if (!response.ok) {
      throw new Error(online ? 'Could not save web sources. You can still search saved sources.' : 'Could not read saved web sources.');
    }

    const data: Partial<WebSearchResult> | null = await response.json().catch(() => null);
    if (!Array.isArray(data?.sources) || !data.sources.every(isSource) ||
        (online && typeof data.offline_fallback !== 'boolean')) {
      throw new Error('ACE returned an invalid web research response.');
    }
    return {
      sources: data.sources.map((source) => ({ ...source, revision: source.revision ?? '' })),
      offline_fallback: online && data.offline_fallback === true,
    };
  }

  return {
    async searchSaved(query: string, signal: AbortSignal): Promise<SavedWebSource[]> {
      return (await request(query, false, signal)).sources;
    },
    searchAndSave: (query: string, signal: AbortSignal): Promise<WebSearchResult> =>
      request(query, true, signal),
    async extractToGraph(source: SavedWebSource, signal: AbortSignal): Promise<WebGraphExtraction> {
      const { url, revision } = source;
      if (!canExtractWebSource(source)) throw new Error('Choose an archived page with saved text.');

      const response = await fetchLocal('/extract', signal, { url, revision });
      if (response.status === 401) throw new Error('The local ACE connection needs a valid API key.');
      if (response.status === 403) throw new Error('Graph extraction requires direct access from this computer.');
      if (response.status === 404) throw new Error('This page is no longer saved. Search saved sources again.');
      if (response.status === 409) throw new Error('This saved page changed. Search saved sources again and retry.');
      if (response.status === 429) throw new Error('Graph extraction is busy. Try again in a moment.');
      if (response.status === 422) throw new Error('This saved page cannot be extracted.');
      if (response.status === 503) throw new Error('Graph extraction is unavailable. Check the local model and try again.');
      if (!response.ok) throw new Error('Could not extract this saved page to the graph.');

      const data: Partial<WebGraphExtraction> | null = await response.json().catch(() => null);
      if (data?.url !== url || data.saved_at !== source.saved_at ||
          !isStoredCount(data.entities_stored) || !isStoredCount(data.relations_stored)) {
        throw new Error('ACE returned an invalid graph extraction response.');
      }
      return {
        url: data.url,
        saved_at: data.saved_at,
        entities_stored: data.entities_stored,
        relations_stored: data.relations_stored,
      };
    },
  };
}
