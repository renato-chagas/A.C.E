import { afterEach, describe, expect, it, vi } from 'vitest';
import { createWebLibraryClient, WEB_LIBRARY_QUERY_LIMIT } from './web-library';

const connection = vi.hoisted(() => ({ base: 'http://127.0.0.1:8000', desktop: true }));
vi.mock('./api', () => ({
  getBase: () => connection.base,
  isTauri: () => connection.desktop,
  authHeaders: (extra = {}) => ({ Authorization: 'Bearer fixture', ...extra }),
}));

const source = {
  title: 'Known source',
  url: 'https://example.org/research',
  saved_at: '2026-11-23T09:15:00Z',
  text: 'A retained finding',
  capture: 'page',
  revision: 'archive-snapshot-id',
} as const;
const { signal } = new AbortController();

afterEach(() => {
  connection.base = 'http://127.0.0.1:8000';
  connection.desktop = true;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('model-independent web research client', () => {
  it('distinguishes browsing and offline queries from the explicit internet action', async () => {
    const fetchLocal = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json({ sources: [source] }))
      .mockResolvedValueOnce(Response.json({ sources: [source] }))
      .mockResolvedValueOnce(Response.json({ sources: [source], offline_fallback: true }));
    const client = createWebLibraryClient()!;

    await expect(client.searchSaved('', signal)).resolves.toEqual([source]);
    await expect(client.searchSaved('  role=admin&find=local  ', signal)).resolves.toEqual([source]);
    await expect(client.searchAndSave('  https://example.org/research  ', signal)).resolves.toEqual({
      sources: [source],
      offline_fallback: true,
    });
    const endpoint = 'http://127.0.0.1:8000/v1/web-library';
    const common = {
      cache: 'no-store',
      credentials: 'omit',
      redirect: 'error',
      referrerPolicy: 'no-referrer',
      signal,
    };
    expect(fetchLocal).toHaveBeenNthCalledWith(1, endpoint, expect.objectContaining({
      ...common, method: 'GET', headers: { Authorization: 'Bearer fixture' }, body: undefined,
    }));
    expect(fetchLocal).toHaveBeenNthCalledWith(
      2,
      endpoint + '?q=role%3Dadmin%26find%3Dlocal',
      expect.objectContaining({ ...common, method: 'GET', body: undefined }),
    );
    expect(fetchLocal).toHaveBeenNthCalledWith(3, endpoint, expect.objectContaining({
      ...common,
      method: 'POST',
      headers: { Authorization: 'Bearer fixture', 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'https://example.org/research' }),
    }));
  });

  it.each(['https://remote.example', 'http://localhost.remote.example', 'http://localhost/v1'])(
    'refuses the non-local API target %s',
    (base) => {
      const fetchNetworks = vi.spyOn(globalThis, 'fetch');
      connection.base = base;
      expect(createWebLibraryClient()).toBeNull();
      expect(fetchNetworks).not.toHaveBeenCalled();
    },
  );

  it('rejects a remotely hosted browser shell even with a local configured API', () => {
    connection.desktop = false;
    vi.stubGlobal('window', { location: { origin: 'https://remote.example' } });
    expect(createWebLibraryClient()).toBeNull();
  });

  it('never sends a blank online query or oversized input', async () => {
    const fetchLocal = vi.spyOn(globalThis, 'fetch');
    const client = createWebLibraryClient()!;
    await expect(client.searchAndSave('  ', signal)).rejects.toThrow('Enter a topic');
    await expect(client.searchSaved('a'.repeat(WEB_LIBRARY_QUERY_LIMIT + 1), signal)).rejects.toThrow('Enter a topic');
    expect(fetchLocal).not.toHaveBeenCalled();
  });

  it('validates returned provenance and masks server error details', async () => {
    const client = createWebLibraryClient()!;
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json({ sources: [{ ...source, capture: 'other' }] }))
      .mockResolvedValueOnce(Response.json({ sources: [source] }))
      .mockResolvedValueOnce(Response.json({ detail: 'C:/private/profile.db' }, { status: 503 }));

    await expect(client.searchSaved('ACE', signal)).rejects.toThrow('invalid web research response');
    await expect(client.searchAndSave('ACE', signal)).rejects.toThrow('invalid web research response');
    await expect(client.searchAndSave('ACE', signal)).rejects.toThrow('Could not save web sources.');
  });
});

describe('saved page graph extraction client', () => {
  const result = {
    url: source.url,
    saved_at: source.saved_at,
    entities_stored: 2,
    relations_stored: 1,
  };

  it('sends only the selected page revision and URL to the local server', async () => {
    const fetchLocal = vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json(result));
    const client = createWebLibraryClient()!;

    await expect(client.extractToGraph(source, signal)).resolves.toEqual(result);
    expect(fetchLocal).toHaveBeenCalledExactlyOnceWith(
      'http://127.0.0.1:8000/v1/web-library/extract',
      {
        method: 'POST',
        headers: { Authorization: 'Bearer fixture', 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: source.url, revision: source.revision }),
        cache: 'no-store',
        credentials: 'omit',
        redirect: 'error',
        referrerPolicy: 'no-referrer',
        signal,
      },
    );
  });

  it('keeps older archives readable and requires a saved page revision for extraction', async () => {
    const { revision, ...legacySource } = source;
    const fetchLocal = vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json({ sources: [legacySource] }));
    const client = createWebLibraryClient()!;

    const saved = await client.searchSaved('', signal);
    expect(saved).toEqual([{ ...legacySource, revision: '' }]);
    await expect(client.extractToGraph(saved[0]!, signal)).rejects.toThrow('Choose an archived page');
    await expect(client.extractToGraph({
      ...legacySource, revision, capture: 'snippet',
    }, signal)).rejects.toThrow('Choose an archived page');
    expect(fetchLocal).toHaveBeenCalledTimes(1);
  });

  it('asks the user to search again if the page changed and validates graph results', async () => {
    const fetchLocal = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json({ detail: 'private server detail' }, { status: 409 }))
      .mockResolvedValueOnce(Response.json({ ...result, saved_at: '2026-11-24T09:15:00Z' }))
      .mockResolvedValueOnce(Response.json({ ...result, relations_stored: -1 }));
    const client = createWebLibraryClient()!;

    await expect(client.extractToGraph(source, signal)).rejects.toThrow(
      'This saved page changed. Search saved sources again and retry.',
    );
    expect(fetchLocal).toHaveBeenCalledTimes(1);
    await expect(client.extractToGraph(source, signal)).rejects.toThrow('invalid graph extraction response');
    await expect(client.extractToGraph(source, signal)).rejects.toThrow('invalid graph extraction response');
  });
});
