import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchMemoryFacts, reviseMemoryFact } from './local-facts';

const config = vi.hoisted(() => ({ base: '' }));
vi.mock('./api', () => ({
  getBase: () => config.base,
  authHeaders: (headers: Record<string, string> = {}) => ({
    ...headers,
    Authorization: 'Bearer local-fixture-key',
  }),
}));

afterEach(() => {
  config.base = '';
  vi.restoreAllMocks();
});

describe('local Memory Facts requests', () => {
  it('requests facts from the local API without caching or redirects', async () => {
    const facts = [{
      text: 'ACE uses a paper notebook.',
      source: 'chat',
      trust: 'auto',
      superseded: false,
      trusted_for_recall: true,
    }];
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json({ facts }));
    const { signal } = new AbortController();
    await expect(fetchMemoryFacts(signal)).resolves.toEqual(facts);
    expect(fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/v1/memory/facts',
      expect.objectContaining({ signal, cache: 'no-store', redirect: 'error' }),
    );
  });

  it('sends a correction and exact old text only to the configured loopback address', async () => {
    config.base = 'http://[::1]:8765/';
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(null, { status: 204 }));
    const { signal } = new AbortController();
    await reviseMemoryFact('Old statement.', 'New statement.', signal);
    expect(fetch).toHaveBeenCalledWith(
      'http://[::1]:8765/v1/memory/facts/revise',
      expect.objectContaining({
        method: 'POST',
        cache: 'no-store',
        redirect: 'error',
        signal,
        body: JSON.stringify({ existing_fact: 'Old statement.', correction: 'New statement.' }),
        headers: {
          Authorization: 'Bearer local-fixture-key',
          'Content-Type': 'application/json',
        },
      }),
    );
  });

  it.each(['https://remote.example', 'http://127.0.0.1.remote.example', 'http://alice@127.0.0.1'])(
    'does not send text to an unapproved API URL %s',
    async (base) => {
      config.base = base;
      const fetch = vi.spyOn(globalThis, 'fetch');
      const { signal } = new AbortController();
      await expect(reviseMemoryFact('Old statement.', 'New statement.', signal))
        .rejects.toThrow('requires a local ACE API URL');
      expect(fetch).not.toHaveBeenCalled();
    },
  );

  it('shows a safe conflict message without exposing the response body', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('Private details', { status: 409 }),
    );
    await expect(reviseMemoryFact('Old statement.', 'New statement.', new AbortController().signal))
      .rejects.toThrow('Refresh the list');
  });
});
