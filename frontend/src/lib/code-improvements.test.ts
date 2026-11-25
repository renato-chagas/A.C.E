import { afterEach, describe, expect, it, vi } from 'vitest';
import { createCodeImprovementsClient, type CodeImprovementSummary } from './code-improvements';

const connection = vi.hoisted(() => ({ base: 'http://127.0.0.1:8000', desktop: true }));
vi.mock('./api', () => ({
  getBase: () => connection.base,
  isTauri: () => connection.desktop,
  authHeaders: (extra = {}) => ({ Authorization: 'Bearer local-fixture', ...extra }),
}));

const summary: CodeImprovementSummary = {
  proposal_id: '123abc456def',
  title: 'Improve local validation',
  status: 'applied',
  files: ['src/read.py'],
  created_at: 1_700_000_000,
  rollback_available: true,
};
const proposal = {
  ...summary,
  repository: 'A:/fixture',
  patch: '-old\\n+new',
  validation: 'Tests passed.',
  validation_receipt: 'a'.repeat(64),
};
const draft = {
  repository: proposal.repository,
  task: 'Melhore a validação.',
  files: summary.files,
};
const { signal } = new AbortController();

afterEach(() => {
  connection.base = 'http://127.0.0.1:8000';
  connection.desktop = true;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('local code improvement review', () => {
  it('lists and fetches details through read-only local requests that cannot redirect', async () => {
    connection.base = 'http://[::1]:8765';
    const fetch = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json({ proposals: [summary] }))
      .mockResolvedValueOnce(Response.json(proposal));
    const api = createCodeImprovementsClient()!;
    connection.base = 'https://remote.example';

    await expect(api.list(signal)).resolves.toEqual([summary]);
    await expect(api.detail(summary.proposal_id, signal)).resolves.toEqual(proposal);

    const options = expect.objectContaining({
      method: 'GET',
      signal,
      cache: 'no-store',
      redirect: 'error',
      referrerPolicy: 'no-referrer',
      headers: { Authorization: 'Bearer local-fixture' },
    });
    const endpoint = 'http://[::1]:8765/v1/code-improvements';
    expect(fetch).toHaveBeenNthCalledWith(1, endpoint + '?limit=50', options);
    expect(fetch).toHaveBeenNthCalledWith(2, endpoint + '/' + summary.proposal_id, options);
  });

  it('sends the requested draft only to the direct local API and reads the saved proposal', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValue(Response.json({ ...proposal, status: 'proposed' }, { status: 201 }));
    const api = createCodeImprovementsClient()!;
    connection.base = 'https://remote.example';

    await expect(api.generate(draft, signal)).resolves.toEqual({ ...proposal, status: 'proposed' });
    expect(fetch).toHaveBeenCalledExactlyOnceWith('http://127.0.0.1:8000/v1/code-improvements', {
      method: 'POST',
      headers: { Authorization: 'Bearer local-fixture', 'Content-Type': 'application/json' },
      body: JSON.stringify(draft),
      cache: 'no-store',
      credentials: 'omit',
      redirect: 'error',
      referrerPolicy: 'no-referrer',
      signal,
    });
  });

  it.each([
    [409, 'A proposal is already being generated'],
    [422, 'Check the Git repository'],
    [503, 'Select a running local model'],
  ])('reports proposal generation error %i without exposing the response body', async (status, message) => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValue(Response.json({ detail: 'Private details' }, { status }));
    await expect(createCodeImprovementsClient()!.generate(draft, signal)).rejects.toThrow(message);
  });

  it.each(['https://remote.example', 'http://127.0.0.1.remote.example', '/api'])(
    'refuses unsafe API bases %s',
    (base) => {
      connection.base = base;
      const fetch = vi.spyOn(globalThis, 'fetch');
      expect(createCodeImprovementsClient()).toBeNull();
      expect(fetch).not.toHaveBeenCalled();
    },
  );

  it('requires a local browser origin as well as a local API', () => {
    connection.desktop = false;
    vi.stubGlobal('window', { location: { origin: 'https://remote.example' } });
    expect(createCodeImprovementsClient()).toBeNull();
  });

  it('rejects invalid command or path text before a request', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch');
    await expect(createCodeImprovementsClient()!.detail('../unsafe; echo', signal))
      .rejects.toThrow('Invalid proposal ID');
    expect(fetch).not.toHaveBeenCalled();
  });

  it('rejects a reply about another proposal', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json(proposal));
    await expect(createCodeImprovementsClient()!.detail('000000000000', signal))
      .rejects.toThrow('invalid proposal');
  });

  it('does not expose private server errors', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('Private details', { status: 404 }));
    await expect(createCodeImprovementsClient()!.detail(summary.proposal_id, signal))
      .rejects.toThrow('This proposal is unavailable');
  });
});
