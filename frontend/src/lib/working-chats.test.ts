import { afterEach, describe, expect, it, vi } from 'vitest';
import { createWorkingChatsClient, WorkingChatError, type WorkingChat } from './working-chats';

const config = vi.hoisted(() => ({ base: '', desktop: true }));
vi.mock('./api', () => ({
  getBase: () => config.base,
  isTauri: () => config.desktop,
  authHeaders: (headers: Record<string, string> = {}) => ({
    ...headers,
    Authorization: 'Bearer fixture',
  }),
}));

const snapshot: WorkingChat = {
  name: 'ACE notes',
  session_id: 'a'.repeat(32),
  revision: 2,
  turns: [['A private note.', 'A private reply.']],
};
const { signal } = new AbortController();

afterEach(() => {
  config.base = '';
  config.desktop = true;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('Shared Chats local client', () => {
  it('validates the browser origin and bypasses the generic proxy on a local preview', async () => {
    config.desktop = false;
    vi.stubGlobal('window', { location: { origin: 'http://localhost:5173' } });
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json({ chats: [snapshot.name] }));
    await expect(createWorkingChatsClient()?.list(signal)).resolves.toEqual([snapshot.name]);
    expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:8000/v1/working-chats', expect.objectContaining({
      method: 'GET',
      cache: 'no-store',
      credentials: 'omit',
      redirect: 'error',
      referrerPolicy: 'no-referrer',
      signal,
    }));
  });

  it.each([
    'https://remote.example',
    'http://127.0.0.1.remote.example',
    'http://user@127.0.0.1',
    'ftp://localhost',
    'http://localhost:8000/generic-proxy',
    'http://localhost:0',
  ])('rejects a destination that is not a direct loopback API %s', (base) => {
    config.base = base;
    const fetch = vi.spyOn(globalThis, 'fetch');
    expect(createWorkingChatsClient()).toBeNull();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('rejects a remote browser page even if configured with a loopback API', () => {
    config.desktop = false;
    config.base = 'http://127.0.0.1:8000';
    vi.stubGlobal('window', { location: { origin: 'https://remote.example' } });
    expect(createWorkingChatsClient()).toBeNull();
  });

  it('uses one pinned loopback endpoint for reading, opening, sending and deleting', async () => {
    config.base = 'http://[::1]:8765';
    const client = createWorkingChatsClient()!;
    const updated = { ...snapshot, revision: 3, turns: [...snapshot.turns, ['Follow-up', 'New reply']] };
    const fetch = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json(snapshot))
      .mockResolvedValueOnce(Response.json(snapshot))
      .mockResolvedValueOnce(Response.json({ chat: updated, answer: 'New reply', saved: true }))
      .mockResolvedValueOnce(Response.json({ deleted: true }));
    config.base = 'https://remote.example';

    await expect(client.read(snapshot.name, signal)).resolves.toEqual(snapshot);
    await expect(client.open(snapshot.name, signal)).resolves.toEqual(snapshot);
    await expect(client.turn(snapshot, 'Follow-up', signal)).resolves.toMatchObject({ chat: updated, saved: true });
    await expect(client.forget(snapshot, signal)).resolves.toBeUndefined();

    const root = 'http://[::1]:8765/v1/working-chats/';
    const identity = { name: snapshot.name, session_id: snapshot.session_id, revision: 2 };
    for (const [index, [path, payload]] of [
      ['read', { name: snapshot.name }],
      ['open', { name: snapshot.name }],
      ['turn', { ...identity, message: 'Follow-up' }],
      ['forget', identity],
    ].entries()) {
      expect(fetch).toHaveBeenNthCalledWith(index + 1, root + path, expect.objectContaining({
        method: 'POST',
        signal,
        redirect: 'error',
        body: JSON.stringify(payload),
        headers: { Authorization: 'Bearer fixture', 'Content-Type': 'application/json' },
      }));
    }
  });

  it('recognizes a services-only browser API by its direct local info endpoint', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      Response.json({ inference_available: false }),
    );
    await expect(createWorkingChatsClient()?.inferenceAvailable(signal)).resolves.toBe(false);
    expect(fetch).toHaveBeenCalledWith('http://127.0.0.1:8000/v1/info', expect.objectContaining({
      method: 'GET',
      redirect: 'error',
    }));
  });

  it.each([
    [409, 'Reload it'],
    [503, 'local model'],
  ])('returns actionable errors for HTTP %s without exposing response text', async (status, message) => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('Private server details', { status }));
    await expect(createWorkingChatsClient()!.turn(snapshot, 'Follow-up', signal))
      .rejects.toMatchObject<Partial<WorkingChatError>>({
        status,
        message: expect.stringContaining(message),
      });
  });

  it('keeps a reply that could not be saved distinct from the durable snapshot', async () => {
    const body = { chat: null, answer: 'Local reply', saved: false, reason: 'changed' };
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json(body));
    await expect(createWorkingChatsClient()!.turn(snapshot, 'Follow-up', signal)).resolves.toEqual(body);
  });

  it('rejects a successful write that returned the wrong session revision', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      Response.json({ chat: snapshot, answer: 'Local reply', saved: true }),
    );
    await expect(createWorkingChatsClient()!.turn(snapshot, 'Follow-up', signal))
      .rejects.toThrow('invalid shared chat response');
  });
});
