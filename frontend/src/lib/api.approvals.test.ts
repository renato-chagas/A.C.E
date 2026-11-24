import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const fetchMock = vi.fn<typeof fetch>();

class MemoryStorage {
  private store = new Map<string, string>();

  getItem(key: string): string | null {
    return this.store.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.store.set(key, String(value));
  }

  removeItem(key: string): void {
    this.store.delete(key);
  }

  clear(): void {
    this.store.clear();
  }
}

beforeEach(() => {
  vi.resetModules();
  fetchMock.mockReset();
  globalThis.fetch = fetchMock;
  (globalThis as unknown as { localStorage: MemoryStorage }).localStorage = new MemoryStorage();
});

afterEach(() => {
  (globalThis as unknown as { localStorage?: MemoryStorage }).localStorage = undefined;
});

async function freshApi() {
  return await import('./api');
}

describe('approval decision requests', () => {
  it('sends the remember flag for an approved pattern', async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          id: 'action-1',
          status: 'executed',
          execution: 'succeeded',
          remembered: true,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    const { approveAction } = await freshApi();

    await expect(approveAction('action-1', true)).resolves.toMatchObject({
      remembered: true,
      execution: 'succeeded',
    });
    expect(fetchMock).toHaveBeenCalledWith(
      '/v1/approvals/action-1/approve?remember_future=true',
      { method: 'POST', headers: {} },
    );
  });

  it('sends remembered denial through the same exact-action endpoint', async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ id: 'action-2', status: 'denied', remembered: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    const { denyAction } = await freshApi();

    await expect(denyAction('action-2', true)).resolves.toMatchObject({
      status: 'denied',
      remembered: true,
    });
    expect(fetchMock).toHaveBeenCalledWith(
      '/v1/approvals/action-2/deny?remember_future=true',
      { method: 'POST', headers: {} },
    );
  });

  it('leaves one-time approval requests unchanged', async () => {
    fetchMock.mockResolvedValue(
      new Response(
        JSON.stringify({ id: 'action-3', status: 'executed', execution: 'succeeded' }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    const { approveAction } = await freshApi();

    await approveAction('action-3');

    expect(fetchMock).toHaveBeenCalledWith('/v1/approvals/action-3/approve', {
      method: 'POST',
      headers: {},
    });
  });
});
