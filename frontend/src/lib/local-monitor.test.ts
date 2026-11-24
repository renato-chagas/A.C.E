import { afterEach, describe, expect, it, vi } from 'vitest';
import { createLocalMonitorClient, type LocalMonitorStatus } from './local-monitor';

const connection = vi.hoisted(() => ({ base: 'http://127.0.0.1:8000', desktop: true }));
vi.mock('./api', () => ({
  getBase: () => connection.base,
  isTauri: () => connection.desktop,
  authHeaders: (extra = {}) => ({ Authorization: 'Bearer local-fixture', ...extra }),
}));

const stopped: LocalMonitorStatus = {
  enabled: false,
  running: false,
  externally_managed: false,
  notifications_enabled: false,
  history_available: true,
  incidents: [],
};
const running = { ...stopped, enabled: true, running: true };
const { signal } = new AbortController();

afterEach(() => {
  connection.base = 'http://127.0.0.1:8000';
  connection.desktop = true;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('local monitoring client', () => {
  it('reads and sends the explicit start and stop decision only to its local origin', async () => {
    connection.base = 'http://[::1]:8001';
    const send = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json(stopped))
      .mockResolvedValueOnce(Response.json(running))
      .mockResolvedValueOnce(Response.json(stopped));
    const client = createLocalMonitorClient()!;
    connection.base = 'https://remote.example';

    await expect(client.status(signal)).resolves.toEqual(stopped);
    await expect(client.setEnabled(true, signal)).resolves.toEqual(running);
    await expect(client.setEnabled(false, signal)).resolves.toEqual(stopped);
    const requestOptions = {
      signal,
      redirect: 'error',
      cache: 'no-store',
      credentials: 'omit',
      referrerPolicy: 'no-referrer',
    };
    const endpoint = 'http://[::1]:8001/v1/local-monitor';
    expect(send).toHaveBeenNthCalledWith(1, endpoint, expect.objectContaining({
      ...requestOptions, method: 'GET', headers: { Authorization: 'Bearer local-fixture' }, body: undefined,
    }));
    for (const [index, enabled] of [[2, true], [3, false]] as const) {
      expect(send).toHaveBeenNthCalledWith(index, endpoint, expect.objectContaining({
        ...requestOptions,
        method: 'PATCH',
        headers: { Authorization: 'Bearer local-fixture', 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      }));
    }
  });

  it.each(['https://remote.example', 'http://127.0.0.1.remote.example', 'http://localhost/api'])(
    'rejects the unsafe base %s',
    (base) => {
      connection.base = base;
      const send = vi.spyOn(globalThis, 'fetch');
      expect(createLocalMonitorClient()).toBeNull();
      expect(send).not.toHaveBeenCalled();
    },
  );

  it('requires a local browser origin in browser mode', () => {
    connection.desktop = false;
    vi.stubGlobal('window', { location: { origin: 'https://remote.example' } });
    expect(createLocalMonitorClient()).toBeNull();
  });

  it('does not display arbitrary detail from a conflict response', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json({ detail: 'private path' }, { status: 409 }));
    await expect(createLocalMonitorClient()!.setEnabled(true, signal)).rejects.toThrow(
      'The monitor is busy or controlled by another session.',
    );
  });

  it('sends the notification opt-in only when starting monitoring', async () => {
    const active = { ...running, notifications_enabled: true };
    const send = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json(active))
      .mockResolvedValueOnce(Response.json(stopped));
    const client = createLocalMonitorClient()!;

    await expect(client.setEnabled(true, signal, true)).resolves.toEqual(active);
    await expect(client.setEnabled(false, signal, true)).resolves.toEqual(stopped);
    expect(send).toHaveBeenNthCalledWith(1, 'http://127.0.0.1:8000/v1/local-monitor', expect.objectContaining({
      method: 'PATCH', body: JSON.stringify({ enabled: true, notifications_enabled: true }),
    }));
    expect(send).toHaveBeenNthCalledWith(2, 'http://127.0.0.1:8000/v1/local-monitor', expect.objectContaining({
      method: 'PATCH', body: JSON.stringify({ enabled: false }),
    }));
  });

  it('acknowledges one alert through a direct authenticated JSON request', async () => {
    const send = vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json(stopped));
    const client = createLocalMonitorClient()!;
    await expect(client.acknowledge(-1, signal)).rejects.toThrow('Choose a valid resource alert.');
    await expect(client.acknowledge(42, signal)).resolves.toEqual(stopped);
    expect(send).toHaveBeenCalledExactlyOnceWith(
      'http://127.0.0.1:8000/v1/local-monitor/incidents/42/acknowledge',
      {
        method: 'POST',
        headers: { Authorization: 'Bearer local-fixture', 'Content-Type': 'application/json' },
        body: '{}',
        cache: 'no-store',
        credentials: 'omit',
        redirect: 'error',
        referrerPolicy: 'no-referrer',
        signal,
      },
    );
  });

  it('handles a removed alert without exposing server detail', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json({ detail: 'private path' }, { status: 404 }));
    await expect(createLocalMonitorClient()!.acknowledge(42, signal)).rejects.toThrow(
      'This alert is no longer available. Refresh monitoring.',
    );
  });

  it('analyzes only the requested incident and asks for a model only when selected', async () => {
    const result = {
      incident_id: 42,
      assessment: { id: 2, analysis: 'Saved measurement', created_at: '2026-11-23T10:00:00Z' },
      mode: 'measurements',
    };
    const send = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json(result))
      .mockResolvedValueOnce(Response.json(result))
      .mockResolvedValueOnce(Response.json({ ...result, incident_id: 43 }));
    const client = createLocalMonitorClient()!;
    const endpoint = 'http://127.0.0.1:8000/v1/local-monitor/incidents/42/analyze';

    await expect(client.analyze(42, false, signal)).resolves.toEqual(result);
    await expect(client.analyze(42, true, signal)).resolves.toEqual(result);
    await expect(client.analyze(42, true, signal)).rejects.toThrow('invalid incident assessment');
    expect(send).toHaveBeenNthCalledWith(1, endpoint, {
      method: 'POST',
      headers: { Authorization: 'Bearer local-fixture', 'Content-Type': 'application/json' },
      body: '{}',
      cache: 'no-store',
      credentials: 'omit',
      redirect: 'error',
      referrerPolicy: 'no-referrer',
      signal,
    });
    expect(send).toHaveBeenNthCalledWith(2, endpoint, expect.objectContaining({
      method: 'POST', body: JSON.stringify({ use_local_model: true }),
    }));
    expect(send).toHaveBeenCalledTimes(3);
  });

  it.each([
    [409, 'This alert changed. Refresh monitoring and try again.'],
    [429, 'Analysis is busy. Try again in a moment.'],
  ] as const)('gives an incident-level error for analysis response %s', async (code, expected) => {
    const send = vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json({ detail: 'private' }, { status: code }));
    await expect(createLocalMonitorClient()!.analyze(42, true, signal)).rejects.toThrow(expected);
    expect(send).toHaveBeenCalledTimes(1);
  });

  it.each([
    { ...stopped, externally_managed: 'false' },
    { ...stopped, notifications_enabled: 'true' },
    { ...stopped, incidents: [{ summary: 'partial alert' }] },
  ])('rejects an invalid response', async (value) => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json(value));
    await expect(createLocalMonitorClient()!.status(signal)).rejects.toThrow('invalid monitoring response');
  });
});
