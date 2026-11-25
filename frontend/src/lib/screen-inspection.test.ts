import { afterEach, describe, expect, it, vi } from 'vitest';
import { createScreenInspectionClient } from './screen-inspection';

const connection = vi.hoisted(() => ({ base: 'http://127.0.0.1:8000', desktop: true }));
vi.mock('./api', () => ({
  getBase: () => connection.base,
  isTauri: () => connection.desktop,
  authHeaders: (extra = {}) => ({ ...extra, Authorization: 'Bearer local-fixture' }),
}));

const signal = new AbortController().signal;

afterEach(() => {
  connection.base = 'http://127.0.0.1:8000';
  connection.desktop = true;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('screen inspection client', () => {
  it('checks capabilities and sends each explicit action only to its local origin', async () => {
    connection.base = 'http://[::1]:8001';
    const status = {
      available: true,
      text_available: true,
      text_question_available: true,
      cloud_vision_available: true,
    };
    const answer = '<script>literal screen text</script>';
    const send = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json(status))
      .mockResolvedValueOnce(Response.json({ answer }))
      .mockResolvedValueOnce(Response.json({ answer: 'Local analysis' }))
      .mockResolvedValueOnce(Response.json({ answer: 'Local OCR analysis' }))
      .mockResolvedValueOnce(Response.json({ answer: 'Cloud analysis' }));
    const client = createScreenInspectionClient()!;
    connection.base = 'https://remote.example';

    await expect(client.status(signal)).resolves.toEqual(status);
    await expect(client.inspect({ mode: 'text' }, signal)).resolves.toBe(answer);
    const vision = { mode: 'vision', question: 'What do you see?' } as const;
    await expect(client.inspect(vision, signal)).resolves.toBe('Local analysis');
    const ocrQuestion = { mode: 'text-question', question: 'What text do you see?' } as const;
    await expect(client.inspect(ocrQuestion, signal)).resolves.toBe('Local OCR analysis');
    const cloudVision = { mode: 'cloud-vision', question: 'What do you see?', confirm_upload: true } as const;
    await expect(client.inspect(cloudVision, signal)).resolves.toBe('Cloud analysis');

    const options = {
      signal,
      redirect: 'error',
      cache: 'no-store',
      credentials: 'omit',
      referrerPolicy: 'no-referrer',
    };
    const endpoint = 'http://[::1]:8001/v1/screen/inspect';
    expect(send).toHaveBeenNthCalledWith(1, endpoint, expect.objectContaining({
      ...options, method: 'GET', body: undefined,
      headers: { Authorization: 'Bearer local-fixture' },
    }));
    const actions = [[2, { mode: 'text' }], [3, vision], [4, ocrQuestion], [5, cloudVision]] as const;
    for (const [index, body] of actions) {
      expect(send).toHaveBeenNthCalledWith(index, endpoint, expect.objectContaining({
        ...options, method: 'POST', body: JSON.stringify(body),
        headers: { Authorization: 'Bearer local-fixture', 'Content-Type': 'application/json' },
      }));
    }
  });

  it.each(['https://remote.example', 'http://localhost.remote.example', 'http://127.0.0.1/api'])(
    'rejects the unsafe base %s', (base) => {
      connection.base = base;
      const send = vi.spyOn(globalThis, 'fetch');
      expect(createScreenInspectionClient()).toBeNull();
      expect(send).not.toHaveBeenCalled();
    },
  );

  it('requires a local browser origin in browser mode', () => {
    connection.desktop = false;
    vi.stubGlobal('window', { location: { origin: 'https://remote.example' } });
    expect(createScreenInspectionClient()).toBeNull();
  });

  it('returns scoped errors instead of displaying backend details', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json({ detail: 'private image data' }, { status: 503 }))
      .mockResolvedValueOnce(Response.json({ detail: 'private model path' }, { status: 502 }));
    const client = createScreenInspectionClient()!;
    await expect(client.inspect({ mode: 'text' }, signal)).rejects.toThrow('Local text recognition or screen capture is unavailable.');
    await expect(client.inspect({ mode: 'vision', question: 'What is here?' }, signal)).rejects.toThrow('The local vision model could not analyze this capture.');
  });

  it('keeps newer actions disabled with an older backend', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json({ available: true, text_available: true }));
    const client = createScreenInspectionClient()!;
    await expect(client.status(signal)).resolves.toEqual({
      available: true, text_available: true, text_question_available: false, cloud_vision_available: false,
    });
  });

  it('reports Cloudflare credential and usage errors without using upstream response details', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json({ detail: 'private token' }, { status: 424 }))
      .mockResolvedValueOnce(Response.json({ detail: 'private account id' }, { status: 429 }));
    const client = createScreenInspectionClient()!;
    const cloudVision = { mode: 'cloud-vision', question: 'Look once', confirm_upload: true } as const;
    await expect(client.inspect(cloudVision, signal)).rejects.toThrow('Cloudflare denied access');
    await expect(client.inspect(cloudVision, signal)).rejects.toThrow('Cloudflare reached a usage or rate limit');
  });

  it('validates availability and requires a nonempty result', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(Response.json({ available: false, text_available: 'true' }))
      .mockResolvedValueOnce(Response.json({ answer: '' }));
    const client = createScreenInspectionClient()!;
    await expect(client.status(signal)).rejects.toThrow('invalid screen analysis availability');
    await expect(client.inspect({ mode: 'text' }, signal)).rejects.toThrow('invalid screen inspection result');
  });
});