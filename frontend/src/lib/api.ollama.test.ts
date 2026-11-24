import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const invokeMock = vi.hoisted(() => vi.fn());
vi.mock('@tauri-apps/api/core', () => ({ invoke: invokeMock }));

const fetchMock = vi.fn<typeof fetch>();

beforeEach(() => {
  vi.resetModules();
  invokeMock.mockReset();
  fetchMock.mockReset();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('fetchOllamaModels', () => {
  it('uses the native bridge in the desktop app instead of browser CORS', async () => {
    vi.stubGlobal('window', { __TAURI_INTERNALS__: {} });
    const models = [{ name: 'qwen3.5:4b', size: 4_200_000_000 }];
    invokeMock.mockResolvedValue(models);
    const { fetchOllamaModels } = await import('./api');

    await expect(fetchOllamaModels()).resolves.toEqual(models);
    expect(invokeMock).toHaveBeenCalledWith('list_ollama_models');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('queries the local Ollama HTTP API in the browser build', async () => {
    vi.stubGlobal('window', {});
    const models = [{ name: 'qwen3.5:4b', size: 4_200_000_000 }];
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ models }), { status: 200 }));
    const { fetchOllamaModels } = await import('./api');

    await expect(fetchOllamaModels()).resolves.toEqual(models);
    expect(fetchMock).toHaveBeenCalledWith('http://127.0.0.1:11434/api/tags');
  });

  it('surfaces a local runtime failure in the browser build', async () => {
    vi.stubGlobal('window', {});
    fetchMock.mockResolvedValue(new Response('', { status: 503 }));
    const { fetchOllamaModels } = await import('./api');

    await expect(fetchOllamaModels()).rejects.toThrow('Ollama returned HTTP 503');
  });
});
