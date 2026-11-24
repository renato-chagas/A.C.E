import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ModelInfo } from '../types';

class MemoryStorage {
  private store = new Map<string, string>();

  getItem(key: string): string | null {
    return this.store.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.store.set(key, String(value));
  }
}

const model = (id: string): ModelInfo => ({
  id,
  object: 'model',
  created: 0,
  owned_by: 'openjarvis',
});

beforeEach(() => {
  vi.resetModules();
  (globalThis as unknown as { localStorage: MemoryStorage }).localStorage =
    new MemoryStorage();
});

afterEach(() => {
  (globalThis as unknown as { localStorage?: MemoryStorage }).localStorage =
    undefined;
});

describe('setModels', () => {
  it('does not select an embedding-only model', async () => {
    const { useAppStore } = await import('./store');

    useAppStore.getState().setModels([model('nomic-embed-text')]);

    expect(useAppStore.getState().selectedModel).toBe('');
  });

  it('clears a missing selection when no chat fallback exists', async () => {
    const { useAppStore } = await import('./store');
    useAppStore.getState().setSelectedModel('deleted-chat-model');

    useAppStore.getState().setModels([model('nomic-embed-text')]);

    expect(useAppStore.getState().selectedModel).toBe('');
  });

  it('replaces an embedding selection with an available chat model', async () => {
    const { useAppStore } = await import('./store');
    useAppStore.getState().setSelectedModel('all-minilm:latest');

    useAppStore.getState().setModels([
      model('all-minilm:latest'),
      model('qwen3.5:4b'),
    ]);

    expect(useAppStore.getState().selectedModel).toBe('qwen3.5:4b');
  });
});

describe('desktop startup model selection', () => {
  const chosen = 'qwen/qwen3.5-9b';
  const recommended = 'qwen/qwen2.5-coder-14b';
  const catalog = [model(recommended), model(chosen)];
  const info = (id: string) => ({ model: id, agent: null, engine: 'lmstudio' });

  it('uses the exact served model over the catalog order and recommendation', async () => {
    const { useAppStore } = await import('./store');
    const store = useAppStore.getState();
    store.setServerInfo(info(chosen));
    store.setModels(catalog, { requireExactModel: true, recommendedModel: recommended });
    expect(useAppStore.getState().selectedModel).toBe(chosen);
  });

  it('keeps the exact model when it is absent from a catalog refresh', async () => {
    const { useAppStore } = await import('./store');
    const store = useAppStore.getState();
    store.setServerInfo(info(chosen));
    store.setModels([], { requireExactModel: true });
    store.setModels([model(recommended)]);
    expect(useAppStore.getState().selectedModel).toBe(chosen);
  });

  it('preserves a deliberate session selection and adopts the new exact model on startup', async () => {
    const { useAppStore } = await import('./store');
    const store = useAppStore.getState();
    store.setServerInfo(info(chosen));
    store.setModels(catalog, { requireExactModel: true });
    store.setSelectedModel(recommended);
    store.setModels(catalog);
    expect(useAppStore.getState().selectedModel).toBe(recommended);

    const replacement = 'local/new-model';
    store.setServerInfo(info(replacement));
    store.setModels([...catalog, model(replacement)], { requireExactModel: true });
    expect(useAppStore.getState().selectedModel).toBe(replacement);
  });

  it('keeps the composer without a selection when the packaged backend provides no model identity', async () => {
    const { useAppStore } = await import('./store');
    const store = useAppStore.getState();
    store.setServerInfo(null);
    store.setModels(catalog, { requireExactModel: true, recommendedModel: recommended });
    expect(useAppStore.getState().selectedModel).toBe('');
  });
});
