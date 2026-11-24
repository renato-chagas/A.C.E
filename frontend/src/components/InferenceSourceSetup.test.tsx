import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CloudflareSetupForm } from './CloudflareSetupForm';
import {
  CloudflareConsent,
  CustomEndpointSetup,
  InferenceRecoveryButton,
  InferenceSourceChooser,
  OllamaConsent,
  inferenceSetupReducer,
  persistAndStartInferenceSource,
  type InferenceSourceSubmission,
} from './InferenceSourceSetup';

afterEach(() => vi.unstubAllGlobals());

describe('first-run inference source navigation', () => {
  it('requires a second, explicit confirmation before local setup', () => {
    expect(inferenceSetupReducer('choose', 'choose_ollama')).toBe('ollama');
    expect(inferenceSetupReducer('ollama', 'back')).toBe('choose');
  });

  it('lets custom endpoint setup cancel safely back to the chooser', () => {
    expect(inferenceSetupReducer('choose', 'choose_custom')).toBe('custom');
    expect(inferenceSetupReducer('custom', 'back')).toBe('choose');
  });

  it('renders the source choices without claiming setup has started', () => {
    const html = renderToStaticMarkup(<InferenceSourceChooser onChoose={vi.fn()} />);

    expect(html).toContain('Local Ollama');
    expect(html).toContain('OpenAI-compatible server');
    expect(html).toContain('Cloudflare Workers AI');
    expect(html).toContain('Nothing is installed, started, or downloaded until you confirm');
    expect(html).not.toContain('Starting local setup');
  });

  it('discloses online chat and shows masked credential fields on desktop', () => {
    vi.stubGlobal('window', { __TAURI_INTERNALS__: {} });
    expect(inferenceSetupReducer('choose', 'choose_cloudflare')).toBe('cloudflare');
    expect(inferenceSetupReducer('cloudflare', 'back')).toBe('choose');
    const html = renderToStaticMarkup(<CloudflareConsent onBack={vi.fn()} onConfirm={vi.fn()} />);

    expect(html).toContain('sends messages and conversation history to Cloudflare Workers AI');
    expect(html).toContain('Cloudflare Account ID');
    expect(html).toContain('Workers AI API Token');
    expect(html).toContain('type="password"');
    expect(html).toContain('Connect to Cloudflare');
    expect(html).toContain('Use REST API');
    expect(html).toContain('Cloudflare model');
    expect(html).toContain('value="@cf/google/gemma-4-26b-a4b-it" selected');
    expect(html).toContain('@cf/qwen/qwen3.8-27b');
    expect(html).toContain('@cf/openai/gpt-oss-20b');
    expect(html).toContain('Enter model ID');
    expect(html).not.toContain('PowerShell');
  });

  it('restores a custom model choice in settings without prefilling any credentials', () => {
    vi.stubGlobal('window', { __TAURI_INTERNALS__: {} });
    const html = renderToStaticMarkup(
      <CloudflareSetupForm inSettings initialModel="@hf/provider/custom-model" onConfirm={vi.fn()} />,
    );
    expect(html).toContain('value="custom" selected');
    expect(html).toContain('Cloudflare model ID');
    expect(html).toContain('value="@hf/provider/custom-model"');
    expect(html).toContain('Save Cloudflare for next launch');
    expect(html).not.toMatch(/<input[^>]*type="password"[^>]*value=/);
  });

  it('directs browser users to desktop before entering Cloudflare credentials', () => {
    const html = renderToStaticMarkup(<CloudflareConsent onBack={vi.fn()} onConfirm={vi.fn()} />);
    expect(html).toContain('Open the A.C.E. desktop app to configure Cloudflare.');
    expect(html).not.toContain('<input');
    expect(html).not.toContain('Connect to Cloudflare');
  });

  it('requires the exact model tag and discloses what the download consent covers', () => {
    const html = renderToStaticMarkup(
      <OllamaConsent onBack={vi.fn()} onConfirm={vi.fn()} />,
    );

    expect(html).toContain('Choose your local model');
    expect(html).toContain('aria-label="Ollama model tag"');
    expect(html).toMatch(/aria-label="Ollama model tag"[^>]*required/);
    expect(html).toContain('download only this model');
    expect(html).toContain('https://ollama.com/library');
    expect(html).toContain('ollama list');
    expect(html).toContain('several gigabytes');
    expect(html).toContain('Going back leaves Ollama stopped');
    expect(html).toContain('Use this model and continue');
    expect(html).toContain('Back');
  });

  it('requires endpoint coordinates and keeps an explicit cancel control', () => {
    const html = renderToStaticMarkup(
      <CustomEndpointSetup onCancel={vi.fn()} onConfirm={vi.fn()} />,
    );

    expect(html).toContain('Server URL');
    expect(html).toContain('Model ID');
    expect(html).toMatch(/aria-label="Server URL"[^>]*required/);
    expect(html).toMatch(/aria-label="Model ID"[^>]*required/);
    expect(html).toContain('Start server');
    expect(html).toContain('http://localhost:1234/v1');
    expect(html).toContain('Cancel');
  });

  it('renders a source-change recovery action with a guarded busy state', () => {
    const ready = renderToStaticMarkup(
      <InferenceRecoveryButton recovering={false} onChange={vi.fn()} />,
    );
    const stopping = renderToStaticMarkup(
      <InferenceRecoveryButton recovering onChange={vi.fn()} />,
    );

    expect(ready).toContain('Change inference source');
    expect(ready).not.toMatch(/ disabled(?:=""|>)/);
    expect(stopping).toContain('Stopping setup...');
    expect(stopping).toMatch(/ disabled(?:=""|>)/);
  });
});

describe('first-run inference source persistence', () => {
  it('persists the exact user-selected Ollama tag before requesting backend startup', async () => {
    const calls: string[] = [];
    const persist = vi.fn(async (source: InferenceSourceSubmission) => {
      calls.push(`persist:${source.kind}:${source.model}`);
    });
    const start = vi.fn(async () => {
      calls.push('start');
    });

    await persistAndStartInferenceSource(
      { kind: 'ollama', model: 'qwen3.5:4b' },
      persist,
      start,
    );

    expect(calls).toEqual(['persist:ollama:qwen3.5:4b', 'start']);
    expect(persist).toHaveBeenCalledWith({
      kind: 'ollama',
      model: 'qwen3.5:4b',
    });
  });

  it('does not start anything when persistence fails', async () => {
    const persist = vi.fn(async () => {
      throw new Error('secure storage unavailable');
    });
    const start = vi.fn(async () => {});

    await expect(
      persistAndStartInferenceSource({ kind: 'ollama', model: 'qwen3.5:4b' }, persist, start),
    ).rejects.toThrow('secure storage unavailable');
    expect(start).not.toHaveBeenCalled();
  });

  it('rolls a staged choice back when startup rejects', async () => {
    const calls: string[] = [];
    const persist = vi.fn(async () => {
      calls.push('persist');
    });
    const start = vi.fn(async () => {
      calls.push('start');
      throw new Error('endpoint unavailable');
    });
    const rollback = vi.fn(async () => {
      calls.push('rollback');
    });

    await expect(
      persistAndStartInferenceSource(
        { kind: 'custom', host: 'http://localhost:1234', model: 'missing' },
        persist,
        start,
        rollback,
      ),
    ).rejects.toThrow('endpoint unavailable');
    expect(calls).toEqual(['persist', 'start', 'rollback']);
  });
});
