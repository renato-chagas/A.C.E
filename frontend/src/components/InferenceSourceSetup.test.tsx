import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import {
  CloudflareConsent,
  CloudflareSetupGuide,
  CustomEndpointSetup,
  InferenceRecoveryButton,
  InferenceSourceChooser,
  OllamaConsent,
  cloudflarePowerShellCommand,
  inferenceSetupReducer,
  persistAndStartInferenceSource,
  type InferenceSourceSubmission,
} from './InferenceSourceSetup';

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
    expect(html).toContain('Cloudflare · Gemma 4');
    expect(html).toContain('Nothing is installed, started, or downloaded until you confirm');
    expect(html).not.toContain('Starting local setup');
  });

  it('discloses online chat before a separate Cloudflare confirmation', () => {
    expect(inferenceSetupReducer('choose', 'choose_cloudflare')).toBe('cloudflare');
    expect(inferenceSetupReducer('cloudflare', 'back')).toBe('choose');
    const html = renderToStaticMarkup(<CloudflareConsent onBack={vi.fn()} onConfirm={vi.fn()} />);
    expect(html).toContain('@cf/google/gemma-4-26b-a4b-it');
    expect(html).toContain('sends messages and conversation history to Cloudflare Workers AI');
    expect(html).toContain('CLOUDFLARE_ACCOUNT_ID');
    expect(html).toContain('CLOUDFLARE_API_TOKEN');
    expect(html).toContain('Use REST API');
    expect(html).toContain('Sair do A.C.E.');
    expect(html).toContain('Copy PowerShell setup');
    expect(html).toContain('-AsSecureString');
    expect(html).toContain('Connect to Cloudflare');
    expect(html).not.toContain('<input');
  });

  it('tells users to save Cloudflare before restarting from Settings', () => {
    const html = renderToStaticMarkup(<CloudflareSetupGuide inSettings />);

    expect(html.indexOf('Save Cloudflare for next launch')).toBeLessThan(html.indexOf('Sair do A.C.E.'));
  });

  it('prompts for both credentials before launching the exact executable path', () => {
    const commands = cloudflarePowerShellCommand("D:\\Pat's Apps\\A.C.E.\\openjarvis-desktop.exe");

    expect(commands).toContain("Read-Host 'Cloudflare Account ID'");
    expect(commands).toContain("Read-Host 'Workers AI API token' -AsSecureString");
    expect(commands).toContain("Start-Process -FilePath 'D:\\Pat''s Apps\\A.C.E.\\openjarvis-desktop.exe'");
    expect(commands.indexOf('$env:CLOUDFLARE_API_TOKEN')).toBeLessThan(commands.indexOf('Start-Process'));
    expect(cloudflarePowerShellCommand()).toContain("Read-Host 'Full path to openjarvis-desktop.exe'");
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
