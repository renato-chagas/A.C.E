import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';

describe('desktop local model settings', () => {
  it('lets the user choose a tag and explicitly install that Ollama model', async () => {
    vi.stubGlobal('localStorage', { getItem: () => null });
    vi.stubGlobal('window', { __TAURI_INTERNALS__: {} });
    const { SettingsPage } = await import('./SettingsPage');
    const html = renderToStaticMarkup(<SettingsPage />);

    expect(html).toContain('aria-label="Ollama model tag to install"');
    expect(html).toContain('aria-label="Ollama model tag"');
    expect(html).toContain('Startup model');
    expect(html).toContain('<option value="cloudflare">Cloudflare Workers AI (online)</option>');
    expect(html).toContain('Install model');
    expect(html).toContain('https://ollama.com/library');
    expect(html).toContain('You choose the Ollama model tag.');
    expect(html).not.toContain('ollama pull &lt;model-name&gt;');
    vi.unstubAllGlobals();
  });
});
