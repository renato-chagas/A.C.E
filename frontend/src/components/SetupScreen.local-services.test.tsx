import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter, Route, Routes } from 'react-router';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { stageInferenceSource, type SetupStatus } from '../lib/api';
import { checkPackagedBackendHealth, desktopReadyMode, initDesktopDistribution, isOfflineDistribution } from '../lib/desktop-startup';
import { CustomEndpointSetup, InferenceSourceChooser, OllamaConsent } from './InferenceSourceSetup';

const native = vi.hoisted(() => ({ invoke: vi.fn() }));
vi.mock('@tauri-apps/api/core', () => ({ invoke: native.invoke }));
vi.mock('./Chat/MessageBubble', () => ({ MessageBubble: () => null }));
vi.mock('./Chat/SystemPanel', () => ({ SystemPanel: () => null }));
vi.mock('./Chat/InputArea', () => ({
  InputArea: ({ inferenceUnavailable }: { inferenceUnavailable?: boolean }) =>
    <textarea aria-label="Composer" data-paused={inferenceUnavailable} />,
}));

const localReady: SetupStatus = {
  phase: 'local_services_ready',
  detail: 'Import an Ollama model to enable chat.',
  server_ready: true,
  ollama_ready: false,
  model_ready: false,
  requires_source: false,
  error: null,
  source: 'unconfigured',
};

afterEach(async () => {
  native.invoke.mockResolvedValue(false);
  await initDesktopDistribution();
  vi.unstubAllGlobals();
  native.invoke.mockReset();
});

describe('desktop without a model', () => {
  it('opens the shell for local services and waits for model readiness before enabling inference', () => {
    expect(desktopReadyMode(localReady)).toBe('local-services');
    expect(desktopReadyMode({ ...localReady, server_ready: false })).toBeNull();
    expect(desktopReadyMode({ ...localReady, error: 'Unavailable' })).toBeNull();
    expect(desktopReadyMode({ ...localReady, phase: 'starting' })).toBeNull();
    expect(desktopReadyMode({ ...localReady, phase: 'ready' })).toBeNull();
    expect(desktopReadyMode({ ...localReady, phase: 'ready', model_ready: true })).toBe('inference');
  });

  it('offers configuration and renders a local memory page when inference is disabled', async () => {
    vi.stubGlobal('localStorage', { getItem: () => null });
    const { Layout } = await import('./Layout');
    const { MemoryFactsPage } = await import('../pages/MemoryFactsPage');
    const html = renderToStaticMarkup(
      <MemoryRouter initialEntries={['/memory-facts']}>
        <Routes>
          <Route element={<Layout onConfigureModel={vi.fn()} localServicesDetail={localReady.detail} />}>
            <Route path="memory-facts" element={<MemoryFactsPage />} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(html).toContain('Memory facts');
    expect(html).toContain('Configure model');
    expect(html).toContain(localReady.detail);
  });

  it('preserves local conversation browsing with the model-only composer removed', async () => {
    vi.stubGlobal('localStorage', { getItem: () => null });
    const { ChatPage } = await import('../pages/ChatPage');
    const html = renderToStaticMarkup(<MemoryRouter><ChatPage localServicesOnly /></MemoryRouter>);
    expect(html).toContain('You can read saved conversations and use local data.');
    expect(html).not.toContain('aria-label="Composer"');
    const withModel = renderToStaticMarkup(<MemoryRouter><ChatPage /></MemoryRouter>);
    expect(withModel).toContain('aria-label="Composer"');
  });

  it('leaves memory open and the composer mounted during an inference outage', async () => {
    vi.stubGlobal('localStorage', { getItem: () => null });
    const { Layout } = await import('./Layout');
    const { ChatPage } = await import('../pages/ChatPage');
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <Routes>
          <Route element={<Layout backendReachable inferenceUnavailable />}>
            <Route index element={<ChatPage inferenceUnavailable />} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(markup).toContain('The selected model is temporarily unavailable.');
    expect(markup).toContain('Memory Facts');
    expect(markup).toContain('aria-label="Composer" data-paused="true"');
    expect(markup).not.toContain('Cannot reach OpenJarvis backend');
  });

  it('offers a retry for saved settings when the packaged backend starts in local-services mode', async () => {
    vi.stubGlobal('localStorage', { getItem: () => null });
    const { Layout } = await import('./Layout');
    const markup = renderToStaticMarkup(
      <MemoryRouter>
        <Layout
          backendReachable
          localServicesDetail={localReady.detail}
          onConfigureModel={vi.fn()}
          onRetryModel={vi.fn()}
        />
      </MemoryRouter>,
    );
    expect(markup).toContain('Retry saved model');
    expect(markup).toContain('Configure model');
  });

  it('reports inference recovery from the native loopback probe', async () => {
    native.invoke
      .mockResolvedValueOnce('inference')
      .mockResolvedValueOnce('inference_unavailable')
      .mockResolvedValueOnce('inference')
      .mockRejectedValueOnce(new Error('IPC unavailable'));
    await expect(checkPackagedBackendHealth()).resolves.toBe('inference');
    await expect(checkPackagedBackendHealth()).resolves.toBe('inference_unavailable');
    await expect(checkPackagedBackendHealth()).resolves.toBe('inference');
    await expect(checkPackagedBackendHealth()).resolves.toBe('unreachable');
    expect(native.invoke).toHaveBeenCalledWith('check_packaged_backend_health');
  });

  it('requires an installed model in the offline distribution and leaves a path back to local services', async () => {
    vi.stubGlobal('window', { __TAURI_INTERNALS__: {} });
    native.invoke.mockResolvedValue(true);
    await initDesktopDistribution();

    expect(native.invoke).toHaveBeenCalledWith('is_offline_distribution');
    expect(isOfflineDistribution()).toBe(true);
    const chooser = renderToStaticMarkup(
      <InferenceSourceChooser onChoose={vi.fn()} onCancel={vi.fn()} />,
    );
    expect(chooser).toContain('Continue with local services');
    expect(chooser).toContain('Cloudflare · Gemma 4');
    expect(chooser).toContain('already installed or imported');
    const modelForm = renderToStaticMarkup(<OllamaConsent onBack={vi.fn()} onConfirm={vi.fn()} />);
    expect(modelForm).toContain('exact tag of a model already installed or imported');
    expect(modelForm).not.toContain('ollama.com/library');
    const customForm = renderToStaticMarkup(<CustomEndpointSetup onCancel={vi.fn()} onConfirm={vi.fn()} />);
    expect(customForm).toContain('Enter a loopback URL');
    expect(customForm).not.toContain('aria-label="API key"');
  });

  it('stages only the Cloudflare source and exact model through native setup', async () => {
    vi.stubGlobal('window', { __TAURI_INTERNALS__: {} });
    await stageInferenceSource({ kind: 'cloudflare' });
    expect(native.invoke).toHaveBeenCalledWith('set_inference_source', {
      kind: 'cloudflare',
      model: '@cf/google/gemma-4-26b-a4b-it',
      pending: true,
    });
  });

  it('preserves standard desktop behavior when a legacy launcher lacks the command', async () => {
    vi.stubGlobal('window', { __TAURI_INTERNALS__: {} });
    native.invoke.mockRejectedValueOnce(new Error('Unknown command'));
    await initDesktopDistribution();
    expect(isOfflineDistribution()).toBe(false);
  });
});
