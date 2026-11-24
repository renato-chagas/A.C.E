import { useEffect, useReducer, useState, type FormEvent, type ReactNode } from 'react';
import { ArrowLeft, Cloud, Cpu, Download, Loader2, Server, ShieldCheck } from 'lucide-react';
import {
  fetchOllamaModels,
  isTauri,
  resetInferenceSource,
  stageInferenceSource,
  startBackend,
  type InferenceSource,
} from '../lib/api';
import { isOfflineDistribution } from '../lib/desktop-startup';
import { CloudflareSetupForm } from './CloudflareSetupForm';

export type InferenceSetupStep = 'choose' | InferenceSource['kind'];
export type InferenceSetupAction = `choose_${InferenceSource['kind']}` | 'back';
export type InferenceSourceSubmission = InferenceSource & { apiKey?: string };

/** Pure navigation state: merely inspecting a source can never start setup. */
export function inferenceSetupReducer(
  state: InferenceSetupStep,
  action: InferenceSetupAction,
): InferenceSetupStep {
  switch (action) {
    case 'choose_ollama':
      return 'ollama';
    case 'choose_custom':
      return 'custom';
    case 'choose_cloudflare':
      return 'cloudflare';
    case 'back':
      return 'choose';
    default:
      return state;
  }
}

/**
 * Persist first, start second. Dependency injection keeps the consent boundary
 * directly testable without a Tauri runtime.
 */
export async function persistAndStartInferenceSource(
  source: InferenceSourceSubmission,
  persist: (value: InferenceSourceSubmission) => Promise<void> = stageInferenceSource,
  start: () => Promise<void> = startBackend,
  rollback: () => Promise<void> = resetInferenceSource,
): Promise<void> {
  let persisted = false;
  try {
    await persist(source);
    persisted = true;
    await start();
  } catch (error) {
    if (persisted) {
      await rollback().catch(() => {});
    }
    throw error;
  }
}

function SetupFrame({ children }: { children: React.ReactNode }) {
  return (
    <div className="fixed inset-0 grid overflow-y-auto" style={{ background: 'var(--color-bg)' }}>
      <div className="m-auto w-full max-w-xl px-4 py-6 sm:px-6">
        <div className="text-center mb-8">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
            style={{ background: 'var(--color-accent-subtle)', color: 'var(--color-accent)' }}
          >
            <Cpu size={32} />
          </div>
          <h1 className="text-2xl font-bold mb-1" style={{ color: 'var(--color-text)' }}>
            Choose your inference source
          </h1>
          <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
            {isOfflineDistribution()
              ? 'Choose an installed model or opt in to online chat.'
              : 'Nothing is installed, started, or downloaded until you confirm a choice.'}
          </p>
        </div>
        <div
          className="p-6 rounded-2xl"
          style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
        >
          {children}
        </div>
      </div>
    </div>
  );
}

export function InferenceSourceChooser({
  onChoose,
  onCancel,
}: {
  onChoose: (source: InferenceSource['kind']) => void;
  onCancel?: () => void;
}) {
  return (
    <SetupFrame>
      <div className="flex flex-col gap-3">
        <button
          type="button"
          onClick={() => onChoose('ollama')}
          className="flex items-start gap-4 p-4 rounded-xl text-left transition-all cursor-pointer"
          style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }}
        >
          <Cpu size={22} className="shrink-0 mt-0.5" style={{ color: 'var(--color-accent)' }} />
          <span>
            <span className="block text-sm font-semibold" style={{ color: 'var(--color-text)' }}>
              Local Ollama
            </span>
            <span className="block text-xs mt-1" style={{ color: 'var(--color-text-secondary)' }}>
              {isOfflineDistribution()
                ? 'Use a model already installed or imported in Ollama.'
                : 'Run models on this computer. Requires a separate confirmation before Ollama starts or a model downloads.'}
            </span>
          </span>
        </button>
        <button
          type="button"
          onClick={() => onChoose('custom')}
          className="flex items-start gap-4 p-4 rounded-xl text-left transition-all cursor-pointer"
          style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }}
        >
          <Server size={22} className="shrink-0 mt-0.5" style={{ color: 'var(--color-accent)' }} />
          <span>
            <span className="block text-sm font-semibold" style={{ color: 'var(--color-text)' }}>
              OpenAI-compatible server
            </span>
            <span className="block text-xs mt-1" style={{ color: 'var(--color-text-secondary)' }}>
              {isOfflineDistribution()
                ? 'Connect to a compatible inference server running on this computer.'
                : 'Connect to LM Studio, vLLM, SGLang, llama.cpp, MLX, or another compatible endpoint. Ollama stays off.'}
            </span>
          </span>
        </button>
        <button
          type="button"
          onClick={() => onChoose('cloudflare')}
          className="flex items-start gap-4 p-4 rounded-xl text-left transition-all cursor-pointer"
          style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }}
        >
          <Cloud size={22} className="shrink-0 mt-0.5" style={{ color: 'var(--color-accent)' }} />
          <span>
            <span className="block text-sm font-semibold" style={{ color: 'var(--color-text)' }}>
              Cloudflare Workers AI
            </span>
            <span className="block text-xs mt-1" style={{ color: 'var(--color-text-secondary)' }}>
              Online chat through your Cloudflare Workers AI account.
            </span>
          </span>
        </button>
        {onCancel && <BackButton onClick={onCancel} label="Continue with local services" />}
      </div>
    </SetupFrame>
  );
}

function BackButton({ onClick, label = 'Back', disabled = false }: { onClick: () => void; label?: string; disabled?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="flex items-center gap-1.5 text-sm mb-5 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
      style={{ color: 'var(--color-text-secondary)' }}
    >
      <ArrowLeft size={15} />
      {label}
    </button>
  );
}

export function InferenceRecoveryButton({
  recovering,
  onChange,
  label = 'Change inference source',
}: {
  recovering: boolean;
  onChange: () => void;
  label?: string;
}) {
  return (
    <button
      type="button"
      onClick={onChange}
      disabled={recovering}
      className="w-full mt-4 py-2.5 px-4 rounded-xl text-sm font-medium cursor-pointer disabled:cursor-not-allowed disabled:opacity-60"
      style={{
        color: 'var(--color-text-secondary)',
        border: '1px solid var(--color-border)',
        background: 'var(--color-surface)',
      }}
    >
      {recovering ? 'Stopping setup...' : label}
    </button>
  );
}

function SetupSteps({ children }: { children: ReactNode }) {
  return (
    <ol className="list-decimal space-y-2 pl-5 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
      {children}
    </ol>
  );
}

export function OllamaSetupGuide({ inSettings = false }: { inSettings?: boolean }) {
  const offlineDistribution = isOfflineDistribution();
  return (
    <SetupSteps>
      <li>Install and open <a href="https://ollama.com/download" target="_blank" rel="noopener noreferrer" className="underline">Ollama</a>.</li>
      <li>
        {offlineDistribution && !inSettings
          ? <>Pick a model below or run <code>ollama list</code> to check its tag.</>
          : <>Run <code>ollama list</code> in a terminal.</>}{' '}
        {offlineDistribution ? (
          <>Use the exact tag of a model already installed or imported. To add a model, run <code>ollama pull</code> followed by the tag you chose.</>
        ) : (
          <>Use a tag from the list or choose one from the <a href="https://ollama.com/library" target="_blank" rel="noopener noreferrer" className="underline">Ollama model library</a>.</>
        )}
      </li>
      <li>
        {(!offlineDistribution || inSettings) && 'Enter the tag below. '}
        {inSettings
          ? <>Click <strong>Save inference source</strong> and reopen A.C.E.</>
          : <>Click <strong>Use this model and continue</strong>.</>}
        {!offlineDistribution && ' A.C.E. will download only this model if it is missing. A download can use several gigabytes.'}
      </li>
    </SetupSteps>
  );
}

type InstalledOllamaModels =
  | { status: 'loading' | 'unavailable' }
  | { status: 'ready'; tags: string[] };

export function OllamaConsent({
  onBack,
  onConfirm,
  busy = false,
  error = '',
}: {
  onBack: () => void;
  onConfirm: (modelTag: string) => void;
  busy?: boolean;
  error?: string;
}) {
  const [modelTag, setModelTag] = useState('');
  const [validationError, setValidationError] = useState('');
  const [installedModels, setInstalledModels] = useState<InstalledOllamaModels>({ status: 'loading' });
  const [lookupAttempt, retryLookup] = useReducer((attempt: number) => attempt + 1, 0);
  const showInstalledModels = isOfflineDistribution() && isTauri();
  const installedTags = installedModels.status === 'ready' ? installedModels.tags : [];

  useEffect(() => {
    if (!showInstalledModels) return;
    let active = true;
    void fetchOllamaModels().then(
      (models) => {
        if (active) setInstalledModels({ status: 'ready', tags: [...new Set(models.map(({ name }) => name))] });
      },
      () => {
        if (active) setInstalledModels({ status: 'unavailable' });
      },
    );
    return () => { active = false; };
  }, [showInstalledModels, lookupAttempt]);

  const chooseTag = (tag: string) => {
    setModelTag(tag);
    setValidationError('');
  };

  const refreshModels = () => {
    setInstalledModels({ status: 'loading' });
    retryLookup();
  };

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const selected = modelTag.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$/.test(selected)) {
      setValidationError("Enter a valid Ollama model tag using letters, numbers, '.', '_', ':', '/', or '-'.");
      return;
    }
    setValidationError('');
    onConfirm(selected);
  };

  return (
    <SetupFrame>
      <BackButton onClick={onBack} disabled={busy} />
      <div className="flex items-start gap-3 mb-5">
        <Download size={22} className="shrink-0 mt-0.5" style={{ color: 'var(--color-accent)' }} />
        <div>
          <h2 className="text-base font-semibold mb-1" style={{ color: 'var(--color-text)' }}>
            Choose your local model
          </h2>
          <OllamaSetupGuide />
        </div>
      </div>
      {showInstalledModels && (
        <div className="mb-4">
          <div className="flex items-center justify-between gap-3 mb-1.5 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
            <label htmlFor="installed-ollama-model" className="font-medium">Installed Ollama models</label>
            <button
              type="button"
              onClick={refreshModels}
              disabled={busy || installedModels.status === 'loading'}
              className="underline cursor-pointer disabled:cursor-not-allowed disabled:opacity-60"
            >
              Refresh models
            </button>
          </div>
          <select
            id="installed-ollama-model"
            value={installedTags.includes(modelTag) ? modelTag : ''}
            onChange={(event) => chooseTag(event.target.value)}
            disabled={busy || !installedTags.length}
            className="w-full text-sm px-3 py-2 rounded-lg outline-none"
            style={{ background: 'var(--color-bg-secondary)', color: 'var(--color-text)', border: '1px solid var(--color-border)' }}
          >
            <option value="" disabled>Choose an installed model</option>
            {installedTags.map((tag) => <option key={tag} value={tag}>{tag}</option>)}
          </select>
          {installedModels.status === 'loading' && (
            <p role="status" className="mt-1.5 text-xs">Checking installed Ollama models…</p>
          )}
          {installedModels.status === 'unavailable' && (
            <p role="status" className="mt-1.5 text-xs">Open Ollama, then refresh the list or enter its model tag below.</p>
          )}
          {installedModels.status === 'ready' && !installedTags.length && (
            <p role="status" className="mt-1.5 text-xs">Ollama has no models installed yet.</p>
          )}
        </div>
      )}
      <label className="block text-xs font-medium mb-1.5" htmlFor="ollama-model-tag" style={{ color: 'var(--color-text-secondary)' }}>
        Ollama model tag
      </label>
      <input
        id="ollama-model-tag"
        aria-label="Ollama model tag"
        type="text"
        value={modelTag}
        onChange={(event) => chooseTag(event.target.value)}
        placeholder="e.g. qwen3.5:4b"
        autoComplete="off"
        spellCheck={false}
        maxLength={128}
        required
        disabled={busy}
        className="w-full text-sm px-3 py-2 rounded-lg outline-none mb-5"
        style={{ background: 'var(--color-bg-secondary)', color: 'var(--color-text)', border: '1px solid var(--color-border)' }}
      />
      <div
        className="flex items-start gap-2.5 px-3.5 py-3 rounded-lg text-xs mb-5"
        style={{ background: 'var(--color-accent-subtle)', color: 'var(--color-text-secondary)' }}
      >
        <ShieldCheck size={16} className="shrink-0" style={{ color: 'var(--color-accent)' }} />
        {showInstalledModels
          ? 'Your choice is saved when you continue.'
          : 'Going back leaves Ollama stopped and does not save this choice.'}
      </div>
      {(validationError || error) && <SetupError message={validationError || error} />}
      <form onSubmit={submit}>
        <button
          type="submit"
          disabled={busy}
          className="w-full py-3 px-4 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed disabled:opacity-70"
          style={{ background: 'var(--color-accent)', color: 'white' }}
        >
          {busy && <Loader2 size={16} className="animate-spin" />}
          {busy ? 'Starting local setup...' : 'Use this model and continue'}
        </button>
      </form>
    </SetupFrame>
  );
}

function SetupError({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="px-3.5 py-3 rounded-lg text-sm mb-4"
      style={{
        background: 'color-mix(in srgb, var(--color-error) 10%, transparent)',
        border: '1px solid color-mix(in srgb, var(--color-error) 20%, transparent)',
        color: 'var(--color-error)',
      }}
    >
      {message}
    </div>
  );
}

export function CustomServerSetupGuide({ engine, inSettings = false }: { engine: string; inSettings?: boolean }) {
  return (
    <SetupSteps>
      <li>{engine === 'lmstudio' ? (
        <>In <a href="https://lmstudio.ai/docs/developer/core/server" target="_blank" rel="noopener noreferrer" className="underline">LM Studio</a>, download and load a model, then turn on <strong>Start server</strong> in the <strong>Developer</strong> tab.</>
      ) : 'Start your OpenAI-compatible server with a model loaded.'}</li>
      <li>
        {isOfflineDistribution() ? 'Enter a loopback URL' : 'Enter the server URL'}, including <code>/v1</code>, and the exact model ID it serves.
        {engine === 'lmstudio' && <> The default LM Studio URL is <code>http://localhost:1234/v1</code>.</>}
        {isOfflineDistribution() && <> Use a server on this computer with <code>localhost</code> or <code>127.0.0.1</code>.</>}
      </li>
      <li>{inSettings
        ? <>Click <strong>Save inference source</strong> and reopen A.C.E.</>
        : <>Click <strong>Save and connect</strong>.</>}</li>
    </SetupSteps>
  );
}

export function CustomEndpointSetup({
  onCancel,
  onConfirm,
  busy = false,
  error = '',
}: {
  onCancel: () => void;
  onConfirm: (source: InferenceSourceSubmission) => void;
  busy?: boolean;
  error?: string;
}) {
  const offlineDistribution = isOfflineDistribution();
  const [host, setHost] = useState('http://localhost:1234/v1');
  const [model, setModel] = useState('');
  const [engine, setEngine] = useState('lmstudio');
  const [apiKey, setApiKey] = useState('');

  const submit = (event: FormEvent) => {
    event.preventDefault();
    onConfirm({
      kind: 'custom',
      host: host.trim(),
      model: model.trim(),
      engine,
      apiKey: offlineDistribution ? undefined : apiKey.trim() || undefined,
    });
  };

  const fieldStyle = {
    background: 'var(--color-surface)',
    color: 'var(--color-text)',
    border: '1px solid var(--color-border)',
  };

  return (
    <SetupFrame>
      <BackButton onClick={onCancel} label="Cancel" disabled={busy} />
      <form onSubmit={submit} className="flex flex-col gap-4">
        <div>
          <h2 className="text-base font-semibold mb-1" style={{ color: 'var(--color-text)' }}>
            Connect your server
          </h2>
          <CustomServerSetupGuide engine={engine} />
        </div>
        <label className="text-xs font-medium" style={{ color: 'var(--color-text-secondary)' }}>
          Server URL
          <input
            aria-label="Server URL"
            type="url"
            required
            value={host}
            onChange={(event) => setHost(event.target.value)}
            className="block w-full mt-1.5 px-3 py-2 rounded-lg text-sm outline-none"
            style={fieldStyle}
          />
        </label>
        <label className="text-xs font-medium" style={{ color: 'var(--color-text-secondary)' }}>
          Model ID
          <input
            aria-label="Model ID"
            type="text"
            required
            value={model}
            onChange={(event) => setModel(event.target.value)}
            placeholder="qwen2.5-7b-instruct"
            className="block w-full mt-1.5 px-3 py-2 rounded-lg text-sm outline-none"
            style={fieldStyle}
          />
        </label>
        <label className="text-xs font-medium" style={{ color: 'var(--color-text-secondary)' }}>
          Server type
          <select
            aria-label="Server type"
            value={engine}
            onChange={(event) => setEngine(event.target.value)}
            className="block w-full mt-1.5 px-3 py-2 rounded-lg text-sm outline-none"
            style={fieldStyle}
          >
            <option value="lmstudio">LM Studio</option>
            <option value="vllm">vLLM</option>
            <option value="sglang">SGLang</option>
            <option value="llamacpp">llama.cpp</option>
            <option value="mlx">MLX</option>
          </select>
        </label>
        {!offlineDistribution && <label className="text-xs font-medium" style={{ color: 'var(--color-text-secondary)' }}>
          API key (optional)
          <input
            aria-label="API key"
            type="password"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            autoComplete="off"
            className="block w-full mt-1.5 px-3 py-2 rounded-lg text-sm outline-none"
            style={fieldStyle}
          />
        </label>}
        {error && <SetupError message={error} />}
        <button
          type="submit"
          disabled={busy || !host.trim() || !model.trim()}
          className="w-full py-3 px-4 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
          style={{ background: 'var(--color-accent)', color: 'white' }}
        >
          {busy && <Loader2 size={16} className="animate-spin" />}
          {busy ? 'Connecting...' : 'Save and connect'}
        </button>
      </form>
    </SetupFrame>
  );
}

export function CloudflareConsent({ onBack, onConfirm }: {
  onBack: () => void;
  onConfirm: (model: string) => Promise<void>;
}) {
  return (
    <SetupFrame>
      <div className="space-y-4">
        <h2 className="font-semibold" style={{ color: 'var(--color-text)' }}>Configure Cloudflare Workers AI</h2>
        <CloudflareSetupForm onBack={onBack} onConfirm={onConfirm} />
      </div>
    </SetupFrame>
  );
}

export function InferenceSourceSetup({
  onStarted,
  onCancel,
}: {
  onStarted: () => void;
  onCancel?: () => void;
}) {
  const [step, dispatch] = useReducer(inferenceSetupReducer, 'choose');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const goBack = () => {
    setError('');
    dispatch('back');
  };

  const begin = async (source: InferenceSourceSubmission) => {
    setBusy(true);
    setError('');
    try {
      await persistAndStartInferenceSource(source);
      onStarted();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  if (step === 'ollama') {
    return (
      <OllamaConsent
        busy={busy}
        error={error}
        onBack={goBack}
        onConfirm={(model) => void begin({ kind: 'ollama', model })}
      />
    );
  }
  if (step === 'custom') {
    return (
      <CustomEndpointSetup
        busy={busy}
        error={error}
        onCancel={goBack}
        onConfirm={(source) => void begin(source)}
      />
    );
  }
  if (step === 'cloudflare') {
    return (
      <CloudflareConsent
        onBack={goBack}
        onConfirm={async (model) => {
          await persistAndStartInferenceSource({ kind: 'cloudflare', model });
          onStarted();
        }}
      />
    );
  }
  return (
    <InferenceSourceChooser
      onCancel={onCancel}
      onChoose={(source) => dispatch(`choose_${source}`)}
    />
  );
}
