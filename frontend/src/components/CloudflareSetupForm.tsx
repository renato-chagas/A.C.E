import { useEffect, useRef, useState, type FormEvent } from 'react';
import {
  CLOUDFLARE_GEMMA_MODEL,
  getCloudflareCredentialsStatus,
  isTauri,
  removeCloudflareCredentials,
  saveCloudflareCredentials,
  type CloudflareCredentialStatus,
} from '../lib/api';

const models = [
  { id: CLOUDFLARE_GEMMA_MODEL, name: 'Gemma 4 26B' },
  { id: '@cf/qwen/qwen3.8-27b', name: 'Qwen 3.8 27B' },
  { id: '@cf/openai/gpt-oss-20b', name: 'GPT-OSS 20B' },
  { id: '@cf/meta/llama-3.3-70b-instruct-fp8-fast', name: 'Llama 3.3 70B' },
];
const modelPattern = /^@(?:cf|hf)\/[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/;
const inputClass = 'block w-full mt-1.5 px-3 py-2 rounded-lg text-sm border bg-transparent';
const inputStyle = { borderColor: 'var(--color-border)', color: 'var(--color-text)' };

export function CloudflareSetupForm({
  inSettings = false, initialModel = CLOUDFLARE_GEMMA_MODEL, onConfirm, onBack, onBusyChange,
}: {
  inSettings?: boolean;
  initialModel?: string;
  onConfirm: (model: string) => Promise<void>;
  onBack?: () => void;
  onBusyChange?: (busy: boolean) => void;
}) {
  const desktop = isTauri();
  const accountField = useRef<HTMLInputElement>(null);
  const tokenField = useRef<HTMLInputElement>(null);
  const operation = useRef(false);
  const [status, setStatus] = useState<CloudflareCredentialStatus | null>(null);
  const [selectedModel, setSelectedModel] = useState(() => models.some(({ id }) => id === initialModel) ? initialModel : 'custom');
  const [customModel, setCustomModel] = useState(initialModel === CLOUDFLARE_GEMMA_MODEL ? '' : initialModel);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  useEffect(() => {
    if (!desktop) return;
    let active = true;
    getCloudflareCredentialsStatus().then((current) => {
      if (active) setStatus(current);
    }, () => {
      if (active) {
        setStatus('missing');
        setError('Could not read Cloudflare credentials from your system credential store.');
      }
    });
    return () => { active = false; };
  }, [desktop]);

  function setOperation(working: boolean) {
    operation.current = working;
    setBusy(working);
    onBusyChange?.(working);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!desktop || status === null || operation.current) return;
    setError('');
    setNotice('');

    const model = selectedModel === 'custom' ? customModel.trim() : selectedModel;
    if (model.length > 128 || !modelPattern.test(model)) {
      setError('Enter a Cloudflare catalog model ID like @cf/provider/model-name.');
      return;
    }

    const accountId = accountField.current?.value.trim() ?? '';
    let apiToken = tokenField.current?.value.trim() ?? '';
    const replaceCredentials = Boolean(accountId || apiToken) || status === 'missing';
    if (replaceCredentials) {
      if (!/^[0-9a-f]{32}$/i.test(accountId)) {
        setError('Enter the 32-character Cloudflare Account ID.');
        return;
      }
      if (!apiToken || apiToken.length > 512 || !/^[!-~]+$/.test(apiToken)) {
        setError('Paste the Workers AI API token by itself, without spaces.');
        return;
      }
    }

    setOperation(true);
    try {
      if (replaceCredentials) {
        if (tokenField.current) tokenField.current.value = '';
        const saved = saveCloudflareCredentials(accountId, apiToken);
        apiToken = '';
        await saved;
        setStatus('stored');
        if (accountField.current) accountField.current.value = '';
      }
      await onConfirm(model);
      if (inSettings) setNotice('Saved. Restart A.C.E. to use this model.');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not configure Cloudflare.');
    } finally {
      setOperation(false);
    }
  }

  async function removeSavedCredentials() {
    if (operation.current) return;
    setError('');
    setNotice('');
    setOperation(true);
    try {
      await removeCloudflareCredentials();
      setStatus('missing');
      if (accountField.current) accountField.current.value = '';
      if (tokenField.current) tokenField.current.value = '';
      setNotice('Saved credentials removed. Restart A.C.E. to end the current connection.');
      setStatus(await getCloudflareCredentialsStatus());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not remove saved Cloudflare credentials.');
    } finally {
      setOperation(false);
    }
  }

  return (
    <form onSubmit={(event) => void submit(event)} className="space-y-4" autoComplete="off">
      <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
        Online chat sends messages and conversation history to Cloudflare Workers AI.
      </p>
      {desktop ? (
        <>
          <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
            Get your Account ID and token under <strong>Workers AI → Use REST API</strong>.{' '}
            <a href="https://dash.cloudflare.com/" target="_blank" rel="noopener noreferrer" className="underline">Open Cloudflare dashboard</a>
          </p>
          <fieldset disabled={busy} className="space-y-3">
            <label className="block text-sm">
              Cloudflare Account ID
              <input ref={accountField} className={inputClass} style={inputStyle} autoComplete="off" spellCheck={false} autoCapitalize="off" />
            </label>
            <label className="block text-sm">
              Workers AI API Token
              <input ref={tokenField} type="password" className={inputClass} style={inputStyle} autoComplete="new-password" spellCheck={false} maxLength={512} />
            </label>
            <p className="text-xs" style={{ color: 'var(--color-text-secondary)' }}>
              {status === 'stored' && 'Credentials saved in your system credential store. Leave both fields blank to keep them.'}
              {status === 'environment' && 'Credentials are available for this session. Fill in both fields to save them for future launches.'}
              {status === 'missing' && 'A.C.E. saves these credentials in your system credential store.'}
              {status === null && 'Checking for saved credentials…'}
            </p>
            <label className="block text-sm">
              Cloudflare model
              <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)} className={inputClass} style={inputStyle}>
                {models.map(({ id, name }) => <option key={id} value={id}>{name}</option>)}
                <option value="custom">Enter model ID</option>
              </select>
            </label>
            {selectedModel === 'custom' && (
              <label className="block text-sm">
                Cloudflare model ID
                <input value={customModel} onChange={(event) => setCustomModel(event.target.value)} className={inputClass} style={inputStyle} placeholder="@cf/provider/model-name" maxLength={128} autoComplete="off" spellCheck={false} />
              </label>
            )}
            <a href="https://developers.cloudflare.com/workers-ai/models/" target="_blank" rel="noopener noreferrer" className="text-xs underline">Browse Cloudflare model catalog</a>
          </fieldset>
          {error && <p role="alert" className="text-sm" style={{ color: 'var(--color-error)' }}>{error}</p>}
          {notice && <p role="status" className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>{notice}</p>}
          <div className="flex flex-wrap items-center gap-3">
            <button type="submit" disabled={busy || status === null} className="rounded-lg px-4 py-2 text-sm font-semibold disabled:opacity-50" style={{ background: 'var(--color-accent)', color: 'white' }}>
              {inSettings ? 'Save Cloudflare for next launch' : busy ? 'Connecting to Cloudflare…' : 'Connect to Cloudflare'}
            </button>
            {inSettings && status === 'stored' && (
              <button type="button" onClick={() => void removeSavedCredentials()} disabled={busy} className="text-xs underline disabled:opacity-50">
                Remove saved credentials
              </button>
            )}
          </div>
        </>
      ) : (
        <p className="text-sm">Open the A.C.E. desktop app to configure Cloudflare.</p>
      )}
      {onBack && <button type="button" onClick={onBack} disabled={busy} className="text-sm underline disabled:opacity-50">Back</button>}
    </form>
  );
}
