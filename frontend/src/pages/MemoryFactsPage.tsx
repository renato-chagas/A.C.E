import { useEffect, useRef, useState, type FormEvent } from 'react';
import { fetchMemoryFacts, reviseMemoryFact, type MemoryFact } from '../lib/local-facts';

const muted = { color: 'var(--color-text-secondary)' };
const border = { borderColor: 'var(--color-border)' };
const button = 'rounded-lg border px-3 py-2 text-sm disabled:opacity-50';

export function MemoryFactsPage() {
  const [facts, setFacts] = useState<MemoryFact[]>([]);
  const [loading, setLoading] = useState(true);
  const [version, setVersion] = useState(0);
  const [loadError, setLoadError] = useState('');
  const [editing, setEditing] = useState<MemoryFact | null>(null);
  const [correction, setCorrection] = useState('');
  const [saveError, setSaveError] = useState('');
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const saveRequest = useRef<AbortController | null>(null);

  useEffect(() => {
    const request = new AbortController();
    void fetchMemoryFacts(request.signal).then(
      (rows) => {
        if (request.signal.aborted) return;
        setFacts(rows);
        setLoading(false);
      },
      (error: Error) => {
        if (request.signal.aborted) return;
        setLoadError(error.message);
        setLoading(false);
      },
    );
    return () => request.abort();
  }, [version]);

  useEffect(() => () => saveRequest.current?.abort(), []);

  const refresh = () => {
    setFacts([]);
    setLoadError('');
    setLoading(true);
    setVersion((current) => current + 1);
  };

  const openCorrection = (fact: MemoryFact) => {
    setEditing(fact);
    setCorrection(fact.text);
    setSaveError('');
    setSaved(false);
  };

  const save = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!editing || saveRequest.current) return;
    const request = new AbortController();
    saveRequest.current = request;
    setSaving(true);
    setSaveError('');
    try {
      await reviseMemoryFact(editing.text, correction.trim(), request.signal);
      if (request.signal.aborted) return;
      setEditing(null);
      setCorrection('');
      setSaved(true);
      refresh();
    } catch (error) {
      if (!request.signal.aborted) {
        setSaveError(error instanceof Error ? error.message : 'Could not save this correction.');
      }
    } finally {
      saveRequest.current = null;
      if (!request.signal.aborted) setSaving(false);
    }
  };

  return (
    <main className="h-full min-h-0 overflow-y-auto p-5 md:p-8" style={{ color: 'var(--color-text)' }}>
      <div className="mx-auto w-full max-w-4xl">
        <header className="mb-5 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold">Memory facts</h1>
            <p className="mt-1 text-sm" style={muted}>Review what ACE remembers across conversations.</p>
          </div>
          <button type="button" onClick={refresh} disabled={loading || editing !== null} className={button} style={border}>Refresh</button>
        </header>

        {saved && <p role="status" className="mb-4 text-sm">Correction saved. The previous wording was kept in history.</p>}
        {loading && <p role="status" className="text-sm" style={muted}>Loading saved facts…</p>}
        {loadError && <p role="alert" className="text-sm" style={{ color: 'var(--color-error)' }}>{loadError}</p>}
        {!loading && !loadError && facts.length === 0 && <p className="text-sm" style={muted}>No facts have been saved yet.</p>}

        <ul className="space-y-3">
          {facts.map((fact, index) => (
            <li key={index} className="rounded-xl border p-4" style={{ ...border, background: 'var(--color-bg-secondary)' }}>
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="mb-1 flex flex-wrap gap-x-2 text-xs font-medium" style={muted}>
                    <span>{fact.superseded ? 'Superseded' : fact.trusted_for_recall ? 'Used in recall' : 'Quarantined'}</span>
                    {fact.source && <span className="break-all">{fact.source}</span>}
                  </div>
                  <p className="break-words whitespace-pre-wrap">{fact.text}</p>
                </div>
                {!editing && fact.trusted_for_recall && (
                  <button type="button" onClick={() => openCorrection(fact)} className={button} style={border}>Correct</button>
                )}
              </div>
              {editing === fact && (
                <form onSubmit={save} className="mt-3 space-y-2">
                  <label htmlFor="fact-correction" className="block text-sm font-medium">Corrected wording</label>
                  <input
                    autoFocus
                    id="fact-correction"
                    value={correction}
                    onChange={(event) => setCorrection(event.target.value)}
                    autoComplete="off"
                    disabled={saving}
                    required
                    maxLength={1000}
                    className="w-full rounded-lg border px-3 py-2"
                    style={{ ...border, background: 'var(--color-bg)' }}
                  />
                  {saveError && <p role="alert" className="text-sm" style={{ color: 'var(--color-error)' }}>{saveError}</p>}
                  <div className="flex gap-2">
                    <button
                      type="submit"
                      disabled={saving || !correction.trim() || correction.trim().toLowerCase() === fact.text.toLowerCase()}
                      className={button}
                      style={{ ...border, color: 'var(--color-accent)' }}
                    >
                      {saving ? 'Saving…' : 'Save correction'}
                    </button>
                    <button type="button" disabled={saving} onClick={() => {
                      setEditing(null);
                      setCorrection('');
                    }} className={button} style={border}>Cancel</button>
                  </div>
                </form>
              )}
            </li>
          ))}
        </ul>
      </div>
    </main>
  );
}
