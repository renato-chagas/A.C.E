import { useEffect, useState } from 'react';
import {
  createCodeImprovementsClient,
  type CodeImprovementProposal,
  type CodeImprovementSummary,
} from '../lib/code-improvements';

const muted = { color: 'var(--color-text-secondary)' };
const border = { borderColor: 'var(--color-border)' };
const errorText = { color: 'var(--color-error)' };
const panel = 'rounded-xl border p-4';
const pre = 'max-h-[28rem] overflow-auto rounded-lg border p-3 text-xs font-mono';
const statuses: ReadonlyMap<string, string> = new Map([
  ['proposed', 'Proposed'],
  ['validated', 'Validated'],
  ['approved', 'Approved'],
  ['applying', 'Applying'],
  ['applied', 'Applied'],
  ['rolling_back', 'Rollback in progress'],
  ['rolled_back', 'Rolled back'],
  ['rejected', 'Rejected'],
]);

function statusLabel(status: string): string {
  return statuses.get(status) ?? 'Unknown status';
}

export function CodeImprovementDetail({ proposal }: { proposal: CodeImprovementProposal }) {
  const { proposal_id: id, status, rollback_available, validation_receipt } = proposal;
  const commands = [{ action: 'show', label: 'Review the current proposal' }];
  if (status === 'proposed' || (['validated', 'approved'].includes(status) && !validation_receipt)) {
    commands.push({ action: 'validate', label: 'Validate in a disposable copy' });
  }
  if (['validated', 'approved'].includes(status) && validation_receipt) {
    commands.push({ action: 'apply', label: 'Review and confirm application' });
  }
  if (['applied', 'rolling_back'].includes(status) && rollback_available) {
    commands.push({
      action: 'rollback',
      label: status === 'rolling_back' ? 'Resume rollback' : 'Review and confirm rollback',
    });
  }

  return (
    <article className="min-w-0 space-y-4">
      <header>
        <p className="text-sm" style={muted}>{statusLabel(status)}</p>
        <h2 className="break-words text-xl font-semibold">{proposal.title}</h2>
        <p className="mt-1 break-all text-sm" style={muted}>
          Repository <code>{proposal.repository}</code>
        </p>
      </header>

      <section className={panel} style={border} aria-labelledby="proposal-files">
        <h3 id="proposal-files" className="mb-2 font-medium">Files</h3>
        <ul className="space-y-1 break-all text-sm">
          {proposal.files.map((file, index) => <li key={index}><code>{file}</code></li>)}
        </ul>
      </section>

      <section className={panel} style={border} aria-labelledby="proposal-validation">
        <h3 id="proposal-validation" className="mb-2 font-medium">Validation</h3>
        {proposal.validation ? (
          <pre className={pre} style={border}>{proposal.validation}</pre>
        ) : (
          <p className="text-sm" style={muted}>No validation report is saved.</p>
        )}
        {validation_receipt && (
          <dl className="mt-3 text-sm">
            <dt style={muted}>Validation receipt</dt>
            <dd className="break-all"><code>{validation_receipt}</code></dd>
          </dl>
        )}
      </section>

      <section className={panel} style={border} aria-labelledby="proposal-patch">
        <h3 id="proposal-patch" className="mb-2 font-medium">Patch</h3>
        <pre className={pre} style={border}><code>{proposal.patch}</code></pre>
      </section>

      <section className={panel} style={border} aria-labelledby="proposal-terminal">
        <h3 id="proposal-terminal" className="font-medium">Terminal commands</h3>
        {['applying', 'rolling_back'].includes(status) && (
          <p className="mt-2 text-sm" style={muted}>If this operation was interrupted, the next improve command can run recovery.</p>
        )}
        <ul className="mt-2 space-y-2 text-sm">
          {commands.map(({ action, label }) => (
            <li key={action}>
              <p style={muted}>{label}</p>
              <code className="select-text break-all">jarvis improve {action} {id}</code>
            </li>
          ))}
        </ul>
      </section>
    </article>
  );
}

export function CodeImprovementsPage() {
  const [client] = useState(createCodeImprovementsClient);
  const [proposals, setProposals] = useState<CodeImprovementSummary[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [proposal, setProposal] = useState<CodeImprovementProposal | null>(null);
  const [listError, setListError] = useState('');
  const [detailError, setDetailError] = useState('');
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    if (!client) return;
    const controller = new AbortController();
    void client.list(controller.signal).then(
      (rows) => { if (!controller.signal.aborted) setProposals(rows); },
      (error: Error) => { if (!controller.signal.aborted) setListError(error.message); },
    );
    return () => controller.abort();
  }, [client, refresh]);

  useEffect(() => {
    if (!client || !selected) return;
    const controller = new AbortController();
    void client.detail(selected, controller.signal).then(
      (row) => { if (!controller.signal.aborted) setProposal(row); },
      (error: Error) => { if (!controller.signal.aborted) setDetailError(error.message); },
    );
    return () => controller.abort();
  }, [client, selected, refresh]);

  const refreshProposals = () => {
    setProposals(null);
    setProposal(null);
    setListError('');
    setDetailError('');
    setRefresh((value) => value + 1);
  };

  const selectProposal = (id: string) => {
    if (selected === id) return;
    setProposal(null);
    setDetailError('');
    setSelected(id);
  };

  return (
    <section className="h-full min-h-0 overflow-y-auto p-5 md:p-8" style={{ color: 'var(--color-text)' }}>
      <div className="mx-auto w-full max-w-5xl space-y-5">
        <header className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold">Code improvements</h1>
            <p className="mt-1 text-sm" style={muted}>Review the latest code proposals saved by ACE on this computer.</p>
          </div>
          {client && (
            <button
              type="button"
              onClick={refreshProposals}
              disabled={proposals === null && !listError}
              className="rounded-lg border px-3 py-2 text-sm disabled:opacity-50"
              style={border}
            >
              Refresh
            </button>
          )}
        </header>

        {!client ? (
          <p role="alert" className="text-sm" style={errorText}>Open ACE on this computer and use a direct local API address in Settings.</p>
        ) : (
          <div className="grid items-start gap-5 md:grid-cols-[minmax(12rem,18rem)_minmax(0,1fr)]">
            <aside className={panel} style={border} aria-label="Saved proposals">
              {listError && <p role="alert" className="text-sm" style={errorText}>{listError}</p>}
              {proposals === null && !listError && <p role="status" className="text-sm" style={muted}>Loading proposals…</p>}
              {proposals?.length === 0 && <p className="text-sm" style={muted}>No code proposals have been saved.</p>}
              {proposals && proposals.length > 0 && (
                <ul className="max-h-[36rem] space-y-1 overflow-y-auto">
                  {proposals.map((entry) => (
                    <li key={entry.proposal_id}>
                      <button
                        type="button"
                        aria-current={selected === entry.proposal_id ? 'true' : undefined}
                        className="w-full rounded-lg px-2 py-1.5 text-left text-sm"
                        style={{ background: selected === entry.proposal_id ? 'var(--color-accent-subtle)' : 'transparent' }}
                        onClick={() => selectProposal(entry.proposal_id)}
                      >
                        <span className="block break-words">{entry.title}</span>
                        <span className="block text-xs" style={muted}>{statusLabel(entry.status)} · {new Date(entry.created_at * 1000).toLocaleDateString()}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </aside>
            {proposal && proposal.proposal_id === selected ? (
              <CodeImprovementDetail proposal={proposal} />
            ) : detailError ? (
              <p role="alert" className="text-sm" style={errorText}>{detailError}</p>
            ) : selected ? (
              <p role="status" className="text-sm" style={muted}>Loading proposal…</p>
            ) : (
              <p className="text-sm" style={muted}>Select a proposal to review its patch and validation.</p>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
