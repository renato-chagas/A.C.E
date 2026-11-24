import { useCallback, useEffect, useRef, useState } from 'react';
import {
  createLocalMonitorClient,
  type LocalMonitorStatus,
  type MonitorAnalysisResult,
  type MonitorAssessment,
  type MonitorIncident,
} from '../lib/local-monitor';

const muted = { color: 'var(--color-text-secondary)' };
const border = { borderColor: 'var(--color-border)' };
const buttonStyle = 'rounded-lg border px-3 py-2 text-sm disabled:opacity-50';
const metrics = new Map([
  ['cpu_high', 'High CPU use'],
  ['memory_low', 'Low available memory'],
  ['disk_low', 'Low free disk space'],
]);

function MonitorTime({ value }: { value: string }) {
  const date = new Date(value);
  return <time dateTime={value}>{Number.isNaN(date.getTime()) ? value : date.toLocaleString()}</time>;
}

type MonitorChange = { enabled: boolean; notificationsEnabled?: boolean } | { incidentId: number };

type AnalysisRequest = { incidentId: number; useLocalModel: boolean };
type AnalysisFeedback = { result: MonitorAnalysisResult; requestedModel: boolean } | { error: string };
type AnalysisControls = {
  localModelAvailable: boolean;
  pending: AnalysisRequest | null;
  feedback: ReadonlyMap<number, AnalysisFeedback>;
  start: (request: AnalysisRequest) => void;
};

function AssessmentText({ assessment }: { assessment: MonitorAssessment }) {
  return (
    <>
      <p className="text-xs" style={muted}><MonitorTime value={assessment.created_at} /></p>
      <p className="whitespace-pre-wrap break-words">{assessment.analysis}</p>
    </>
  );
}

function Incident({ incident, changing, reviewing, onReview, analysis }: {
  incident: MonitorIncident;
  changing: boolean;
  reviewing: boolean;
  onReview: () => void;
  analysis?: AnalysisControls;
}) {
  const feedback = analysis?.feedback.get(incident.id);
  const outcome = feedback && 'result' in feedback ? feedback : null;
  const currentAnalysis = analysis?.pending?.incidentId === incident.id ? analysis.pending : null;
  const savedAssessments = incident.assessments.filter((item) => item.id !== outcome?.result.assessment.id);
  const severity = incident.severity === 'critical' ? 'Critical'
    : incident.severity === 'warning' ? 'Warning' : 'Unknown severity';
  const status = incident.status === 'resolved' ? 'Recovered'
    : incident.status === 'active' ? 'Active' : 'Unknown status';
  return (
    <li className="min-w-0 space-y-3 rounded-xl border p-4" style={border}>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="font-medium">{metrics.get(incident.metric) ?? 'Resource alert'}</h3>
        <p aria-live="polite" className="text-sm" style={muted}>{severity} · {status}{incident.acknowledged_at && ' · Reviewed'}</p>
      </div>
      <p className="break-words text-sm">{incident.summary}</p>
      <dl className="grid gap-2 text-xs sm:grid-cols-2" style={muted}>
        <div><dt className="font-medium">First detected</dt><dd><MonitorTime value={incident.first_seen} /></dd></div>
        <div><dt className="font-medium">Last observed</dt><dd><MonitorTime value={incident.last_seen} /></dd></div>
        {incident.resolved_at && (
          <div><dt className="font-medium">Recovered</dt><dd><MonitorTime value={incident.resolved_at} /></dd></div>
        )}
        <div><dt className="font-medium">Occurrences in the last 30 days</dt><dd>{incident.episodes_last_30d}</dd></div>
      </dl>
      {savedAssessments.length > 0 && (
        <details className="border-t pt-3 text-sm" style={border}>
          <summary className="cursor-pointer">Saved assessments</summary>
          <ul className="mt-2 space-y-3">
            {savedAssessments.map((item) => (
              <li key={item.id}><AssessmentText assessment={item} /></li>
            ))}
          </ul>
        </details>
      )}
      <div className="flex flex-wrap gap-2">
        {analysis && metrics.has(incident.metric) && (
          <>
            <button
              type="button"
              onClick={() => analysis.start({ incidentId: incident.id, useLocalModel: false })}
              disabled={analysis.pending !== null}
              className={buttonStyle}
              style={border}
            >
              Analyze incident
            </button>
            {analysis.localModelAvailable && (
              <button
                type="button"
                onClick={() => analysis.start({ incidentId: incident.id, useLocalModel: true })}
                disabled={analysis.pending !== null}
                className={buttonStyle}
                style={border}
              >
                Analyze with local model
              </button>
            )}
          </>
        )}
        {!incident.acknowledged_at && (
          <button type="button" onClick={onReview} disabled={changing} className={buttonStyle} style={border}>
            {reviewing ? 'Marking as reviewed…' : 'Mark as reviewed'}
          </button>
        )}
      </div>
      {currentAnalysis && (
        <p role="status" className="text-sm" style={muted}>
          {currentAnalysis.useLocalModel ? 'Analyzing with a local model…' : 'Analyzing saved measurements…'}
        </p>
      )}
      {feedback && 'error' in feedback && (
        <p role="alert" className="text-sm" style={{ color: 'var(--color-error)' }}>{feedback.error}</p>
      )}
      {outcome && (
        <div className="space-y-1 border-t pt-3 text-sm" style={border}>
          <p role="status" className="font-medium">
            {outcome.result.mode === 'local_model' ? 'Local model assessment saved.'
              : outcome.requestedModel ? 'Used saved measurements. Measurement assessment saved.'
                : 'Measurement assessment saved.'}
          </p>
          <AssessmentText assessment={outcome.result.assessment} />
        </div>
      )}
    </li>
  );
}

export function LocalMonitorOverview({ status, change = null, onChange, analysis }: {
  status: LocalMonitorStatus;
  change?: MonitorChange | null;
  onChange: (change: MonitorChange) => void;
  analysis?: AnalysisControls;
}) {
  const [requestNotifications, setRequestNotifications] = useState(false);
  const changing = change !== null;
  const changingMonitor = change && 'enabled' in change;
  const reviewingId = change && 'incidentId' in change ? change.incidentId : null;
  const stopping = !status.enabled && status.running;
  const statusText = status.enabled
    ? (status.running ? 'Monitoring is running.' : 'Monitoring is starting.')
    : (stopping ? 'Monitoring is stopping.' : 'Monitoring is off.');
  return (
    <>
      <section aria-labelledby="monitor-state" className="space-y-3 rounded-xl border p-4" style={border}>
        <h2 id="monitor-state" className="font-medium">Monitoring</h2>
        <p role="status" className="text-sm" style={muted}>
          {statusText}
        </p>
        <p className="text-sm" style={muted}>
          ACE checks CPU use, available memory and free disk space. Resource alerts are saved on this computer.
        </p>
        {status.externally_managed ? (
          <p className="text-sm" style={muted}>
            This monitor was started outside this panel. Use the same tool to stop it.
          </p>
        ) : (
          <>
            <p className="text-sm" style={muted}>
              Once enabled, monitoring continues after the app closes, until you disable it or sign out.
            </p>
            {status.enabled ? (
              <p className="text-sm" style={muted}>
                Desktop notifications are {status.notifications_enabled ? 'on' : 'off'}. Disable and restart monitoring to change this setting.
              </p>
            ) : !stopping && (
              <label className="flex items-start gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={requestNotifications}
                  disabled={changing}
                  onChange={(event) => setRequestNotifications(event.target.checked)}
                  className="mt-1"
                />
                Show desktop notifications for resource alerts and recovery
              </label>
            )}
            <button
              type="button"
              onClick={() => {
                onChange({ enabled: !status.enabled, notificationsEnabled: !status.enabled && requestNotifications });
                setRequestNotifications(false);
              }}
              disabled={changing || stopping}
              className={buttonStyle}
              style={border}
            >
              {changingMonitor || stopping ? 'Updating monitoring…' : status.enabled ? 'Disable monitoring' : 'Enable monitoring'}
            </button>
          </>
        )}
      </section>

      <section aria-labelledby="monitor-history" className="space-y-3">
        <h2 id="monitor-history" className="text-lg font-medium">Recent alerts</h2>
        {!status.history_available ? (
          <p role="status" className="text-sm" style={muted}>Alert history is temporarily unavailable.</p>
        ) : status.incidents.length === 0 ? (
          <p className="text-sm" style={muted}>No resource alerts have been recorded.</p>
        ) : (
          <ul className="space-y-3">
            {status.incidents.map((incident) => (
              <Incident
                key={incident.id}
                incident={incident}
                changing={changing}
                reviewing={reviewingId === incident.id}
                analysis={analysis}
                onReview={() => onChange({ incidentId: incident.id })}
              />
            ))}
          </ul>
        )}
      </section>
    </>
  );
}

export function LocalMonitorPage({ localModelAvailable = false }: { localModelAvailable?: boolean }) {
  const [client] = useState(createLocalMonitorClient);
  const [status, setStatus] = useState<LocalMonitorStatus | null>(null);
  const [error, setError] = useState('');
  const [change, setChange] = useState<MonitorChange | null>(null);
  const [loading, setLoading] = useState(false);
  const request = useRef<{ controller: AbortController; changing: boolean } | null>(null);
  const analysisRequest = useRef<AbortController | null>(null);
  const [pendingAnalysis, setPendingAnalysis] = useState<AnalysisRequest | null>(null);
  const [analysisFeedback, setAnalysisFeedback] = useState<ReadonlyMap<number, AnalysisFeedback>>(() => new Map());

  async function analyze(selection: AnalysisRequest) {
    if (!client || analysisRequest.current || (selection.useLocalModel && !localModelAvailable)) return;
    const controller = new AbortController();
    const { incidentId, useLocalModel } = selection;
    analysisRequest.current = controller;
    setPendingAnalysis(selection);
    setAnalysisFeedback((previous) => {
      const next = new Map(previous);
      next.delete(incidentId);
      return next;
    });
    try {
      const result = await client.analyze(incidentId, useLocalModel, controller.signal);
      if (!controller.signal.aborted) {
        setAnalysisFeedback((previous) => new Map(previous).set(incidentId, { result, requestedModel: useLocalModel }));
      }
    } catch (cause) {
      if (!controller.signal.aborted) {
        const error = cause instanceof Error ? cause.message : 'Could not analyze this alert.';
        setAnalysisFeedback((previous) => new Map(previous).set(incidentId, { error }));
      }
    } finally {
      if (analysisRequest.current === controller) {
        analysisRequest.current = null;
        setPendingAnalysis(null);
      }
    }
  }

  const update = useCallback(async (mutation?: MonitorChange) => {
    if (!client) return;
    const writing = mutation !== undefined;
    if (request.current && (!writing || request.current.changing)) return;
    request.current?.controller.abort();
    const controller = new AbortController();
    request.current = { controller, changing: writing };
    setChange(mutation ?? null);
    setLoading(!writing);
    try {
      const signal = controller.signal;
      const result = mutation === undefined
        ? await client.status(signal)
        : 'enabled' in mutation
          ? await client.setEnabled(mutation.enabled, signal, mutation.notificationsEnabled)
          : await client.acknowledge(mutation.incidentId, signal);
      if (controller.signal.aborted) return;
      setStatus(result);
      setError('');
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : 'The local monitoring request failed.');
      }
    } finally {
      if (request.current?.controller === controller) {
        request.current = null;
        setChange(null);
        setLoading(false);
      }
    }
  }, [client]);

  useEffect(() => {
    if (!client) return;
    void update();
    const interval = window.setInterval(() => {
      if (document.visibilityState === 'visible') void update();
    }, 15000);
    return () => {
      window.clearInterval(interval);
      request.current?.controller.abort();
      request.current = null;
      analysisRequest.current?.abort();
      analysisRequest.current = null;
    };
  }, [client, update]);

  return (
    <section className="h-full min-h-0 overflow-y-auto p-5 md:p-8" style={{ color: 'var(--color-text)' }}>
      <div className="mx-auto w-full max-w-5xl space-y-5">
        <header className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-2xl font-semibold">System monitoring</h1>
          {client && (
            <button type="button" onClick={() => void update()} disabled={loading || change !== null}
              className={buttonStyle} style={border}>
              Refresh
            </button>
          )}
        </header>
        {!client ? (
          <p role="alert" className="text-sm" style={muted}>Open ACE on this computer and use a direct local API address in Settings.</p>
        ) : (
          <>
            {error && <p role="alert" className="text-sm" style={{ color: 'var(--color-error)' }}>{error}</p>}
            {!status && !error && <p role="status" className="text-sm" style={muted}>Loading monitoring status…</p>}
            {status && (
              <LocalMonitorOverview
                status={status}
                change={change}
                onChange={(mutation) => void update(mutation)}
                analysis={{
                  localModelAvailable,
                  pending: pendingAnalysis,
                  feedback: analysisFeedback,
                  start: (selection) => void analyze(selection),
                }}
              />
            )}
          </>
        )}
      </div>
    </section>
  );
}
