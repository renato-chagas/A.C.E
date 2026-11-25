import { authHeaders } from './api';
import { resolveLoopbackApiOrigin } from './loopback-api';

export type MonitorAssessment = {
  id: number;
  analysis: string;
  created_at: string;
};

export type MonitorAnalysisResult = {
  incident_id: number;
  assessment: MonitorAssessment;
  mode: 'measurements' | 'local_model';
};

export type MonitorIncident = {
  id: number;
  metric: string;
  severity: string;
  status: string;
  summary: string;
  first_seen: string;
  last_seen: string;
  resolved_at: string | null;
  acknowledged_at: string | null;
  observations: number;
  episodes_last_30d: number;
  assessments: MonitorAssessment[];
};

export type MonitorReadings = {
  cpu_percent: number | null;
  memory_available_gb: number | null;
  disk_free_gb: number | null;
};

export type LocalMonitorStatus = {
  enabled: boolean;
  running: boolean;
  externally_managed: boolean;
  notifications_enabled: boolean;
  history_available: boolean;
  incidents: MonitorIncident[];
  readings?: MonitorReadings;
};

function isCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

function isReading(value: unknown, limit = Infinity): value is number | null {
  return value === null ||
    (typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= limit);
}

function isReadings(value: unknown): value is MonitorReadings {
  if (!value || typeof value !== 'object') return false;
  const readings = value as Partial<MonitorReadings>;
  return isReading(readings.cpu_percent, 100) &&
    isReading(readings.memory_available_gb) &&
    isReading(readings.disk_free_gb);
}

function isAssessment(value: unknown): value is MonitorAssessment {
  if (!value || typeof value !== 'object') return false;
  const item = value as Partial<MonitorAssessment>;
  return isCount(item.id) &&
    typeof item.analysis === 'string' &&
    typeof item.created_at === 'string';
}

function isIncident(value: unknown): value is MonitorIncident {
  if (!value || typeof value !== 'object') return false;
  const item = value as Partial<MonitorIncident>;
  return isCount(item.id) &&
    typeof item.metric === 'string' &&
    typeof item.severity === 'string' &&
    typeof item.status === 'string' &&
    typeof item.summary === 'string' &&
    typeof item.first_seen === 'string' &&
    typeof item.last_seen === 'string' &&
    (item.resolved_at === null || typeof item.resolved_at === 'string') &&
    (item.acknowledged_at === null || typeof item.acknowledged_at === 'string') &&
    isCount(item.observations) &&
    isCount(item.episodes_last_30d) &&
    Array.isArray(item.assessments) && item.assessments.every(isAssessment);
}

export function createLocalMonitorClient() {
  const origin = resolveLoopbackApiOrigin();
  if (!origin) return null;
  const endpoint = origin + '/v1/local-monitor';

  async function fetchLocal(
    signal: AbortSignal,
    method: 'GET' | 'PATCH' | 'POST' = 'GET',
    path = '',
    body?: object,
  ) {
    try {
      return await fetch(endpoint + path, {
        method,
        headers: body ? authHeaders({ 'Content-Type': 'application/json' }) : authHeaders(),
        body: body ? JSON.stringify(body) : undefined,
        cache: 'no-store',
        credentials: 'omit',
        redirect: 'error',
        referrerPolicy: 'no-referrer',
        signal,
      });
    } catch (error) {
      if (signal.aborted) throw error;
      throw new Error('Could not reach the local ACE server.');
    }
  }

  function checkLocalAccess(response: Response) {
    if (response.status === 401) throw new Error('The local ACE connection needs a valid API key.');
    if (response.status === 403) throw new Error('Monitoring requires direct access from this computer.');
  }

  async function request(
    signal: AbortSignal,
    method: 'GET' | 'PATCH' | 'POST' = 'GET',
    path = '',
    body?: object,
  ): Promise<LocalMonitorStatus> {
    const response = await fetchLocal(signal, method, path, body);
    checkLocalAccess(response);

    if (response.status === 409) throw new Error('The monitor is busy or controlled by another session. Refresh to check its current state.');
    if (method === 'POST' && response.status === 404) throw new Error('This alert is no longer available. Refresh monitoring.');
    if (!response.ok) {
      if (method === 'POST') throw new Error('Could not mark the alert as reviewed. Refresh to check its current state.');
      throw new Error(method === 'PATCH'
        ? 'Could not change monitoring. Refresh to check its current state.'
        : 'Could not read the local monitoring status.');
    }

    const data: Partial<LocalMonitorStatus> | null = await response.json().catch(() => null);
    if (typeof data?.enabled !== 'boolean' || typeof data.running !== 'boolean' ||
        typeof data.externally_managed !== 'boolean' || typeof data.history_available !== 'boolean' ||
        (data.notifications_enabled !== undefined && typeof data.notifications_enabled !== 'boolean') ||
        !Array.isArray(data.incidents) || !data.incidents.every(isIncident)) {
      throw new Error('ACE returned an invalid monitoring response.');
    }
    return {
      enabled: data.enabled,
      running: data.running,
      externally_managed: data.externally_managed,
      notifications_enabled: data.notifications_enabled ?? false,
      history_available: data.history_available,
      incidents: data.incidents,
      ...(isReadings(data.readings) ? { readings: data.readings } : {}),
    };
  }

  return {
    status: (signal: AbortSignal) => request(signal),
    setEnabled(enabled: boolean, signal: AbortSignal, notificationsEnabled = false) {
      const options = enabled && notificationsEnabled ? { enabled, notifications_enabled: true } : { enabled };
      return request(signal, 'PATCH', '', options);
    },
    async acknowledge(incidentId: number, signal: AbortSignal) {
      if (!isCount(incidentId) || incidentId === 0) throw new Error('Choose a valid resource alert.');
      return request(signal, 'POST', '/incidents/' + incidentId + '/acknowledge', {});
    },
    async analyze(
      incidentId: number, useLocalModel: boolean, signal: AbortSignal,
    ): Promise<MonitorAnalysisResult> {
      if (!isCount(incidentId) || incidentId === 0) throw new Error('Choose a valid resource alert.');

      const response = await fetchLocal(
        signal, 'POST', '/incidents/' + incidentId + '/analyze',
        useLocalModel ? { use_local_model: true } : {},
      );
      checkLocalAccess(response);
      if (response.status === 404) throw new Error('This alert is no longer available. Refresh monitoring.');
      if (response.status === 409) throw new Error('This alert changed. Refresh monitoring and try again.');
      if (response.status === 422) throw new Error('This alert has no supported measurements to analyze.');
      if (response.status === 429) throw new Error('Analysis is busy. Try again in a moment.');
      if (!response.ok) throw new Error('Could not analyze this alert. Try again.');

      const data: Partial<MonitorAnalysisResult> | null = await response.json().catch(() => null);
      if (data?.incident_id !== incidentId || !isAssessment(data.assessment) ||
          !(data.mode === 'measurements' || (useLocalModel && data.mode === 'local_model'))) {
        throw new Error('ACE returned an invalid incident assessment.');
      }
      return { incident_id: data.incident_id, assessment: data.assessment, mode: data.mode };
    },
  };
}
