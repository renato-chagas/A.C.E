import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { LocalMonitorOverview, LocalMonitorPage } from './LocalMonitorPage';
import type { LocalMonitorStatus, MonitorIncident } from '../lib/local-monitor';

vi.mock('../lib/api', () => ({
  getBase: () => 'http://127.0.0.1:8000',
  isTauri: () => true,
  authHeaders: () => ({}),
}));

const date = '2026-11-22T22:00:00+00:00';
const alert: MonitorIncident = {
  id: 1,
  metric: 'cpu_high',
  severity: 'warning',
  status: 'resolved',
  summary: '<img src="bad" />',
  first_seen: date,
  last_seen: date,
  resolved_at: date,
  acknowledged_at: date,
  observations: 10,
  episodes_last_30d: 2,
  assessments: [{ id: 2, analysis: '<script>alert(1)</script>', created_at: date }],
};
const status: LocalMonitorStatus = {
  enabled: false,
  running: false,
  externally_managed: false,
  notifications_enabled: false,
  history_available: true,
  incidents: [alert],
};

describe('system monitoring', () => {
  it('starts with status loading and no enable or disable control until the local status is known', () => {
    const html = renderToStaticMarkup(<LocalMonitorPage />);
    expect(html).toContain('System monitoring');
    expect(html).toContain('Loading monitoring status');
    expect(html).not.toContain('Enable monitoring');
    expect(html).not.toContain('Disable monitoring');
  });

  it('keeps recovered alerts and assessments visible while monitoring is off', () => {
    const html = renderToStaticMarkup(<LocalMonitorOverview status={status} onChange={vi.fn()} />);
    expect(html).toContain('Monitoring is off');
    expect(html).toContain('Enable monitoring');
    expect(html).toContain('after the app closes');
    expect(html).toContain('High CPU use');
    expect(html).toContain('Warning · Recovered · Reviewed');
    expect(html).toContain('Occurrences in the last 30 days');
    expect(html).toContain('Saved assessments');
    expect(html).toContain('&lt;img');
    expect(html).toContain('&lt;script&gt;');
    expect(html).not.toContain('<img');
    expect(html).not.toContain('<script');
  });

  it('keeps the stop control available when history is unavailable', () => {
    const starting = { ...status, enabled: true, history_available: false, incidents: [] };
    const html = renderToStaticMarkup(<LocalMonitorOverview status={starting} onChange={vi.fn()} />);
    expect(html).toContain('Monitoring is starting');
    expect(html).toContain('Disable monitoring');
    expect(html).toContain('Alert history is temporarily unavailable');
    expect(html).not.toContain('No resource alerts');
    expect(html).not.toContain('disabled=""');
  });

  it('shows current readings with no model or alerts', () => {
    const running = {
      ...status,
      enabled: true,
      running: true,
      incidents: [],
      readings: { cpu_percent: 0, memory_available_gb: 8, disk_free_gb: 123.5 },
    };
    const html = renderToStaticMarkup(<LocalMonitorOverview status={running} onChange={vi.fn()} />);
    expect(html).toContain('Live resource usage');
    expect(html).toContain('0.0%');
    expect(html).toContain('8.0 GiB');
    expect(html).toContain('123.5 GiB');
    expect(html).toContain('No resource alerts');
  });

  it('blocks changes while the daemon is stopping', () => {
    const stopping = { ...status, running: true };
    const html = renderToStaticMarkup(<LocalMonitorOverview status={stopping} onChange={vi.fn()} />);
    expect(html).toContain('Monitoring is stopping');
    expect(html).toContain('disabled=""');
    expect(html).not.toContain('Enable monitoring');
  });

  it('shows the running state and alerts of an external monitor without displaying start or stop', () => {
    const external = { ...status, enabled: true, running: true, externally_managed: true };
    const html = renderToStaticMarkup(<LocalMonitorOverview status={external} onChange={vi.fn()} />);
    expect(html).toContain('Monitoring is running');
    expect(html).toContain('started outside this panel');
    expect(html).toContain('Recent alerts');
    expect(html).not.toContain('<button');
  });
});

describe('resource alert review and notification opt-in', () => {
  it('offers an unchecked notification opt-in before starting and shows the current setting after', () => {
    const initial = renderToStaticMarkup(<LocalMonitorOverview status={status} onChange={vi.fn()} />);
    expect(initial).toContain('type="checkbox"');
    expect(initial).toContain('Show desktop notifications for resource alerts and recovery');
    expect(initial).not.toContain('checked=""');

    const running = { ...status, enabled: true, running: true, notifications_enabled: true };
    const active = renderToStaticMarkup(<LocalMonitorOverview status={running} onChange={vi.fn()} />);
    expect(active).toContain('Desktop notifications are on.');
    expect(active).toContain('Disable and restart monitoring');
    expect(active).not.toContain('type="checkbox"');
  });

  it('shows a review action for an active alert and leaves reviewed alerts in history', () => {
    const unreviewed = { ...alert, id: 42, status: 'active', resolved_at: null, acknowledged_at: null };
    const multiple = { ...status, incidents: [unreviewed, alert] };
    const html = renderToStaticMarkup(<LocalMonitorOverview status={multiple} onChange={vi.fn()} />);
    expect(html.match(/Mark as reviewed/g)).toHaveLength(1);
    expect(html).toContain('Warning · Active');
    expect(html).toContain('Warning · Recovered · Reviewed');

    const pending = renderToStaticMarkup(
      <LocalMonitorOverview status={multiple} change={{ incidentId: 42 }} onChange={vi.fn()} />,
    );
    expect(pending).toContain('Marking as reviewed…');
    expect(pending).toContain('disabled=""');
  });
});

describe('explicit incident analysis', () => {
  const analysis = { localModelAvailable: false, pending: null, feedback: new Map(), start: vi.fn() };

  it('keeps measurement analysis available without an inference model', () => {
    const html = renderToStaticMarkup(<LocalMonitorOverview status={status} onChange={vi.fn()} analysis={analysis} />);
    expect(html).toContain('Analyze incident');
    expect(html).not.toContain('Analyze with local model');
  });

  it('shows model progress without disabling the stop and review controls', () => {
    const active = {
      ...status, enabled: true, running: true, incidents: [{ ...alert, acknowledged_at: null }],
    };
    const html = renderToStaticMarkup(<LocalMonitorOverview status={active} onChange={vi.fn()} analysis={{
      ...analysis, localModelAvailable: true, pending: { incidentId: alert.id, useLocalModel: true },
    }} />);
    expect(html).toContain('Analyzing with a local model…');
    expect(html).toContain('Analyze with local model');
    expect(html).toMatch(/<button(?![^>]* disabled="")[^>]*>Disable monitoring<\/button>/);
    expect(html).toMatch(/<button(?![^>]* disabled="")[^>]*>Mark as reviewed<\/button>/);
  });

  it('shows one escaped assessment when the user asked for a model and measurements were used', () => {
    const assessment = alert.assessments[0]!;
    const html = renderToStaticMarkup(<LocalMonitorOverview status={status} onChange={vi.fn()} analysis={{
      ...analysis,
      feedback: new Map([[alert.id, {
        result: { incident_id: alert.id, assessment, mode: 'measurements' },
        requestedModel: true,
      }]]),
    }} />);
    expect(html).toContain('Measurement assessment saved.');
    expect(html.match(/&lt;script&gt;/g)).toHaveLength(1);
    expect(html).not.toContain('<script>');
  });
});
