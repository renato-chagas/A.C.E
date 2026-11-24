import { useEffect, useState } from 'react';
import { Outlet, useNavigate } from 'react-router';
import { ApprovalBell } from './ApprovalBell';
import { Sidebar } from './Sidebar/Sidebar';
import { SystemPulse } from './SystemPulse';
import { useAppStore } from '../lib/store';
import { checkHealth } from '../lib/api';

export function Layout({
  localServicesDetail,
  onConfigureModel,
  backendReachable,
  inferenceUnavailable,
  onRetryModel,
  retryingModel,
  retryError,
}: {
  localServicesDetail?: string;
  onConfigureModel?: () => void;
  backendReachable?: boolean | null;
  inferenceUnavailable?: boolean;
  onRetryModel?: () => void;
  retryingModel?: boolean;
  retryError?: string;
}) {
  const sidebarOpen = useAppStore((s) => s.sidebarOpen);
  const [genericReachability, setGenericReachability] = useState<boolean | null>(null);
  const apiReachable = backendReachable === undefined ? genericReachability : backendReachable;

  useEffect(() => {
    if (backendReachable !== undefined) return;
    const check = () => checkHealth().then(setGenericReachability);
    check();
    const interval = setInterval(check, 30000);
    const onFocus = () => check();
    window.addEventListener('focus', onFocus);
    return () => {
      clearInterval(interval);
      window.removeEventListener('focus', onFocus);
    };
  }, [backendReachable]);

  const navigate = useNavigate();

  return (
    <div className="flex flex-col h-full w-full overflow-hidden relative" style={{ paddingTop: '3px' }}>
      <div className="hud-backdrop" aria-hidden="true" />
      <SystemPulse apiReachable={apiReachable} />
      <ApprovalBell />

      {(inferenceUnavailable || onConfigureModel) && (
        <div
          className="flex flex-wrap items-center justify-between gap-2 border-b px-4 py-2 text-sm"
          style={{ borderColor: 'var(--color-border)', color: 'var(--color-text-secondary)' }}
        >
          <span role="status">{inferenceUnavailable
            ? 'The selected model is temporarily unavailable. Local services are still ready.'
            : localServicesDetail || 'Local services are ready. Chat needs a model.'}</span>
          <div className="flex gap-3" style={{ color: 'var(--color-accent)' }}>
            {onRetryModel && (
              <button type="button" onClick={onRetryModel} disabled={retryingModel} className="underline">
                {retryingModel ? 'Restarting…' : 'Retry saved model'}
              </button>
            )}
            {onConfigureModel && (
              <button type="button" onClick={onConfigureModel} disabled={retryingModel} className="underline">
                Configure model
              </button>
            )}
          </div>
          {retryError && <span role="alert" className="w-full">{retryError}</span>}
        </div>
      )}

      {/* Health check banner */}
      {apiReachable === false && (
        <div
          className="flex items-center gap-3 px-4 py-2 text-sm shrink-0"
          style={{
            background: 'color-mix(in srgb, var(--color-error) 8%, transparent)',
            borderBottom: '1px solid color-mix(in srgb, var(--color-error) 15%, transparent)',
            color: 'var(--color-text)',
          }}
        >
          <span
            className="w-1.5 h-1.5 rounded-full shrink-0"
            style={{ background: 'var(--color-error)' }}
          />
          <span>Cannot reach the A.C.E. backend</span>
          <button
            onClick={() => navigate('/settings')}
            className="text-sm underline cursor-pointer ml-auto shrink-0"
            style={{ color: 'var(--color-accent)' }}
          >
            Change URL
          </button>
        </div>
      )}

      <div className="flex flex-1 min-h-0 relative z-10">
        <Sidebar onConfigureModel={onConfigureModel} />
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-20 bg-black/40 md:hidden"
            onClick={() => useAppStore.getState().setSidebarOpen(false)}
          />
        )}
        <main className="flex-1 flex flex-col min-w-0 h-full relative overflow-hidden" style={{ background: 'transparent' }}>
          <div className="flex-1 flex flex-col min-w-0 min-h-0 relative z-[2]">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
