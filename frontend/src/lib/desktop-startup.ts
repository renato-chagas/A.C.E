import type { SetupStatus } from './api';

export type DesktopReadyMode = 'inference' | 'local-services';

export type PackagedBackendHealth =
  | 'inference'
  | 'inference_unavailable'
  | 'local_services'
  | 'unreachable';

export async function checkPackagedBackendHealth(): Promise<PackagedBackendHealth> {
  try {
    const { invoke } = await import('@tauri-apps/api/core');
    return await invoke<PackagedBackendHealth>('check_packaged_backend_health');
  } catch {
    return 'unreachable';
  }
}

export function desktopReadyMode(status: SetupStatus | null): DesktopReadyMode | null {
  if (status?.error || !status?.server_ready) return null;
  if (status.phase === 'local_services_ready' && !status.model_ready) return 'local-services';
  if (status.phase === 'ready' && status.model_ready) return 'inference';
  return null;
}

let offlineDistribution = false;

export function isOfflineDistribution(): boolean {
  return offlineDistribution;
}

export async function initDesktopDistribution(): Promise<void> {
  if (typeof window === 'undefined' || !window.__TAURI_INTERNALS__) return;
  try {
    const { invoke } = await import('@tauri-apps/api/core');
    offlineDistribution = (await invoke<boolean>('is_offline_distribution')) === true;
  } catch {
    // Legacy desktop builds do not expose this command.
    offlineDistribution = false;
  }
}
