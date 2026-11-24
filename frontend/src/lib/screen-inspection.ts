import { authHeaders } from './api';
import { resolveLoopbackApiOrigin } from './loopback-api';

export const SCREEN_QUESTION_LIMIT = 2000;

export type ScreenInspection =
  | { mode: 'text' }
  | { mode: 'vision'; question: string };

export type ScreenAvailability = {
  available: boolean;
  text_available: boolean;
};

export function createScreenInspectionClient() {
  const origin = resolveLoopbackApiOrigin();
  if (!origin) return null;
  const endpoint = origin + '/v1/screen/inspect';

  async function request(signal: AbortSignal, inspection?: ScreenInspection): Promise<Response> {
    let response: Response;
    try {
      response = await fetch(endpoint, {
        method: inspection ? 'POST' : 'GET',
        headers: inspection ? authHeaders({ 'Content-Type': 'application/json' }) : authHeaders(),
        body: inspection ? JSON.stringify(inspection) : undefined,
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

    if (response.status === 401) throw new Error('The local ACE connection needs a valid API key.');
    if (response.status === 403) throw new Error('Screen analysis requires direct access from this computer.');
    if (response.status === 409) throw new Error('Another screen inspection is already running.');
    if (inspection) {
      if (response.status === 422) throw new Error('Check the question and try again.');
      if (response.status === 503) {
        throw new Error(inspection.mode === 'text'
          ? 'Local text recognition or screen capture is unavailable.'
          : 'A compatible local vision model and screen capture are required.');
      }
      if (response.status === 502) {
        throw new Error(inspection.mode === 'text'
          ? 'ACE could not read text from this capture.'
          : 'The local vision model could not analyze this capture.');
      }
    }
    if (!response.ok) {
      throw new Error(inspection ? 'Could not inspect the screen.' : 'Could not check screen analysis availability.');
    }
    return response;
  }

  return {
    async status(signal: AbortSignal): Promise<ScreenAvailability> {
      const response = await request(signal);
      const data: Partial<ScreenAvailability> | null = await response.json().catch(() => null);
      if (typeof data?.available !== 'boolean' || typeof data.text_available !== 'boolean') {
        throw new Error('ACE returned invalid screen analysis availability.');
      }
      return { available: data.available, text_available: data.text_available };
    },
    async inspect(inspection: ScreenInspection, signal: AbortSignal): Promise<string> {
      const response = await request(signal, inspection);
      const data: { answer?: unknown } | null = await response.json().catch(() => null);
      if (typeof data?.answer !== 'string' || !data.answer.trim()) {
        throw new Error('ACE returned an invalid screen inspection result.');
      }
      return data.answer;
    },
  };
}