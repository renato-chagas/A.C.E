import { authHeaders } from './api';
import { resolveLoopbackApiOrigin } from './loopback-api';

export const SCREEN_QUESTION_LIMIT = 2000;

export type ScreenInspection =
  | { mode: 'text' }
  | { mode: 'vision' | 'text-question'; question: string }
  | { mode: 'cloud-vision'; question: string; confirm_upload: true };

export type ScreenAvailability = {
  available: boolean;
  text_available: boolean;
  text_question_available: boolean;
  cloud_vision_available: boolean;
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
      if (inspection.mode === 'cloud-vision') {
        if (response.status === 424) throw new Error('Cloudflare denied access. Check the account and token permissions.');
        if (response.status === 429) throw new Error('Cloudflare reached a usage or rate limit.');
      }
      if (response.status === 503) {
        const messages = {
          text: 'Local text recognition or screen capture is unavailable.',
          vision: 'A compatible local vision model and screen capture are required.',
          'text-question': 'Local text recognition, screen capture, and a compatible local model are required.',
          'cloud-vision': 'Cloudflare Gemma 4 or screen capture is unavailable.',
        };
        throw new Error(messages[inspection.mode]);
      }
      if (response.status === 502) {
        const messages = {
          text: 'ACE could not read text from this capture.',
          vision: 'The local vision model could not analyze this capture.',
          'text-question': 'ACE could not read or analyze the text in this capture.',
          'cloud-vision': 'Cloudflare could not analyze this capture.',
        };
        throw new Error(messages[inspection.mode]);
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
      return {
        available: data.available,
        text_available: data.text_available,
        text_question_available: data.text_question_available === true,
        cloud_vision_available: data.cloud_vision_available === true,
      };
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