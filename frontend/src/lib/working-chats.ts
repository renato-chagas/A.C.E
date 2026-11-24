import { authHeaders } from './api';
import { resolveLoopbackApiOrigin } from './loopback-api';

export const WORKING_CHAT_MESSAGE_LIMIT = 8000;

export type WorkingChatRef = {
  name: string;
  session_id: string;
  revision: number;
};

export type WorkingChat = WorkingChatRef & {
  turns: [user: string, assistant: string][];
};

export type WorkingTurn = {
  chat: WorkingChat | null;
  answer: string;
  saved: boolean;
  reason?: 'changed' | 'scanner' | 'incomplete' | 'storage';
};

type WorkingChatOperation = 'info' | 'list' | 'read' | 'open' | 'forget' | 'turn';

export class WorkingChatError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message);
    this.name = 'WorkingChatError';
  }
}

function snapshot(value: unknown, name: string): value is WorkingChat {
  if (!value || typeof value !== 'object') return false;
  const chat = value as Partial<WorkingChat>;
  return chat.name === name &&
    typeof chat.session_id === 'string' && /^[a-f0-9]{32}$/.test(chat.session_id) &&
    typeof chat.revision === 'number' && Number.isSafeInteger(chat.revision) && chat.revision >= 0 &&
    Array.isArray(chat.turns) && chat.turns.every((pair) =>
      Array.isArray(pair) && pair.length === 2 &&
      typeof pair[0] === 'string' && typeof pair[1] === 'string',
    );
}

function invalidResponse(): Error {
  return new Error('ACE returned an invalid shared chat response.');
}

function requestError(status: number, operation: WorkingChatOperation): WorkingChatError {
  const messages: Record<number, string> = {
    400: 'Check the chat name or message and try again.',
    401: 'The local ACE connection needs a valid API key.',
    403: 'Shared chats require direct access from this computer.',
    404: 'This shared chat no longer exists. Refresh the list.',
    409: 'This chat changed. Reload it before sending a message or deleting it.',
    413: 'The message or chat name exceeds the server limit.',
    422: 'Check the chat name or message and try again.',
    502: 'The local model could not complete a text reply.',
    503: operation === 'turn'
      ? 'The active local model or shared-chat service is unavailable. Check your model and try again.'
      : 'The local shared-chat store is unavailable.',
  };
  return new WorkingChatError(
    messages[status] ?? 'The local ACE server could not complete this request.',
    status,
  );
}

export function createWorkingChatsClient() {
  const origin = resolveLoopbackApiOrigin();
  if (!origin) return null;
  const endpoint = origin + '/v1/working-chats';

  async function request(
    operation: WorkingChatOperation,
    signal: AbortSignal,
    payload?: Record<string, string | number>,
  ): Promise<unknown> {
    const url = operation === 'info'
      ? origin + '/v1/info'
      : operation === 'list' ? endpoint : endpoint + '/' + operation;
    let response: Response;
    try {
      response = await fetch(url, {
        method: payload ? 'POST' : 'GET',
        headers: authHeaders(payload ? { 'Content-Type': 'application/json' } : {}),
        body: payload ? JSON.stringify(payload) : undefined,
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
    if (!response.ok) throw requestError(response.status, operation);
    return response.json().catch(() => { throw invalidResponse(); });
  }

  async function getChat(operation: 'read' | 'open', name: string, signal: AbortSignal): Promise<WorkingChat> {
    const body = await request(operation, signal, { name });
    if (!snapshot(body, name)) throw invalidResponse();
    return body;
  }

  return {
    async inferenceAvailable(signal: AbortSignal): Promise<boolean> {
      const info = await request('info', signal) as { inference_available?: boolean } | null;
      return info?.inference_available !== false;
    },
    async list(signal: AbortSignal): Promise<string[]> {
      const body = await request('list', signal) as { chats?: unknown } | null;
      if (!Array.isArray(body?.chats) || !body.chats.every((name) => typeof name === 'string')) {
        throw invalidResponse();
      }
      return body.chats;
    },
    read: (name: string, signal: AbortSignal) => getChat('read', name, signal),
    open: (name: string, signal: AbortSignal) => getChat('open', name, signal),
    async forget({ name, session_id, revision }: WorkingChatRef, signal: AbortSignal): Promise<void> {
      const body = await request('forget', signal, { name, session_id, revision }) as { deleted?: unknown } | null;
      if (body?.deleted !== true) throw invalidResponse();
    },
    async turn(
      { name, session_id, revision }: WorkingChatRef,
      message: string,
      signal: AbortSignal,
    ): Promise<WorkingTurn> {
      const body = await request('turn', signal, { name, session_id, revision, message }) as Partial<WorkingTurn> | null;
      if (!body || typeof body.answer !== 'string' || typeof body.saved !== 'boolean' ||
        !(body.chat === null || snapshot(body.chat, name)) ||
        (body.saved && (!body.chat || body.chat.session_id !== session_id || body.chat.revision !== revision + 1)) ||
        (body.reason != null && !['changed', 'scanner', 'incomplete', 'storage'].includes(body.reason))) {
        throw invalidResponse();
      }
      return { chat: body.chat, answer: body.answer, saved: body.saved, reason: body.reason ?? undefined };
    },
  };
}

export type WorkingChatsClient = NonNullable<ReturnType<typeof createWorkingChatsClient>>;
