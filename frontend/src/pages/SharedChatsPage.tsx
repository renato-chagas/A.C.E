import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import {
  createWorkingChatsClient,
  WorkingChatError,
  WORKING_CHAT_MESSAGE_LIMIT,
  type WorkingChat,
  type WorkingChatsClient,
  type WorkingTurn,
} from '../lib/working-chats';

const muted = { color: 'var(--color-text-secondary)' };
const border = { borderColor: 'var(--color-border)' };
const errorText = { color: 'var(--color-error)' };
const button = 'rounded-lg border px-3 py-2 text-sm disabled:opacity-50';
const field = 'w-full rounded-lg border bg-transparent px-3 py-2 text-sm';
const unsavedReasons = {
  changed: 'A concurrent edit prevented saving this reply. Reload the chat before continuing.',
  scanner: 'The local scanner blocked saving this exchange.',
  incomplete: 'The model returned an incomplete reply. ACE kept the saved transcript.',
  storage: 'ACE could not save this exchange. Reload the chat before continuing.',
};

type VisibleReply = {
  user: string;
  answer: string;
  saved: boolean;
};

export function SharedChatTranscript({
  turns,
  latest,
}: {
  turns: WorkingChat['turns'];
  latest: VisibleReply | null;
}) {
  const exchanges = turns.map(([user, answer]) => ({ user, answer, saved: true }));
  if (latest?.saved && exchanges.length) {
    exchanges[exchanges.length - 1] = latest;
  } else if (latest?.answer) {
    exchanges.push(latest);
  }

  if (!exchanges.length) return <p className="text-sm" style={muted}>This chat has no saved messages.</p>;

  return (
    <ol aria-label="Shared conversation" className="space-y-3">
      {exchanges.map(({ user, answer, saved }, index) => (
        <li key={index} className="space-y-3 rounded-xl border p-4" style={border}>
          <div>
            <p className="text-xs font-semibold" style={muted}>You</p>
            <p className="mt-1 whitespace-pre-wrap break-words">{user}</p>
          </div>
          <div>
            <p className="text-xs font-semibold" style={muted}>{saved ? 'ACE' : 'ACE · Unsaved reply'}</p>
            <p className="mt-1 whitespace-pre-wrap break-words">{answer}</p>
          </div>
        </li>
      ))}
    </ol>
  );
}

export function SharedChatsPage({ localServicesOnly = false }: { localServicesOnly?: boolean }) {
  const [client] = useState(createWorkingChatsClient);
  const [names, setNames] = useState<string[]>([]);
  const [chat, setChat] = useState<WorkingChat | null>(null);
  const [nameInput, setNameInput] = useState('');
  const [draft, setDraft] = useState('');
  const [latest, setLatest] = useState<VisibleReply | null>(null);
  const [pending, setPending] = useState('');
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');
  const [needsReload, setNeedsReload] = useState(false);
  const [confirmForget, setConfirmForget] = useState(false);
  const [serverWithoutInference, setServerWithoutInference] = useState(false);
  const inferenceDisabled = localServicesOnly || serverWithoutInference;
  const request = useRef<AbortController | null>(null);

  const run = useCallback(async <T,>(
    label: string,
    work: (api: WorkingChatsClient, signal: AbortSignal) => Promise<T>,
    accept: (result: T) => void,
    onError?: (cause: unknown) => void,
  ) => {
    if (!client || request.current) return;
    const controller = new AbortController();
    request.current = controller;
    setPending(label);
    setError('');
    setNotice('');
    try {
      const result = await work(client, controller.signal);
      if (!controller.signal.aborted) accept(result);
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : 'The local request failed.');
        onError?.(cause);
      }
    } finally {
      if (request.current === controller) {
        request.current = null;
        setPending('');
      }
    }
  }, [client]);

  const refreshNames = useCallback(() => {
    void run('Loading shared chats…', (api, signal) => api.list(signal), setNames);
  }, [run]);

  useEffect(() => {
    refreshNames();
    return () => {
      request.current?.abort();
      request.current = null;
    };
  }, [refreshNames]);

  useEffect(() => {
    if (!client || localServicesOnly) return;
    const controller = new AbortController();
    void client.inferenceAvailable(controller.signal).then((available) => {
      if (!controller.signal.aborted) setServerWithoutInference(!available);
    }).catch(() => {});
    return () => controller.abort();
  }, [client, localServicesOnly]);

  const showChat = (next: WorkingChat) => {
    if (chat?.name !== next.name) setDraft('');
    setChat(next);
    setLatest(null);
    setNeedsReload(false);
    setConfirmForget(false);
  };

  const readChat = (name: string) => {
    void run('Opening shared chat…', (api, signal) => api.read(name, signal), showChat, (cause) => {
      if (cause instanceof WorkingChatError && cause.status === 404) {
        setNames((current) => current.filter((entry) => entry !== name));
        if (chat?.name === name) {
          setChat(null);
          setDraft('');
          setLatest(null);
          setNeedsReload(false);
          setConfirmForget(false);
        }
      }
    });
  };

  const requestedName = nameInput.trim();
  const nameTooLong = [...requestedName].length > 80;
  const canOpen = !!requestedName && !nameTooLong;

  const openChat = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canOpen) return;
    void run('Opening shared chat…', (api, signal) => api.open(requestedName, signal), (next) => {
      showChat(next);
      setNames((current) => current.includes(next.name) ? current : [next.name, ...current]);
      setNameInput('');
    });
  };

  const markConflict = (cause: unknown) => {
    if (cause instanceof WorkingChatError && (cause.status === 409 || cause.status === 404)) {
      setNeedsReload(true);
      setConfirmForget(false);
    }
  };

  const forgetChat = () => {
    if (!chat || !confirmForget || needsReload) return;
    void run('Deleting shared chat…', (api, signal) => api.forget(chat, signal), () => {
      setNames((current) => current.filter((name) => name !== chat.name));
      setChat(null);
      setDraft('');
      setLatest(null);
      setConfirmForget(false);
      setNotice('Shared chat deleted.');
    }, markConflict);
  };

  const message = draft.trim();
  const exceedsMessageLimit = [...message].length > WORKING_CHAT_MESSAGE_LIMIT;

  const send = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!chat || inferenceDisabled || needsReload || confirmForget || !message || exceedsMessageLimit) return;
    void run('Waiting for the local model…', (api, signal) => api.turn(chat, message, signal), (turn: WorkingTurn) => {
      if (turn.chat) setChat(turn.chat);
      setDraft('');
      setLatest({ user: message, answer: turn.answer, saved: turn.saved });
      if (turn.saved) {
        setNames((current) => [chat.name, ...current.filter((name) => name !== chat.name)]);
      } else {
        if (!turn.chat || turn.reason === 'changed') setNeedsReload(true);
        setNotice(turn.reason ? unsavedReasons[turn.reason] : 'This exchange is only visible in the current view.');
      }
    }, markConflict);
  };

  return (
    <section className="h-full min-h-0 overflow-y-auto p-5 md:p-8" style={{ color: 'var(--color-text)' }}>
      <div className="mx-auto w-full max-w-5xl space-y-5">
        <header className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold">Shared chats</h1>
            <p className="mt-1 text-sm" style={muted}>Resume named text conversations from the ACE terminal on this computer.</p>
          </div>
          {client && <button type="button" disabled={!!pending} onClick={refreshNames} className={button} style={border}>Refresh list</button>}
        </header>

        {!client ? (
          <p role="alert" className="text-sm" style={errorText}>
            Open ACE on this computer and use a direct local API address in Settings.
          </p>
        ) : (
          <>
            {pending && <p role="status" className="text-sm" style={muted}>{pending}</p>}
            {notice && <p role="status" className="text-sm" style={muted}>{notice}</p>}
            {error && <p role="alert" className="text-sm" style={errorText}>{error}</p>}

            <div className="grid gap-4 md:grid-cols-[minmax(12rem,17rem)_minmax(0,1fr)]">
              <div className="space-y-3 rounded-xl border p-4" style={border}>
                <form onSubmit={openChat} className="space-y-2">
                  <label htmlFor="shared-chat-name" className="block text-sm font-medium">Chat name</label>
                  <input
                    id="shared-chat-name"
                    autoComplete="off"
                    required
                    value={nameInput}
                    onChange={(event) => setNameInput(event.target.value)}
                    maxLength={160}
                    disabled={!!pending}
                    className={field}
                    style={border}
                  />
                  {nameTooLong && <p className="text-xs" style={muted}>Use up to 80 characters.</p>}
                  <button type="submit" disabled={!canOpen || !!pending} className={button} style={border}>Open or create</button>
                </form>
                {names.length === 0 && !pending && !error && <p className="text-sm" style={muted}>No shared chats yet.</p>}
                <ul aria-label="Saved chat names" className="max-h-72 space-y-1 overflow-y-auto">
                  {names.map((name) => (
                    <li key={name}>
                      <button
                        type="button"
                        aria-current={chat?.name === name ? 'true' : undefined}
                        disabled={!!pending}
                        onClick={() => readChat(name)}
                        className="w-full break-words rounded-lg px-2 py-1.5 text-left text-sm disabled:opacity-50"
                        style={{ background: chat?.name === name ? 'var(--color-accent-subtle)' : 'transparent' }}
                      >
                        {name}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>

              {chat ? (
                <div className="min-w-0 space-y-4">
                  <header className="flex flex-wrap items-start justify-between gap-3">
                    <h2 className="min-w-0 break-words font-semibold">{chat.name}</h2>
                    <div className="flex flex-wrap gap-2">
                      <button type="button" onClick={() => readChat(chat.name)} disabled={!!pending} className={button} style={border}>Reload chat</button>
                      <button type="button" onClick={() => setConfirmForget(true)} disabled={!!pending || needsReload} className={button} style={{ ...border, ...errorText }}>Delete</button>
                    </div>
                  </header>
                  {confirmForget && (
                    <div role="group" aria-label="Delete shared chat" className="space-y-2 rounded-lg border p-3" style={border}>
                      <p className="text-sm">Delete this saved transcript from the terminal and desktop?</p>
                      <div className="flex gap-2">
                        <button type="button" onClick={forgetChat} disabled={!!pending} className={button} style={{ ...border, ...errorText }}>Delete transcript</button>
                        <button type="button" onClick={() => setConfirmForget(false)} disabled={!!pending} className={button} style={border}>Cancel</button>
                      </div>
                    </div>
                  )}
                  <SharedChatTranscript turns={chat.turns} latest={latest} />
                  {inferenceDisabled ? (
                    <p className="text-sm" style={muted}>Configure a local model to send messages.</p>
                  ) : (
                    <form onSubmit={send} className="space-y-2">
                      <label htmlFor="shared-chat-message" className="block text-sm font-medium">Message</label>
                      <textarea
                        id="shared-chat-message"
                        value={draft}
                        onChange={(event) => setDraft(event.target.value)}
                        autoComplete="off"
                        maxLength={WORKING_CHAT_MESSAGE_LIMIT * 2}
                        disabled={!!pending || confirmForget}
                        rows={3}
                        className={field}
                        style={border}
                      />
                      {exceedsMessageLimit && <p className="text-xs" style={muted}>Use up to 8,000 characters.</p>}
                      <div className="flex flex-wrap items-center gap-3">
                        <button
                          type="submit"
                          className={button}
                          style={{ ...border, color: 'var(--color-accent)' }}
                          disabled={!!pending || needsReload || confirmForget || !message || exceedsMessageLimit}
                        >
                          Send
                        </button>
                        <p className="text-xs" style={muted}>Uses the server’s active local model for text replies.</p>
                      </div>
                    </form>
                  )}
                </div>
              ) : (
                <p className="pt-2 text-sm" style={muted}>Choose a saved chat or enter a name to start one.</p>
              )}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
