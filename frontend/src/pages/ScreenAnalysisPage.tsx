import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Button } from '../components/ui/button';
import {
  createScreenInspectionClient,
  SCREEN_QUESTION_LIMIT,
  type ScreenAvailability,
  type ScreenInspection,
} from '../lib/screen-inspection';

const COUNTDOWN_SECONDS = 3;
const muted = { color: 'var(--color-text-secondary)' };
const border = { borderColor: 'var(--color-border)' };

type PendingCapture = { inspection: ScreenInspection; captureAt: number };
type InspectionResult = { mode: ScreenInspection['mode']; answer: string };

export function ScreenAnalysisPage({
  backendUnavailable = false,
  visionUnavailable = false,
}: {
  backendUnavailable?: boolean;
  visionUnavailable?: boolean;
}) {
  const [client] = useState(createScreenInspectionClient);
  const [availability, setAvailability] = useState<ScreenAvailability | null>(null);
  const [checking, setChecking] = useState(false);
  const [checkError, setCheckError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [question, setQuestion] = useState('Describe what is visible on the screen.');
  const [pending, setPending] = useState<PendingCapture | null>(null);
  const [secondsLeft, setSecondsLeft] = useState(COUNTDOWN_SECONDS);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<InspectionResult | null>(null);
  const pendingCapture = useRef<PendingCapture | null>(null);
  const request = useRef<{ controller: AbortController; mode: ScreenInspection['mode'] } | null>(null);

  const textReady = !backendUnavailable && availability?.text_available === true;
  const visionReady = !backendUnavailable && !visionUnavailable && availability?.available === true;
  const pendingAllowed = pending?.inspection.mode === 'text' ? textReady : visionReady;

  useEffect(() => {
    if (!client || backendUnavailable) {
      setAvailability(null);
      return;
    }
    const controller = new AbortController();
    setChecking(true);
    setCheckError('');
    void client.status(controller.signal).then(
      (status) => {
        if (controller.signal.aborted) return;
        setAvailability(status);
        setChecking(false);
      },
      (cause: unknown) => {
        if (controller.signal.aborted) return;
        setAvailability(null);
        setChecking(false);
        setCheckError(cause instanceof Error ? cause.message : 'Could not check screen analysis availability.');
      },
    );
    return () => controller.abort();
  }, [client, backendUnavailable, visionUnavailable, refresh]);

  useLayoutEffect(() => {
    if (!pending || !client) return;
    if (!pendingAllowed) {
      pendingCapture.current = null;
      setPending(null);
      return;
    }

    const { inspection, captureAt } = pending;
    const clock = window.setInterval(() => {
      setSecondsLeft(Math.max(1, Math.ceil((captureAt - Date.now()) / 1000)));
    }, 250);
    const timer = window.setTimeout(() => {
      if (pendingCapture.current !== pending) return;
      pendingCapture.current = null;
      const controller = new AbortController();
      request.current = { controller, mode: inspection.mode };
      setAnalyzing(true);
      setPending(null);
      setResult(null);
      void client.inspect(inspection, controller.signal).then(
        (answer) => {
          if (!controller.signal.aborted) setResult({ mode: inspection.mode, answer });
        },
        (cause: unknown) => {
          if (!controller.signal.aborted) {
            setError(cause instanceof Error ? cause.message : 'Could not inspect the screen.');
          }
        },
      ).finally(() => {
        if (request.current?.controller !== controller) return;
        request.current = null;
        setAnalyzing(false);
      });
    }, Math.max(0, captureAt - Date.now()));
    return () => {
      window.clearInterval(clock);
      window.clearTimeout(timer);
    };
  }, [client, pending, pendingAllowed]);

  useEffect(() => {
    const active = request.current;
    if (active && !(active.mode === 'text' ? textReady : visionReady)) {
      active.controller.abort();
      request.current = null;
      setAnalyzing(false);
    }
  }, [textReady, visionReady]);

  useEffect(() => () => {
    request.current?.controller.abort();
    request.current = null;
  }, []);

  function scheduleCapture(inspection: ScreenInspection) {
    if (pendingCapture.current || request.current || !(inspection.mode === 'text' ? textReady : visionReady)) return;
    setError('');
    setSecondsLeft(COUNTDOWN_SECONDS);
    const capture = { inspection, captureAt: Date.now() + COUNTDOWN_SECONDS * 1000 };
    pendingCapture.current = capture;
    setPending(capture);
  }

  function cancelCapture() {
    pendingCapture.current = null;
    setPending(null);
  }

  const busy = pending !== null || analyzing;
  const couldCheck = client !== null && !backendUnavailable;

  return (
    <section className="h-full min-h-0 overflow-y-auto p-5 md:p-8" style={{ color: 'var(--color-text)' }}>
      <div className="mx-auto w-full max-w-5xl space-y-5">
        <header className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-2xl font-semibold">Screen analysis</h1>
          {couldCheck && (
            <Button variant="outline" disabled={checking || busy} onClick={() => setRefresh((value) => value + 1)}>
              Check availability
            </Button>
          )}
        </header>
        {!client ? (
          <p role="alert" className="text-sm" style={muted}>Open ACE on this computer and use a direct local API address in Settings.</p>
        ) : backendUnavailable ? (
          <p role="alert" className="text-sm" style={muted}>The local ACE server is unavailable.</p>
        ) : (
          <>
            <p className="text-sm" style={muted}>ACE captures one image of your primary screen three seconds after you choose an action.</p>
            {checking && !availability && <p role="status" className="text-sm" style={muted}>Checking screen analysis availability…</p>}
            {checkError && <p role="alert" className="text-sm" style={{ color: 'var(--color-error)' }}>{checkError}</p>}
            <div className="grid gap-4 md:grid-cols-2">
              <section aria-labelledby="screen-text-heading" className="space-y-3 rounded-xl border p-4" style={border}>
                <h2 id="screen-text-heading" className="font-medium">Text recognition</h2>
                <p className="text-sm" style={muted}>Extract visible text on this computer.</p>
                {availability && !textReady && <p className="text-sm" style={muted}>Local text recognition is unavailable on this computer.</p>}
                <Button variant="outline" disabled={!textReady || busy} onClick={() => scheduleCapture({ mode: 'text' })}>
                  Read screen text
                </Button>
              </section>
              <form
                aria-labelledby="screen-vision-heading"
                className="space-y-3 rounded-xl border p-4"
                style={border}
                onSubmit={(event) => {
                  event.preventDefault();
                  const trimmed = question.trim();
                  if (trimmed) scheduleCapture({ mode: 'vision', question: trimmed });
                }}
              >
                <h2 id="screen-vision-heading" className="font-medium">Vision model</h2>
                <p className="text-sm" style={muted}>Ask your local vision model about the capture.</p>
                {(visionUnavailable || (availability && !visionReady)) && (
                  <p className="text-sm" style={muted}>Select and start a compatible local vision model in Settings.</p>
                )}
                <label htmlFor="screen-analysis-question" className="block text-sm font-medium">Question</label>
                <textarea
                  id="screen-analysis-question"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  rows={3}
                  maxLength={SCREEN_QUESTION_LIMIT}
                  disabled={busy}
                  className="min-h-20 w-full resize-y rounded-lg border px-3 py-2 text-sm disabled:opacity-50"
                  style={{ ...border, background: 'var(--color-bg-secondary)' }}
                />
                <Button type="submit" disabled={!visionReady || busy || !question.trim()}>
                  Capture and analyze
                </Button>
              </form>
            </div>
            {pending && (
              <div className="flex flex-wrap items-center gap-3">
                <p role="status" className="text-sm" style={muted}>
                  Capturing in {secondsLeft} {secondsLeft === 1 ? 'second' : 'seconds'}. Switch to the window you want to inspect.
                </p>
                <Button variant="outline" onClick={cancelCapture}>Cancel capture</Button>
              </div>
            )}
            {analyzing && <p role="status" className="text-sm" style={muted}>Processing the screen capture…</p>}
            {error && <p role="alert" className="text-sm" style={{ color: 'var(--color-error)' }}>{error}</p>}
          </>
        )}
        {result && (
          <section aria-labelledby="screen-result-heading" className="space-y-3 rounded-xl border p-4" style={border}>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 id="screen-result-heading" className="font-medium">
                {result.mode === 'text' ? 'Screen text' : 'Analysis result'}
              </h2>
              <Button variant="outline" onClick={() => setResult(null)}>Clear result</Button>
            </div>
            <p className="whitespace-pre-wrap break-words text-sm">{result.answer}</p>
          </section>
        )}
      </div>
    </section>
  );
}