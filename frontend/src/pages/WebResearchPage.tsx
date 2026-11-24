import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '../components/ui/button';
import { safeExternalUrl } from '../lib/knowledge-graph';
import {
  canExtractWebSource,
  createWebLibraryClient,
  WEB_LIBRARY_QUERY_LIMIT,
  type SavedWebSource,
  type WebGraphExtraction,
} from '../lib/web-library';

const muted = { color: 'var(--color-text-secondary)' };
const border = { borderColor: 'var(--color-border)' };
const captureLabels = {
  page: 'Page text',
  snippet: 'Search snippet',
  unknown: 'Capture type unknown',
} as const;

type SearchKind = 'saved' | 'online';

type SourceView = {
  kind: SearchKind;
  query: string;
  sources: SavedWebSource[];
  offlineFallback: boolean;
};

type ExtractionFeedback = { result: WebGraphExtraction } | { error: string };

type ExtractionControls = {
  start?: (source: SavedWebSource) => void;
  pending: string | null;
  feedback: ReadonlyMap<string, ExtractionFeedback>;
};

function extractionSummary({ entities_stored, relations_stored }: WebGraphExtraction): string {
  const entities = `${entities_stored} ${entities_stored === 1 ? 'entity' : 'entities'}`;
  const relations = `${relations_stored} ${relations_stored === 1 ? 'relation' : 'relations'}`;
  return relations_stored === 0
    ? `No relations found. Stored ${entities} in the graph.`
    : `Stored ${entities} and ${relations} in the graph.`;
}

export function WebSourceList({ sources, extraction }: {
  sources: SavedWebSource[];
  extraction?: ExtractionControls;
}) {
  return (
    <ul className="space-y-3">
      {sources.map((source) => {
        const href = safeExternalUrl(source.url);
        const date = new Date(source.saved_at);
        const canExtract = canExtractWebSource(source);
        const extracting = extraction?.pending === source.revision;
        const feedback = extraction?.feedback.get(source.revision);
        return (
          <li key={source.url} className="min-w-0 space-y-2 rounded-xl border p-4" style={border}>
            <h3 className="break-words font-medium">{source.title || 'Saved web source'}</h3>
            <p className="break-all text-sm" style={muted}>
              {href ? (
                <a href={href} target="_blank" rel="noopener noreferrer" className="underline underline-offset-2">
                  {source.url}
                </a>
              ) : source.url}
            </p>
            <p className="text-xs" style={muted}>
              {captureLabels[source.capture]} · Saved{' '}
              {source.saved_at ? (
                <time dateTime={source.saved_at}>
                  {Number.isNaN(date.getTime()) ? source.saved_at : date.toLocaleString()}
                </time>
              ) : 'on an unknown date'}
            </p>
            <details className="text-sm">
              <summary className="cursor-pointer font-medium">Read saved text</summary>
              <p className="mt-2 max-h-72 overflow-y-auto whitespace-pre-wrap break-words leading-relaxed">
                {source.text}
              </p>
            </details>
            {canExtract && extraction?.start && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={extraction.pending !== null}
                onClick={() => extraction.start?.(source)}
              >
                {extracting ? 'Extracting…' : 'Extract to graph'}
              </Button>
            )}
            {feedback && ('error' in feedback ? (
              <p role="alert" className="text-sm" style={{ color: 'var(--color-error)' }}>{feedback.error}</p>
            ) : (
              <p role="status" className="text-sm" style={muted}>{extractionSummary(feedback.result)}</p>
            ))}
          </li>
        );
      })}
    </ul>
  );
}

export function WebResearchPage({ backendUnavailable = false, inferenceUnavailable = false }: {
  backendUnavailable?: boolean;
  inferenceUnavailable?: boolean;
}) {
  const [client] = useState(createWebLibraryClient);
  const [query, setQuery] = useState('');
  const [view, setView] = useState<SourceView | null>(null);
  const [pending, setPending] = useState<SearchKind | null>(null);
  const [error, setError] = useState('');
  const active = useRef<AbortController | null>(null);
  const activeExtraction = useRef<AbortController | null>(null);
  const [extracting, setExtracting] = useState<string | null>(null);
  const [extractionFeedback, setExtractionFeedback] = useState<ReadonlyMap<string, ExtractionFeedback>>(
    () => new Map(),
  );
  const reachable = client !== null && !backendUnavailable;
  const extractionAvailable = reachable && !inferenceUnavailable;

  useEffect(() => {
    if (!extractionAvailable) setExtracting(null);
    return () => {
      activeExtraction.current?.abort();
      activeExtraction.current = null;
    };
  }, [extractionAvailable]);

  async function extract(source: SavedWebSource) {
    if (!client || !extractionAvailable || activeExtraction.current) return;
    const controller = new AbortController();
    const { revision } = source;
    activeExtraction.current = controller;
    setExtracting(revision);
    setExtractionFeedback((previous) => {
      const next = new Map(previous);
      next.delete(revision);
      return next;
    });

    try {
      const result = await client.extractToGraph(source, controller.signal);
      if (!controller.signal.aborted) {
        setExtractionFeedback((previous) => new Map(previous).set(revision, { result }));
      }
    } catch (cause) {
      if (!controller.signal.aborted) {
        const error = cause instanceof Error ? cause.message : 'Graph extraction failed.';
        setExtractionFeedback((previous) => new Map(previous).set(revision, { error }));
      }
    } finally {
      if (activeExtraction.current === controller) {
        activeExtraction.current = null;
        setExtracting(null);
      }
    }
  }

  const search = useCallback(async (kind: SearchKind, term: string) => {
    if (!client || backendUnavailable || active.current) return;
    const controller = new AbortController();
    active.current = controller;
    setPending(kind);
    setError('');
    try {
      const result = kind === 'online'
        ? await client.searchAndSave(term, controller.signal)
        : { sources: await client.searchSaved(term, controller.signal), offline_fallback: false };
      if (!controller.signal.aborted) {
        setView({ kind, query: term, sources: result.sources, offlineFallback: result.offline_fallback });
      }
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(cause instanceof Error ? cause.message : 'The local web research request failed.');
      }
    } finally {
      if (active.current === controller) {
        active.current = null;
        setPending(null);
      }
    }
  }, [client, backendUnavailable]);

  useEffect(() => {
    void search('saved', '');
    return () => {
      active.current?.abort();
      active.current = null;
    };
  }, [search]);

  const term = query.trim();

  return (
    <section className="h-full min-h-0 overflow-y-auto p-5 md:p-8" style={{ color: 'var(--color-text)' }}>
      <div className="mx-auto w-full max-w-5xl space-y-5">
        <h1 className="text-2xl font-semibold">Web research</h1>
        {!client ? (
          <p role="alert" className="text-sm" style={muted}>
            Open ACE on this computer and use a direct local API address in Settings.
          </p>
        ) : backendUnavailable ? (
          <p role="alert" className="text-sm" style={muted}>The local ACE server is unavailable.</p>
        ) : null}

        <form
          className="space-y-3 rounded-xl border p-4"
          style={border}
          onSubmit={(event) => {
            event.preventDefault();
            void search('saved', term);
          }}
        >
          <label htmlFor="web-library-query" className="block text-sm font-medium">Topic or public page URL</label>
          <input
            id="web-library-query"
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            maxLength={WEB_LIBRARY_QUERY_LIMIT}
            disabled={!reachable}
            className="w-full rounded-lg border px-3 py-2 text-sm disabled:opacity-50"
            style={{ ...border, background: 'var(--color-bg-secondary)' }}
          />
          <p className="text-sm" style={muted}>
            Web research sends the topic to a search provider or fetches the public URL you enter. It saves results on this computer.
          </p>
          <div className="flex flex-wrap gap-3">
            <Button type="submit" variant="outline" disabled={!reachable || pending !== null}>Search saved sources</Button>
            <Button
              type="button"
              disabled={!reachable || pending !== null || !term}
              onClick={() => void search('online', term)}
            >
              Search and save on the web
            </Button>
          </div>
        </form>

        {reachable && pending && (
          <p role="status" className="text-sm" style={muted}>
            {pending === 'online' ? 'Searching the web and saving sources…' : 'Reading saved sources…'}
          </p>
        )}
        {reachable && error && <p role="alert" className="text-sm" style={{ color: 'var(--color-error)' }}>{error}</p>}
        {view && (
          <section aria-labelledby="web-library-results" className="space-y-3">
            <div>
              <h2 id="web-library-results" className="text-lg font-medium">
                {view.kind === 'online' && !view.offlineFallback ? 'Web research results' : 'Saved web sources'}
              </h2>
              {view.query && <p className="break-words text-sm" style={muted}>{view.query}</p>}
            </div>
            {view.offlineFallback && (
              <p role="status" className="text-sm" style={muted}>
                The web search could not refresh these sources. Showing local history.
              </p>
            )}
            {reachable && inferenceUnavailable && view.sources.some(canExtractWebSource) && (
              <p className="text-sm" style={muted}>Choose a local model to extract saved pages to the graph.</p>
            )}
            {view.sources.length > 0 ? (
              <WebSourceList
                sources={view.sources}
                extraction={{
                  start: extractionAvailable ? (source) => void extract(source) : undefined,
                  pending: extracting,
                  feedback: extractionFeedback,
                }}
              />
            ) : (
              <p className="text-sm" style={muted}>
                {view.query ? 'No sources match this search.' : 'There are no saved web sources yet.'}
              </p>
            )}
          </section>
        )}
      </div>
    </section>
  );
}
