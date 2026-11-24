import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { GraphEpisodeTranscript } from '../components/GraphEpisodeDialog';
import { fetchMemoryEpisode } from '../lib/api';
import { graphRelationGeometry, graphViewPath } from '../lib/knowledge-graph';
import { EvidenceDates, graphEventDescription, GraphRelationEdge, KnowledgeGraphPage, type GraphEvent } from './KnowledgeGraphPage';

function event(overrides: Partial<GraphEvent>): GraphEvent {
  return {
    id: 1,
    event_type: 'entity_added',
    entity_id: 'ace',
    relation_id: 0,
    source: 'manual',
    created_at: 1_700_000_000,
    before: {},
    after: {},
    ...overrides,
  };
}

describe('Knowledge Graph relation filter', () => {
  it('shows a separate filter with all stored claim statuses', () => {
    const html = renderToStaticMarkup(<KnowledgeGraphPage />);
    expect(html).toContain('Filter by entity type');
    expect(html).toContain('Filter relations by status');
    expect(html).toContain('<option value="all" selected="">All relation statuses</option>');
    expect(html).toContain('<option value="hypothesis">Hypothesis</option>');
    expect(html).toContain('<option value="verified">Verified</option>');
    expect(html).toContain('<option value="contradicted">Contradicted</option>');
    expect(html).toContain('<option value="superseded">Superseded</option>');
  });
});

describe('Knowledge Graph change history labels', () => {
  it('describes entity rename events with the before and after names', () => {
    expect(graphEventDescription(event({
      event_type: 'entity_updated',
      before: { name: 'ACE' },
      after: { name: 'A.C.E.' },
    }), [])).toBe('Updated entity: ACE → A.C.E.');
  });

  it('uses stored entity names for relation history', () => {
    expect(graphEventDescription(event({
      event_type: 'relation_added',
      after: {
        source_id: 'ace',
        target_id: 'jarvis',
        relation_type: 'uses',
        polarity: 'affirmed',
      },
    }), [
      { id: 'ace', name: 'A.C.E.' },
      { id: 'jarvis', name: 'OpenJarvis' },
    ])).toBe('Added relation: A.C.E. —uses→ OpenJarvis');
  });

  it('shows connected relation count when an entity was deleted', () => {
    expect(graphEventDescription(event({
      event_type: 'entity_deleted',
      before: { name: 'Old project', removed_relation_count: 3 },
    }), [])).toBe('Deleted entity: Old project (removed 3 connected relations)');
  });

  it('shows counts without restoring purged conversation content', () => {
    expect(graphEventDescription(event({
      event_type: 'memory_data_purged',
      before: {
        evidence_removed: 2,
        entities_removed: 1,
        entities_updated: 0,
        relations_removed: 0,
        relations_updated: 1,
      },
    }), [])).toBe('Purged linked memory data (evidence removed: 2; entities removed: 1; relations removed: 0; entities updated: 0; relations updated: 1)');
  });
});

describe('Knowledge Graph conversation episodes', () => {
  const episode = {
    episode_id: 'a'.repeat(32),
    created_at: 1_700_000_000,
    source: 'chat',
    user_text: '<img src="/capture" />',
    assistant_text: '[A link](https://example.org)',
  };

  afterEach(() => vi.restoreAllMocks());

  it('fetches a validated private episode using an abortable request without caching or redirects', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json(episode));
    const { signal } = new AbortController();
    await expect(fetchMemoryEpisode(episode.episode_id.toUpperCase(), signal)).resolves.toEqual(episode);
    expect(fetch).toHaveBeenCalledWith(
      expect.stringMatching(new RegExp('/v1/memory/episodes/' + episode.episode_id + '$')),
      expect.objectContaining({ signal, cache: 'no-store', redirect: 'error' }),
    );
  });

  it('rejects unsafe episode IDs before calling the API', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch');
    await expect(fetchMemoryEpisode('../stats', new AbortController().signal))
      .rejects.toThrow('Invalid conversation episode ID.');
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each([
    [404, 'This episode is unavailable.'],
    [403, 'The server only allows access to conversation episodes from this computer.'],
    [401, 'Your ACE connection needs a valid API key'],
    [500, 'Could not load this conversation episode.'],
  ])('explains HTTP %s without showing the response body', async (status, message) => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('Internal details', { status }));
    await expect(fetchMemoryEpisode(episode.episode_id, new AbortController().signal))
      .rejects.toThrow(message);
  });

  it('rejects a response for a different episode', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(Response.json({ ...episode, episode_id: 'b'.repeat(32) }));
    await expect(fetchMemoryEpisode(episode.episode_id, new AbortController().signal))
      .rejects.toThrow('ACE returned an invalid conversation episode.');
  });

  it('renders stored HTML and Markdown as literal transcript text', () => {
    const html = renderToStaticMarkup(<GraphEpisodeTranscript episode={episode} />);
    expect(html).toContain('Source: chat');
    expect(html).toContain('2023-11-14T22:13:20.000Z');
    expect(html).toContain('&lt;img src=&quot;/capture&quot; /&gt;');
    expect(html).toContain('[A link](https://example.org)');
    expect(html).not.toContain('<img');
    expect(html).not.toContain('href=');
  });
});

describe('Knowledge Graph relation selection', () => {
  const [geometry] = graphRelationGeometry([{
    id: 'denial',
    source: 'ace',
    target: 'jarvis',
    type: 'uses',
    polarity: 'negated' as const,
    status: 'reported',
    confidence: 0.9,
    weight: 1,
    properties: {},
    evidence: [],
  }], new Map([['ace', { x: 25, y: 25 }], ['jarvis', { x: 225, y: 25 }]]), { x: 120, y: 120 });

  it('makes a directional claim a focusable control and exposes its selection', () => {
    const html = renderToStaticMarkup(
      <svg>
        <GraphRelationEdge
          geometry={geometry}
          sourceName="ACE"
          targetName="Jarvis"
          selected
          possibleConflict
          reviewDue={false}
          onSelect={() => {}}
        />
      </svg>,
    );
    expect(html).toContain('role="button"');
    expect(html).toContain('tabindex="0"');
    expect(html).toContain('aria-pressed="true"');
    expect(html).toContain('aria-label="Select relation ACE → Jarvis, NOT uses (Reported, Possible conflict)"');
    expect(html).toContain('pointer-events="stroke"');
    expect(html).toContain('<polygon');
  });

  it('activates a claim on pointer click, Enter, or Space without taking arrow keys', () => {
    const onSelect = vi.fn();
    const { props } = GraphRelationEdge({
      geometry,
      sourceName: 'ACE',
      targetName: 'Jarvis',
      selected: false,
      possibleConflict: false,
      reviewDue: false,
      onSelect,
    });
    const preventDefault = vi.fn();
    props.onKeyDown({ key: 'ArrowDown', preventDefault });
    expect(preventDefault).not.toHaveBeenCalled();
    expect(onSelect).not.toHaveBeenCalled();

    props.onClick();
    props.onKeyDown({ key: 'Enter', preventDefault });
    props.onKeyDown({ key: ' ', preventDefault });
    expect(onSelect).toHaveBeenCalledTimes(3);
    expect(preventDefault).toHaveBeenCalledTimes(2);
  });
});


describe('Knowledge Graph neighborhood navigation', () => {
  it('requests the exact focused entity and treats the cursor separately from a search', () => {
    const path = graphViewPath('other search', 'project:ACE #1', 47);
    const params = new URL(path, 'http://localhost').searchParams;
    expect(params.get('focus_id')).toBe('project:ACE #1');
    expect(params.get('before_relation_id')).toBe('47');
    expect(params.get('limit')).toBe('32');
    expect(params.has('query')).toBe(false);
  });

  it('does not carry a neighborhood cursor into the global view', () => {
    const path = graphViewPath('storage', '', 47);
    const params = new URL(path, 'http://localhost').searchParams;
    expect(params.get('query')).toBe('storage');
    expect(params.has('before_relation_id')).toBe(false);
    expect(params.has('focus_id')).toBe(false);
  });
});

describe('web graph evidence date', () => {
  it('displays when the archived page was captured', () => {
    const html = renderToStaticMarkup(<EvidenceDates evidence={{ source: 'web', observed_at: 1_700_000_000 }} />);
    expect(html).toContain('Page captured');
    expect(html).toContain('<time dateTime="2023-11-14T22:13:20.000Z">');
  });

  it.each([
    { source: 'web', observed_at: Infinity },
    { source: 'web', observed_at: 0 },
    { source: 'web' },
    { source: 'manual', observed_at: 1_700_000_000 },
  ])('omits a capture date without a valid web capture timestamp', (evidence) => {
    const html = renderToStaticMarkup(<EvidenceDates evidence={evidence} />);
    expect(html).not.toContain('Page captured');
  });
});
