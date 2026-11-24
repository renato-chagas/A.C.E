import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type PointerEvent as ReactPointerEvent } from 'react';
import { Network, RefreshCw, Search, Trash2, Save, Plus, ExternalLink, Pencil, Check, X } from 'lucide-react';
import { apiFetch, isMemoryEpisodeId } from '../lib/api';
import { GraphEpisodeDialog } from '../components/GraphEpisodeDialog';
import { evidenceValidityLabel, graphPanFromPointer, graphRelationGeometry, graphViewPath, safeExternalUrl, type GraphPanStart } from '../lib/knowledge-graph';

type GraphEntity = {
  id: string;
  type: string;
  name: string;
  properties: Record<string, unknown>;
  source: string;
  source_url: string;
  confidence: number;
  status: string;
};

type GraphEvidence = { source: string; source_url: string; quote: string; confidence: number; status: string; observed_at?: number; valid_from?: string; valid_until?: string; episode_id?: string };
type GraphRelation = {
  id: string;
  source: string;
  target: string;
  type: string;
  confidence: number;
  weight: number;
  status: string;
  polarity: 'affirmed' | 'negated';
  properties: Record<string, unknown>;
  evidence: GraphEvidence[];
};
type GraphConflictClaim = { relation: GraphRelation; target: GraphEntity };
type GraphConflict = { subject: GraphEntity; predicate: string; kind: 'different_targets' | 'opposing_polarity' | 'mixed'; claims: GraphConflictClaim[] };
type GraphEvidenceReviewCandidate = { subject: GraphEntity; relation: GraphRelation; target: GraphEntity; last_evidence: GraphEvidence; last_evidence_at: number; age_days: number; review_after_days: number };
type GraphData = { next_relation_id?: number | null; entities: GraphEntity[]; relations: GraphRelation[]; possible_conflicts: GraphConflict[]; evidence_review_candidates: GraphEvidenceReviewCandidate[]; counts: { entities: number; relations: number; evidence: number } };
type GraphEventSnapshot = {
  entity_id?: string;
  entity_type?: string;
  name?: string;
  status?: string;
  confidence?: number;
  source_id?: string;
  target_id?: string;
  relation_type?: string;
  polarity?: string;
  details_purged?: boolean;
  removed_relation_count?: number;
  entities_removed?: number;
  entities_updated?: number;
  relations_removed?: number;
  relations_updated?: number;
  evidence_removed?: number;
  entities?: number;
  relations?: number;
  evidence_count?: number;
  evidence?: Array<{ source?: string; source_url?: string; quote?: string; status?: string }>;
};
export type GraphEvent = {
  id: number;
  event_type: string;
  entity_id: string;
  relation_id: number;
  source: string;
  created_at: number;
  before: GraphEventSnapshot;
  after: GraphEventSnapshot;
};
type GraphEventPage = { events: GraphEvent[]; next_before_id: number | null };

const emptyGraph: GraphData = { entities: [], relations: [], possible_conflicts: [], evidence_review_candidates: [], counts: { entities: 0, relations: 0, evidence: 0 } };

const relationStatusStyles: Record<string, { label: string; color: string; dash?: string }> = {
  reported: { label: 'Reported', color: 'var(--color-text-tertiary)' },
  hypothesis: { label: 'Hypothesis', color: 'var(--color-warning)', dash: '2 3' },
  verified: { label: 'Verified', color: 'var(--color-success)' },
  contradicted: { label: 'Contradicted', color: 'var(--color-error)', dash: '7 4' },
  superseded: { label: 'Superseded', color: 'var(--color-text-tertiary)', dash: '7 4' },
};

export function EvidenceDates({ evidence }: {
  evidence: Pick<GraphEvidence, 'source' | 'observed_at' | 'valid_from' | 'valid_until'>;
}) {
  const label = evidenceValidityLabel(evidence.valid_from, evidence.valid_until);
  const observed = evidence.observed_at;
  const capture = evidence.source === 'web' && typeof observed === 'number' && observed > 0
    ? new Date(observed * 1000)
    : null;

  return (
    <>
      {capture && Number.isFinite(capture.getTime()) && (
        <p>Page captured <time dateTime={capture.toISOString()}>{capture.toLocaleString()}</time></p>
      )}
      {label && <p>{label}</p>}
    </>
  );
}

function EvidenceEpisode({ episodeId, onOpen }: {
  episodeId?: string;
  onOpen: (episodeId: string, trigger: HTMLButtonElement) => void;
}) {
  if (!isMemoryEpisodeId(episodeId)) return null;
  return (
    <button
      type="button"
      aria-haspopup="dialog"
      className="block break-all text-left underline underline-offset-2"
      style={{ color: 'var(--color-accent)' }}
      onClick={(event) => onOpen(episodeId, event.currentTarget)}
    >
      Open episode <code>{episodeId}</code>
    </button>
  );
}

function EvidenceSourceLink({ url, label }: { url?: string; label: string }) {
  const href = safeExternalUrl(url);
  if (!href) return <span className="break-all">{label}</span>;
  return <a className="break-all underline" href={href} target="_blank" rel="noopener noreferrer">{label}</a>;
}

function entityTypeColor(type: string): string {
  let hash = 0;
  for (const character of type.trim().toLocaleLowerCase()) {
    hash = (hash * 31 + character.codePointAt(0)!) >>> 0;
  }
  return `hsl(${hash % 360} 68% 48%)`;
}

function linkedEpisodeIds(relation: GraphRelation): string[] {
  const values = relation.properties?.episode_ids;
  if (!Array.isArray(values)) return [];
  const evidenceIds = new Set(relation.evidence.map((item) => item.episode_id?.toLowerCase()));
  const ids = new Set(values.filter(isMemoryEpisodeId).map((id) => id.toLowerCase()));
  return [...ids].filter((id) => !evidenceIds.has(id)).slice(-50);
}

export function graphEventDescription(
  event: GraphEvent,
  entities: Array<Pick<GraphEntity, 'id' | 'name'>>,
): string {
  const noun = event.event_type.startsWith('entity_') ? 'entity' : 'relation';
  const verb = event.event_type.slice(noun.length + 1).replace(/_/g, ' ');
  if (event.event_type === 'graph_cleared') {
    return `Cleared graph (${event.before.entities ?? 0} entities, ${event.before.relations ?? 0} relations)`;
  }
  if (event.event_type === 'memory_data_purged') {
    return `Purged linked memory data (evidence removed: ${event.before.evidence_removed ?? 0}; entities removed: ${event.before.entities_removed ?? 0}; relations removed: ${event.before.relations_removed ?? 0}; entities updated: ${event.before.entities_updated ?? 0}; relations updated: ${event.before.relations_updated ?? 0})`;
  }
  if (event.event_type === 'entity_deleted' && event.before.details_purged === true) {
    const removed = event.before.removed_relation_count ?? 0;
    return `Deleted entity (details purged; removed ${removed} connected relations)`;
  }
  if (event.event_type === 'relation_deleted' && event.before.details_purged === true) {
    const evidence = event.before.evidence_count ?? 0;
    return `Deleted relation (details purged; ${evidence} evidence items purged)`;
  }
  if (noun === 'entity') {
    const snapshot = event.after.name ? event.after : event.before;
    const name = snapshot.name || snapshot.entity_id || event.entity_id || 'entity';
    if (event.event_type === 'entity_updated' && event.before.name && event.after.name && event.before.name !== event.after.name) {
      return `Updated entity: ${event.before.name} → ${event.after.name}`;
    }
    const affected = snapshot.removed_relation_count;
    return `${verb[0]?.toUpperCase()}${verb.slice(1)} entity: ${name}${affected ? ` (removed ${affected} connected relations)` : ''}`;
  }

  const snapshot = event.after.relation_type ? event.after : event.before;
  const entityName = (id?: string) => entities.find((entity) => entity.id === id)?.name || id || 'unknown';
  const polarity = snapshot.polarity === 'negated' ? 'NOT ' : '';
  const claim = `${entityName(snapshot.source_id)} —${polarity}${snapshot.relation_type || 'related_to'}→ ${entityName(snapshot.target_id)}`;
  return `${verb[0]?.toUpperCase()}${verb.slice(1)} relation: ${claim}`;
}

type RelationGeometry = ReturnType<typeof graphRelationGeometry<GraphRelation>>[number];

export function GraphRelationEdge({
  geometry,
  sourceName,
  targetName,
  selected,
  possibleConflict,
  reviewDue,
  onSelect,
}: {
  geometry: RelationGeometry;
  sourceName: string;
  targetName: string;
  selected: boolean;
  possibleConflict: boolean;
  reviewDue: boolean;
  onSelect: () => void;
}) {
  const { relation, path, arrow, label, labelAngle, hitWidth } = geometry;
  const relationLabel = relation.polarity === 'negated' ? `NOT ${relation.type}` : relation.type;
  const appearance = relationStatusStyles[relation.status] ?? relationStatusStyles.reported;
  const color = possibleConflict ? 'var(--color-error)' : reviewDue ? 'var(--color-warning)' : appearance.color;
  const qualifiers = [appearance.label];
  if (possibleConflict) qualifiers.push('Possible conflict');
  if (reviewDue) qualifiers.push('Evidence review due');
  const description = `Select relation ${sourceName} → ${targetName}, ${relationLabel} (${qualifiers.join(', ')})`;
  const edgeLabel = relation.status === 'reported' ? relationLabel : `${relationLabel} (${appearance.label})`;

  return (
    <g
      role="button"
      tabIndex={0}
      data-graph-relation="true"
      aria-label={description}
      aria-pressed={selected}
      className="group cursor-pointer"
      onClick={onSelect}
      onKeyDown={(event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        onSelect();
      }}
    >
      <title>{description}</title>
      <path d={path} fill="none" stroke="transparent" strokeWidth={hitWidth} pointerEvents="stroke" />
      <path
        d={path}
        fill="none"
        stroke="var(--color-accent)"
        strokeWidth="5"
        pointerEvents="none"
        className={selected ? 'opacity-65' : 'opacity-0 group-focus-visible:opacity-65'}
      />
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeDasharray={possibleConflict || reviewDue ? '5 4' : appearance.dash}
        strokeWidth={selected ? 2 : 1.5}
        pointerEvents="none"
      />
      <polygon points={arrow} fill={color} pointerEvents="none" />
      <text
        x={label.x}
        y={label.y - 4}
        transform={`rotate(${labelAngle} ${label.x} ${label.y})`}
        textAnchor="middle"
        fontSize="10"
        fontWeight={selected ? 'bold' : undefined}
        fill={color}
        stroke="var(--color-bg)"
        strokeWidth="2.5"
        paintOrder="stroke"
      >{edgeLabel}</text>
    </g>
  );
}

export function KnowledgeGraphPage() {
  const [loadedView, setLoadedView] = useState<{ path: string; data: GraphData } | null>(null);
  const [query, setQuery] = useState('');
  const [activeQuery, setActiveQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');
  const [relationStatus, setRelationStatus] = useState('all');
  const [scope, setScope] = useState<'global' | 'local'>('global');
  const [focusPages, setFocusPages] = useState<Array<number | null>>([null]);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [selectedId, setSelectedId] = useState('');
  const [focusedRelationId, setFocusedRelationId] = useState('');
  const [episodeId, setEpisodeId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [events, setEvents] = useState<GraphEvent[]>([]);
  const [eventCursor, setEventCursor] = useState<number | null>(null);
  const [eventsHaveMore, setEventsHaveMore] = useState(false);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [eventsError, setEventsError] = useState('');
  const [editName, setEditName] = useState('');
  const [editType, setEditType] = useState('concept');
  const [editProperties, setEditProperties] = useState('{}');
  const [newName, setNewName] = useState('');
  const [newType, setNewType] = useState('concept');
  const [relationTarget, setRelationTarget] = useState('');
  const [relationType, setRelationType] = useState('related_to');
  const [relationPolarity, setRelationPolarity] = useState<'affirmed' | 'negated'>('affirmed');
  const [relationQuote, setRelationQuote] = useState('');
  const [relationSourceUrl, setRelationSourceUrl] = useState('');
  const [relationValidFrom, setRelationValidFrom] = useState('');
  const [relationValidUntil, setRelationValidUntil] = useState('');
  const [editingRelation, setEditingRelation] = useState('');
  const [editRelationSource, setEditRelationSource] = useState('');
  const [editRelationTarget, setEditRelationTarget] = useState('');
  const [editRelationType, setEditRelationType] = useState('');
  const [editRelationStatus, setEditRelationStatus] = useState('reported');
  const [editRelationPolarity, setEditRelationPolarity] = useState<'affirmed' | 'negated'>('affirmed');
  const [editRelationQuote, setEditRelationQuote] = useState('');
  const [editRelationSourceUrl, setEditRelationSourceUrl] = useState('');
  const [editRelationValidFrom, setEditRelationValidFrom] = useState('');
  const [editRelationValidUntil, setEditRelationValidUntil] = useState('');
  const eventRequestSequence = useRef(0);
  const panStart = useRef<{ pointerId: number; point: GraphPanStart } | null>(null);
  const episodeTrigger = useRef<HTMLButtonElement>(null);
  const activeRelationDetails = useRef<HTMLDivElement>(null);

  const selectEntity = (id: string) => {
    setSelectedId(id);
    setFocusPages([null]);
    if (!id) setScope('global');
    setFocusedRelationId('');
    setEditingRelation('');
  };

  const selectRelation = (relation: GraphRelation) => {
    if (selectedId !== relation.source && selectedId !== relation.target) setSelectedId(relation.source);
    setFocusedRelationId(relation.id);
    if (editingRelation !== relation.id) setEditingRelation('');
  };

  const openEpisode = (id: string, trigger: HTMLButtonElement) => {
    episodeTrigger.current = trigger;
    setEpisodeId(id);
  };

  const loadEvents = useCallback(async (beforeId?: number, reset = true) => {
    const requestId = ++eventRequestSequence.current;
    setEventsLoading(true);
    setEventsError('');
    try {
      const params = new URLSearchParams({ limit: '20' });
      if (beforeId !== undefined) params.set('before_id', String(beforeId));
      const response = await apiFetch(`/v1/memory/graph/events?${params}`);
      if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
      const page = (await response.json()) as GraphEventPage;
      if (requestId !== eventRequestSequence.current) return;
      setEvents((current) => {
        if (reset) return page.events;
        const known = new Set(current.map((event) => event.id));
        return [...current, ...page.events.filter((event) => !known.has(event.id))];
      });
      setEventCursor(page.next_before_id);
      setEventsHaveMore(page.next_before_id !== null);
    } catch (reason) {
      if (requestId === eventRequestSequence.current) {
        setEventsError(reason instanceof Error ? reason.message : 'Could not load graph history.');
      }
    } finally {
      if (requestId === eventRequestSequence.current) setEventsLoading(false);
    }
  }, []);

  const focusId = scope === 'local' ? selectedId : '';
  const beforeId = focusPages[focusPages.length - 1];
  const requestPath = graphViewPath(activeQuery, focusId, beforeId);
  const data = loadedView?.path === requestPath ? loadedView.data : emptyGraph;

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError('');
    void (async () => {
      try {
        const response = await apiFetch(requestPath, { signal: controller.signal, cache: 'no-store' });
        if (!response.ok) {
          throw new Error(response.status === 404 && focusId
            ? 'This entity is no longer available. Return to the global view to choose another.'
            : 'Could not load the graph.');
        }
        const result = (await response.json()) as GraphData;
        if (controller.signal.aborted) return;
        setLoadedView({ path: requestPath, data: result });
        if (!focusId) {
          setSelectedId((current) => result.entities.some((entity) => entity.id === current) ? current : '');
        }
        setFocusedRelationId((current) => result.relations.some((relation) => relation.id === current) ? current : '');
        setEditingRelation((current) => result.relations.some((relation) => relation.id === current) ? current : '');
      } catch (reason) {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Could not load the graph.');
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    })();
    return () => controller.abort();
  }, [focusId, requestPath, refreshVersion]);

  useEffect(() => {
    void loadEvents();
    return () => { eventRequestSequence.current += 1; };
  }, [loadEvents]);

  const refreshGraph = () => {
    setRefreshVersion((version) => version + 1);
    void loadEvents();
  };

  const showGlobal = () => {
    setScope('global');
    setFocusPages([null]);
    setFocusedRelationId('');
    setEditingRelation('');
  };

  const changeFocusPage = (direction: 'previous' | 'next') => {
    const next = data.next_relation_id;
    if (direction === 'next' && (!Number.isSafeInteger(next) || !next || next <= 0)) return;
    setFocusPages((pages) => direction === 'previous'
      ? (pages.length > 1 ? pages.slice(0, -1) : pages)
      : [...pages, next as number]);
    setFocusedRelationId('');
    setEditingRelation('');
  };

  const types = useMemo(() => [...new Set(data.entities.map((entity) => entity.type))].sort(), [data.entities]);
  const visible = useMemo(() => data.entities.filter((entity) =>
    entity.id === focusId || typeFilter === 'all' || entity.type === typeFilter
  ), [data.entities, focusId, typeFilter]);
  const entityNames = useMemo(() => new Map(data.entities.map((entity) => [entity.id, entity.name])), [data.entities]);
  const selected = data.entities.find((entity) => entity.id === selectedId);
  const selectedSourceHref = safeExternalUrl(selected?.source_url);
  const positions = useMemo(() => {
    const width = 1000;
    const height = 620;
    const outer = focusId ? visible.filter((entity) => entity.id !== focusId) : visible;
    const radius = Math.min(250, 75 + outer.length * 5.1);
    const points = new Map(outer.map((entity, index) => {
      const angle = (2 * Math.PI * index / Math.max(outer.length, 1)) - Math.PI / 2;
      return [entity.id, { x: width / 2 + Math.cos(angle) * radius, y: height / 2 + Math.sin(angle) * radius }] as const;
    }));
    if (focusId) points.set(focusId, { x: width / 2, y: height / 2 });
    return points;
  }, [visible, focusId]);
  const filteredRelations = useMemo(
    () => relationStatus === 'all' ? data.relations
      : data.relations.filter((relation) => relation.status === relationStatus),
    [data.relations, relationStatus],
  );
  const visibleEdges = useMemo(
    () => graphRelationGeometry(filteredRelations, positions, { x: 500, y: 310 }),
    [filteredRelations, positions],
  );
  const selectedRelations = filteredRelations.filter(
    (relation) => relation.source === selectedId || relation.target === selectedId,
  );
  const focusedRelation = visibleEdges.find(({ relation }) =>
    relation.id === focusedRelationId && (relation.source === selectedId || relation.target === selectedId),
  )?.relation;
  const inspectedRelations = focusedRelation ? [focusedRelation] : selectedRelations;
  const possibleConflictRelationIds = useMemo(
    () => new Set(data.possible_conflicts.flatMap((conflict) => conflict.claims.map((claim) => claim.relation.id))),
    [data.possible_conflicts],
  );
  const evidenceReviewRelationIds = useMemo(
    () => new Set(data.evidence_review_candidates.map((candidate) => candidate.relation.id)),
    [data.evidence_review_candidates],
  );

  const beginPan = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.button !== 0 || (event.target instanceof Element && event.target.closest('[data-graph-node], [data-graph-relation]'))) return;
    panStart.current = { pointerId: event.pointerId, point: { clientX: event.clientX, clientY: event.clientY, ...pan } };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const movePan = (event: ReactPointerEvent<SVGSVGElement>) => {
    const start = panStart.current;
    if (!start || start.pointerId !== event.pointerId) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    setPan(graphPanFromPointer(start.point, event.clientX, event.clientY, bounds.width, bounds.height));
  };

  // A lost capture can arrive before pointer-up during a drag. Keep the gesture
  // active until the pointer is released or cancelled.
  const endPan = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (panStart.current?.pointerId === event.pointerId) panStart.current = null;
  };

  const panWithKeyboard = (event: ReactKeyboardEvent<SVGSVGElement>) => {
    if (event.target !== event.currentTarget) return;
    const delta = 40;
    const offsets: Record<string, { x: number; y: number }> = {
      ArrowLeft: { x: delta, y: 0 }, ArrowRight: { x: -delta, y: 0 },
      ArrowUp: { x: 0, y: delta }, ArrowDown: { x: 0, y: -delta },
    };
    const offset = offsets[event.key];
    if (!offset) return;
    event.preventDefault();
    setPan((current) => ({ x: current.x + offset.x, y: current.y + offset.y }));
  };

  useEffect(() => {
    setEditName(selected?.name ?? '');
    setEditType(selected?.type ?? 'concept');
    setEditProperties(JSON.stringify(selected?.properties ?? {}, null, 2));
  }, [selected?.id, selected?.name, selected?.type, selected?.properties]);

  const mutate = async (path: string, method: string, body?: unknown) => {
    const response = await apiFetch(path, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
    refreshGraph();
  };

  const saveEntity = async () => {
    if (!selected || !editName.trim()) return;
    let properties: unknown;
    try {
      properties = JSON.parse(editProperties.trim() || '{}');
    } catch {
      setError('Properties must be valid JSON.');
      return;
    }
    if (!properties || typeof properties !== 'object' || Array.isArray(properties)) {
      setError('Properties must be a JSON object.');
      return;
    }
    try {
      await mutate('/v1/memory/graph/entities', 'POST', {
        entity_id: selected.id, entity_type: editType, name: editName.trim(),
        properties, source: selected.source || 'manual',
        source_url: selected.source_url, confidence: selected.confidence, status: selected.status,
      });
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not save entity.'); }
  };

  const addEntity = async () => {
    if (!newName.trim()) return;
    const id = `${newType}:${crypto.randomUUID()}`;
    try {
      const search = newName.trim();
      await mutate('/v1/memory/graph/entities', 'POST', { entity_id: id, entity_type: newType, name: search, source: 'manual' });
      showGlobal();
      setQuery(search);
      setActiveQuery(search);
      setNewName('');
      setSelectedId(id);
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not add entity.'); }
  };

  const addRelation = async () => {
    if (!selected || !relationTarget) return;
    try {
      await mutate('/v1/memory/graph/relations', 'POST', {
        source_id: selected.id, target_id: relationTarget, relation_type: relationType.trim() || 'related_to',
        polarity: relationPolarity, source: 'manual', source_url: relationSourceUrl.trim(), evidence_quote: relationQuote.trim(), confidence: 0.8,
        valid_from: relationValidFrom || null, valid_until: relationValidUntil || null,
      });
      setRelationQuote('');
      setRelationSourceUrl('');
      setRelationValidFrom('');
      setRelationValidUntil('');
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not add relation.'); }
  };

  const beginEditRelation = (relation: GraphRelation) => {
    setEditingRelation(relation.id);
    setEditRelationSource(relation.source);
    setEditRelationTarget(relation.target);
    setEditRelationType(relation.type);
    setEditRelationStatus(relation.status);
    setEditRelationPolarity(relation.polarity);
    setEditRelationQuote('');
    setEditRelationSourceUrl('');
    setEditRelationValidFrom('');
    setEditRelationValidUntil('');
  };

  const clearEditedEvidence = () => {
    setEditRelationStatus('reported');
    setEditRelationQuote('');
    setEditRelationSourceUrl('');
    setEditRelationValidFrom('');
    setEditRelationValidUntil('');
  };

  const saveRelation = async (relation: GraphRelation) => {
    const relationType = editRelationType.trim() || relation.type;
    const identityChanged = editRelationSource !== relation.source
      || editRelationTarget !== relation.target
      || relationType !== relation.type
      || editRelationPolarity !== relation.polarity;
    try {
      await mutate(`/v1/memory/graph/relations/${encodeURIComponent(relation.id)}`, 'PUT', {
        source_id: editRelationSource, target_id: editRelationTarget,
        relation_type: relationType,
        weight: relation.weight,
        confidence: identityChanged ? 0.8 : relation.confidence,
        status: editRelationStatus, polarity: editRelationPolarity,
        source: 'manual',
        source_url: editRelationSourceUrl.trim(),
        evidence_quote: editRelationQuote.trim(),
        valid_from: editRelationValidFrom || null,
        valid_until: editRelationValidUntil || null,
      });
      setEditingRelation('');
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not edit relation.'); }
  };

  const deleteEntity = async () => {
    if (!selected || !window.confirm(`Delete “${selected.name}” and its connected relations?`)) return;
    try { await mutate(`/v1/memory/graph/entities/${encodeURIComponent(selected.id)}`, 'DELETE'); selectEntity(''); }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not delete entity.'); }
  };

  const deleteRelation = async (id: string) => {
    if (!window.confirm('Delete this relation and its evidence?')) return;
    try { await mutate(`/v1/memory/graph/relations/${encodeURIComponent(id)}`, 'DELETE'); }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not delete relation.'); }
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto p-5 md:p-8" style={{ color: 'var(--color-text)' }}>
      <div className="mx-auto flex w-full max-w-[1500px] flex-col gap-5">
        <header className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="mb-2 flex items-center gap-2" style={{ color: 'var(--color-accent)' }}><Network size={20} /><span className="text-xs font-semibold uppercase tracking-[0.2em]">A.C.E. Memory</span></div>
            <h1 className="text-2xl font-semibold">Knowledge graph</h1>
            <p className="mt-1 text-sm" style={{ color: 'var(--color-text-secondary)' }}>Local entities, relations, and evidence from saved sources.</p>
          </div>
          <button onClick={refreshGraph} className="inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }}><RefreshCw size={15} /> Refresh</button>
        </header>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
          {([
            ['Entities', data.counts.entities], ['Relations', data.counts.relations], ['Evidence records', data.counts.evidence], ['Possible conflicts', data.possible_conflicts.length], ['Evidence to recheck', data.evidence_review_candidates.length],
          ] as const).map(([label, value]) => <div key={label} className="rounded-xl border p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }}><div className="text-xs uppercase tracking-wider" style={{ color: 'var(--color-text-tertiary)' }}>{label}</div><div className="mt-1 text-2xl font-semibold">{value}</div></div>)}
        </div>

        <section className="rounded-xl border p-4" aria-labelledby="graph-history-heading" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 id="graph-history-heading" className="font-semibold">Change history</h2>
              <p className="mt-1 text-sm" style={{ color: 'var(--color-text-secondary)' }}>Recent additions, edits, and deletions stored with the local graph.</p>
            </div>
            <button type="button" onClick={() => void loadEvents()} disabled={eventsLoading} className="rounded-lg border px-3 py-2 text-sm disabled:opacity-50" style={{ borderColor: 'var(--color-border)' }}>Refresh history</button>
          </div>
          {eventsError && <p className="mt-3 text-sm" role="alert" style={{ color: 'var(--color-error)' }}>{eventsError}</p>}
          {!eventsLoading && !eventsError && events.length === 0 && <p className="mt-3 text-sm" style={{ color: 'var(--color-text-tertiary)' }}>No graph changes recorded yet.</p>}
          {!!events.length && <ol className="mt-3 divide-y" style={{ borderColor: 'var(--color-border)' }}>
            {events.map((event) => <li key={event.id} className="py-3 first:pt-0 last:pb-0">
              <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <p className="text-sm font-medium">{graphEventDescription(event, data.entities)}</p>
                <time className="text-xs" dateTime={new Date(event.created_at * 1000).toISOString()} style={{ color: 'var(--color-text-tertiary)' }}>{new Date(event.created_at * 1000).toLocaleString()}</time>
              </div>
              <p className="mt-1 text-xs" style={{ color: 'var(--color-text-tertiary)' }}>Source: {event.source || 'unspecified'} · Event {event.id}</p>
              {(Object.keys(event.before).length > 0 || Object.keys(event.after).length > 0) && <details className="mt-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                <summary className="w-fit cursor-pointer">Change details</summary>
                <pre className="mt-2 max-h-48 overflow-auto rounded-lg p-2 text-[11px]" style={{ background: 'var(--color-bg)' }}>{JSON.stringify({ before: event.before, after: event.after }, null, 2)}</pre>
              </details>}
            </li>)}
          </ol>}
          {eventsHaveMore && <button type="button" onClick={() => eventCursor !== null && void loadEvents(eventCursor, false)} disabled={eventsLoading} className="mt-3 rounded-lg border px-3 py-2 text-sm disabled:opacity-50" style={{ borderColor: 'var(--color-border)' }}>{eventsLoading ? 'Loading…' : 'Load earlier changes'}</button>}
        </section>

        {!!data.possible_conflicts.length && <section className="rounded-xl border p-4" aria-labelledby="graph-conflicts-heading" style={{ borderColor: 'var(--color-error)', background: 'var(--color-bg-secondary)' }}>
          <h2 id="graph-conflicts-heading" className="font-semibold">Possible conflicts to review</h2>
          <p className="mt-1 text-sm" style={{ color: 'var(--color-text-secondary)' }}>These claims share a subject and predicate but differ in target or polarity. They may both be valid or refer to different times; inspect the evidence before marking either claim contradicted.</p>
          <div className="mt-3 space-y-3">{data.possible_conflicts.map((conflict) => <article key={`${conflict.subject.id}-${conflict.predicate}`} className="rounded-lg border p-3" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}>
            <button type="button" onClick={() => selectEntity(conflict.subject.id)} className="text-left text-sm font-medium underline decoration-dotted underline-offset-2">{conflict.subject.name} —{conflict.predicate}→</button>
            <p className="mt-1 text-xs" style={{ color: 'var(--color-text-tertiary)' }}>{conflict.kind === 'opposing_polarity' ? 'The same target has both affirmed and negated claims.' : conflict.kind === 'different_targets' ? 'The same relation has multiple targets.' : 'The target and polarity both differ.'} Review each source; this is a candidate, not an automatic judgment.</p>
            <div className="mt-2 grid gap-2 md:grid-cols-2">{conflict.claims.map(({ relation, target }) => <div key={relation.id} className="rounded-md border p-2 text-sm" style={{ borderColor: 'var(--color-border)' }}>
              <div className="font-medium">{target.name}</div>
              <div className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>{relation.polarity} · {relation.status} · confidence {relation.confidence.toFixed(2)}</div>
              {relation.evidence.map((evidence, index) => <div key={`${relation.id}-${index}`} className="mt-2 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
                <EvidenceDates evidence={evidence} />
                {evidence.quote && <p>“{evidence.quote}”</p>}
                <EvidenceEpisode episodeId={evidence.episode_id} onOpen={openEpisode} />
                {(evidence.source_url || evidence.source) && <EvidenceSourceLink url={evidence.source_url} label={evidence.source_url || evidence.source} />}
              </div>)}
            </div>)}</div>
          </article>)}</div>
        </section>}

        {!!data.evidence_review_candidates.length && <section className="rounded-xl border p-4" aria-labelledby="graph-evidence-review-heading" style={{ borderColor: 'var(--color-warning)', background: 'var(--color-bg-secondary)' }}>
          <h2 id="graph-evidence-review-heading" className="font-semibold">Evidence review due</h2>
          <p className="mt-1 text-sm" style={{ color: 'var(--color-text-secondary)' }}>These changing claims have not had evidence recorded within their review interval. This is a reminder to re-check the source, not a claim that the information is false or outdated.</p>
          <div className="mt-3 space-y-2">{data.evidence_review_candidates.map((candidate) => <article key={candidate.relation.id} className="rounded-lg border p-3" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}>
            <button type="button" onClick={() => selectEntity(candidate.subject.id)} className="text-left text-sm font-medium underline decoration-dotted underline-offset-2">{candidate.subject.name} —{candidate.relation.type}→ {candidate.target.name}</button>
            <p className="mt-1 text-xs" style={{ color: 'var(--color-warning)' }}>Last evidence recorded {candidate.age_days} days ago · review interval {candidate.review_after_days} days · {candidate.relation.status}</p>
            {candidate.last_evidence && <div className="mt-2 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
              <EvidenceDates evidence={candidate.last_evidence} />
              {candidate.last_evidence.quote && <p>“{candidate.last_evidence.quote}”</p>}
              <EvidenceEpisode episodeId={candidate.last_evidence.episode_id} onOpen={openEpisode} />
              {(candidate.last_evidence.source_url || candidate.last_evidence.source) && <EvidenceSourceLink url={candidate.last_evidence.source_url} label={candidate.last_evidence.source_url || candidate.last_evidence.source} />}
            </div>}
          </article>)}</div>
        </section>}

        <section className="rounded-xl border p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }}>
          <div className="flex flex-wrap gap-2">
            <form className="flex min-w-[230px] flex-1 gap-2" onSubmit={(event) => { event.preventDefault(); showGlobal(); setActiveQuery(query); }}>
              <div className="flex flex-1 items-center gap-2 rounded-lg border px-3" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}><Search size={15} style={{ color: 'var(--color-text-tertiary)' }} /><input aria-label="Search graph" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search entities and connected evidence" className="w-full bg-transparent py-2 text-sm outline-none" /></div>
              <button className="rounded-lg px-4 text-sm font-medium" style={{ background: 'var(--color-accent)', color: 'white' }}>Search</button>
            </form>
            <div className="flex gap-1 rounded-lg border p-1" style={{ borderColor: 'var(--color-border)' }}><button type="button" aria-pressed={scope === 'global'} onClick={showGlobal} className="rounded px-2 py-1 text-xs" style={{ background: scope === 'global' ? 'var(--color-bg-tertiary)' : 'transparent' }}>Global</button><button type="button" aria-pressed={scope === 'local'} onClick={() => setScope('local')} disabled={!selectedId} className="rounded px-2 py-1 text-xs disabled:opacity-40" style={{ background: scope === 'local' ? 'var(--color-bg-tertiary)' : 'transparent' }}>Local focus</button></div>
            <div className="flex gap-1 rounded-lg border p-1" style={{ borderColor: 'var(--color-border)' }}><button aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(0.6, value - 0.15))} className="rounded px-2 text-sm">−</button><span className="px-1 py-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>{Math.round(zoom * 100)}%</span><button aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(1.8, value + 0.15))} className="rounded px-2 text-sm">+</button><button aria-label="Reset graph view" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} className="rounded px-2 text-xs">Reset</button></div>
            <select aria-label="Filter by entity type" value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)} className="rounded-lg border px-3 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}><option value="all">All types</option>{types.map((type) => <option key={type}>{type}</option>)}</select>
            <select aria-label="Filter relations by status" value={relationStatus} onChange={(event) => setRelationStatus(event.target.value)} className="rounded-lg border px-3 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}>
              <option value="all">All relation statuses</option>
              {Object.entries(relationStatusStyles).map(([status, appearance]) => <option key={status} value={status}>{appearance.label}</option>)}
            </select>
          </div>
          {!!types.length && <div className="mt-3 flex flex-wrap gap-x-4 gap-y-2" role="list" aria-label="Node colors by entity type">
            {types.map((type) => <span key={type} role="listitem" className="inline-flex items-center gap-1.5 text-xs" style={{ color: 'var(--color-text-secondary)' }}><span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: entityTypeColor(type) }} />{type}</span>)}
          </div>}
          {focusId && (
            <nav aria-label="Graph neighborhood pages" className="mt-3 flex flex-wrap items-center gap-2 text-sm">
              <span>{selected ? `Neighbors of ${selected.name}` : 'Neighborhood'} · Page {focusPages.length}</span>
              <button type="button" onClick={() => changeFocusPage('previous')} disabled={loading || focusPages.length <= 1} className="rounded border px-2 py-1 disabled:opacity-40" style={{ borderColor: 'var(--color-border)' }}>Previous</button>
              <button type="button" onClick={() => changeFocusPage('next')} disabled={loading || !data.next_relation_id} className="rounded border px-2 py-1 disabled:opacity-40" style={{ borderColor: 'var(--color-border)' }}>More neighbors</button>
            </nav>
          )}
          {error && <p role="alert" className="mt-3 text-sm" style={{ color: 'var(--color-error)' }}>{error}</p>}
          {loading && <p className="mt-3 text-sm" style={{ color: 'var(--color-text-secondary)' }}>Loading local graph…</p>}
          {!loading && !error && !visible.length && <p className="mt-5 rounded-lg border border-dashed p-8 text-center text-sm" style={{ borderColor: 'var(--color-border)', color: 'var(--color-text-secondary)' }}>{activeQuery ? 'No entities matched this search.' : 'No saved graph data yet. Use web research in chat to gather sources, or add an entity below.'}</p>}

          {!!visible.length && <div className="mt-4 overflow-hidden rounded-xl border" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}>
            <svg viewBox="0 0 1000 620" role="group" aria-label="Interactive local knowledge graph. Drag the background to pan; use arrow keys when the graph is focused." tabIndex={0} onKeyDown={panWithKeyboard} onPointerDown={beginPan} onPointerMove={movePan} onPointerUp={endPan} onPointerCancel={endPan} className="h-[420px] w-full touch-none cursor-grab active:cursor-grabbing focus-visible:outline focus-visible:outline-2" style={{ touchAction: 'none', outlineColor: 'var(--color-accent)' }}>
              <g transform={`translate(${pan.x} ${pan.y}) translate(500 310) scale(${zoom}) translate(-500 -310)`}>
              {visibleEdges.map((geometry) => (
                <GraphRelationEdge
                  key={geometry.relation.id}
                  geometry={geometry}
                  sourceName={entityNames.get(geometry.relation.source) || geometry.relation.source}
                  targetName={entityNames.get(geometry.relation.target) || geometry.relation.target}
                  selected={focusedRelation?.id === geometry.relation.id}
                  possibleConflict={possibleConflictRelationIds.has(geometry.relation.id)}
                  reviewDue={evidenceReviewRelationIds.has(geometry.relation.id)}
                  onSelect={() => selectRelation(geometry.relation)}
                />
              ))}
              {visible.map((entity) => {
                const point = positions.get(entity.id)!;
                const active = entity.id === selectedId;
                const fill = entityTypeColor(entity.type);
                return <g
                  key={entity.id}
                  role="button"
                  tabIndex={0}
                  data-graph-node="true"
                  aria-label={`Select ${entity.name} (${entity.type})`}
                  aria-pressed={active}
                  onClick={() => selectEntity(entity.id)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      selectEntity(entity.id);
                    }
                  }}
                  className="cursor-pointer focus-visible:opacity-80"
                >
                  <title>{entity.name} · {entity.type}</title>
                  <circle cx={point.x} cy={point.y} r={active ? 11 : 8} fill={fill} stroke={active ? 'var(--color-accent)' : 'var(--color-border)'} strokeWidth={active ? 3 : 1.5} />
                  <text x={point.x} y={point.y + 25} textAnchor="middle" fontSize="12" fill="var(--color-text)">{entity.name.length > 24 ? `${entity.name.slice(0, 22)}…` : entity.name}</text>
                </g>;
              })}
              </g>
            </svg>
            <div className="border-t px-3 py-2 text-xs" style={{ borderColor: 'var(--color-border)', color: 'var(--color-text-tertiary)' }}>
              {visible.length} nodes · {visibleEdges.length} edges shown. Drag an open area or use arrow keys with the graph focused to pan. Select a node or relation to inspect it.
              {focusedRelation && (
                <button
                  type="button"
                  className="ml-2 underline underline-offset-2"
                  style={{ color: 'var(--color-accent)' }}
                  onClick={() => activeRelationDetails.current?.focus()}
                >View relation details</button>
              )}
            </div>
          </div>}
        </section>

        <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
          <section className="rounded-xl border p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }}>
            <h2 className="font-semibold">Add entity</h2>
            <div className="mt-3 flex flex-wrap gap-2"><input aria-label="Entity name" value={newName} onChange={(event) => setNewName(event.target.value)} placeholder="Name" className="min-w-[150px] flex-1 rounded-lg border px-3 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }} /><input aria-label="Entity type" value={newType} onChange={(event) => setNewType(event.target.value)} placeholder="Type" className="w-36 rounded-lg border px-3 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }} /><button onClick={() => void addEntity()} className="inline-flex items-center gap-1 rounded-lg px-3 py-2 text-sm" style={{ background: 'var(--color-accent)', color: 'white' }}><Plus size={14} /> Add</button></div>
          </section>

          <section className="rounded-xl border p-4" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }}>
            <h2 className="font-semibold">Entity details</h2>
            {!selected && <p className="mt-3 text-sm" style={{ color: 'var(--color-text-secondary)' }}>Select an entity or relation in the graph to view evidence or make changes.</p>}
            {selected && <div className="mt-3 space-y-3">
              <div className="flex flex-wrap gap-2"><input aria-label="Edit entity name" value={editName} onChange={(event) => setEditName(event.target.value)} className="min-w-[160px] flex-1 rounded-lg border px-3 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }} /><input aria-label="Edit entity type" value={editType} onChange={(event) => setEditType(event.target.value)} className="w-36 rounded-lg border px-3 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }} /><button onClick={() => void saveEntity()} title="Save changes" className="rounded-lg border p-2" style={{ borderColor: 'var(--color-border)' }}><Save size={16} /></button><button onClick={() => void deleteEntity()} title="Delete entity" className="rounded-lg border p-2" style={{ borderColor: 'var(--color-border)', color: 'var(--color-error)' }}><Trash2 size={16} /></button></div>
              <div className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>{selected.type} · {selected.status} · confidence {selected.confidence.toFixed(2)} · {selected.source || 'local'}</div>
              <label className="block text-xs font-medium" htmlFor="graph-entity-properties">Properties (JSON object)</label>
              <textarea
                id="graph-entity-properties"
                aria-label="Edit entity properties as JSON"
                value={editProperties}
                onChange={(event) => setEditProperties(event.target.value)}
                rows={5}
                spellCheck={false}
                className="w-full rounded-lg border px-3 py-2 font-mono text-xs"
                style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
              />
              {selected.source_url && (selectedSourceHref
                ? <a href={selectedSourceHref} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-sm" style={{ color: 'var(--color-accent)' }}>{selected.source_url}<ExternalLink size={13} /></a>
                : <span className="break-all text-sm" style={{ color: 'var(--color-text-secondary)' }}>{selected.source_url}</span>)}
              {relationStatus !== 'all' && !selectedRelations.length && <p className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>No connected relations match this status.</p>}
              {focusedRelation && (
                <button
                  type="button"
                  className="text-sm underline underline-offset-2"
                  style={{ color: 'var(--color-accent)' }}
                  onClick={() => setFocusedRelationId('')}
                >Show all connected relations</button>
              )}
              <div className="space-y-2">{inspectedRelations.map((relation) => <div
                key={relation.id}
                ref={focusedRelation?.id === relation.id ? activeRelationDetails : undefined}
                tabIndex={focusedRelation?.id === relation.id ? -1 : undefined}
                role={focusedRelation?.id === relation.id ? 'group' : undefined}
                aria-label={focusedRelation?.id === relation.id ? 'Selected relation details' : undefined}
                className="rounded-lg border p-3 text-sm"
                style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{entityNames.get(relation.source) || relation.source} → {entityNames.get(relation.target) || relation.target}</span>
                  <span className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>{relation.polarity} {relation.type} · {relation.status} · {relation.confidence.toFixed(2)}</span>
                  <button onClick={() => beginEditRelation(relation)} className="ml-auto rounded p-1" title="Edit relation"><Pencil size={13} /></button>
                  <button onClick={() => void deleteRelation(relation.id)} className="rounded p-1" title="Delete relation"><Trash2 size={13} /></button>
                </div>
                {relation.evidence.map((evidence, index) => <div key={`${relation.id}-${index}`} className="mt-2 text-xs" style={{ color: 'var(--color-text-secondary)' }}><span>{evidence.status} evidence · {evidence.source || 'manual'}</span><EvidenceDates evidence={evidence} />{evidence.quote && <p>“{evidence.quote}”</p>}<EvidenceEpisode episodeId={evidence.episode_id} onOpen={openEpisode} />{evidence.source_url && <EvidenceSourceLink url={evidence.source_url} label={evidence.source_url} />}</div>)}
                {linkedEpisodeIds(relation).map((id) => (
                  <div key={id} className="mt-2 text-xs">
                    <EvidenceEpisode episodeId={id} onOpen={openEpisode} />
                  </div>
                ))}
                {editingRelation === relation.id && (
                  <div className="mt-3 flex flex-wrap gap-2 border-t pt-3" style={{ borderColor: 'var(--color-border)' }}>
                    <select
                      aria-label="Edit relation source"
                      value={editRelationSource}
                      onChange={(event) => {
                        if (event.target.value !== relation.source) clearEditedEvidence();
                        setEditRelationSource(event.target.value);
                      }}
                      className="min-w-[150px] flex-1 rounded-lg border px-2 py-2 text-xs"
                      style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }}
                    >
                      {data.entities.map((entity) => <option key={entity.id} value={entity.id}>{entity.name} · {entity.type}</option>)}
                    </select>
                    <select
                      aria-label="Edit relation target"
                      value={editRelationTarget}
                      onChange={(event) => {
                        if (event.target.value !== relation.target) clearEditedEvidence();
                        setEditRelationTarget(event.target.value);
                      }}
                      className="min-w-[150px] flex-1 rounded-lg border px-2 py-2 text-xs"
                      style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }}
                    >
                      {data.entities.map((entity) => <option key={entity.id} value={entity.id}>{entity.name} · {entity.type}</option>)}
                    </select>
                    <input
                      aria-label="Edit relation type"
                      value={editRelationType}
                      onChange={(event) => {
                        if (event.target.value !== relation.type) clearEditedEvidence();
                        setEditRelationType(event.target.value);
                      }}
                      className="min-w-[130px] flex-1 rounded-lg border px-2 py-2 text-xs"
                      style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }}
                    />
                    <select
                      aria-label="Edit relation polarity"
                      value={editRelationPolarity}
                      onChange={(event) => {
                        if (event.target.value !== relation.polarity) clearEditedEvidence();
                        setEditRelationPolarity(event.target.value as 'affirmed' | 'negated');
                      }}
                      className="rounded-lg border px-2 py-2 text-xs"
                      style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }}
                    >
                      <option value="affirmed">affirmed</option>
                      <option value="negated">negated</option>
                    </select>
                    <select
                      aria-label="Edit relation status"
                      value={editRelationStatus}
                      onChange={(event) => setEditRelationStatus(event.target.value)}
                      className="rounded-lg border px-2 py-2 text-xs"
                      style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }}
                    >
                      {Object.entries(relationStatusStyles).map(([status, appearance]) => <option key={status} value={status}>{appearance.label}</option>)}
                    </select>
                    <input type="date" aria-label="Evidence valid from" value={editRelationValidFrom} onChange={(event) => setEditRelationValidFrom(event.target.value)} className="rounded-lg border px-2 py-2 text-xs" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }} />
                    <input type="date" aria-label="Evidence valid until" value={editRelationValidUntil} onChange={(event) => setEditRelationValidUntil(event.target.value)} className="rounded-lg border px-2 py-2 text-xs" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }} />
                    <input aria-label="Add relation evidence" value={editRelationQuote} onChange={(event) => setEditRelationQuote(event.target.value)} placeholder="New supporting quote (optional)" className="min-w-[180px] flex-1 rounded-lg border px-2 py-2 text-xs" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }} />
                    <input aria-label="New evidence source URL" value={editRelationSourceUrl} onChange={(event) => setEditRelationSourceUrl(event.target.value)} placeholder="Source URL" className="min-w-[180px] flex-1 rounded-lg border px-2 py-2 text-xs" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }} />
                    <button onClick={() => void saveRelation(relation)} className="rounded-lg p-2" title="Save relation"><Check size={14} /></button>
                    <button onClick={() => setEditingRelation('')} className="rounded-lg p-2" title="Cancel"><X size={14} /></button>
                  </div>
                )}
              </div>)}</div>
            <div className="rounded-lg border p-3" style={{ borderColor: 'var(--color-border)' }}>
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--color-text-tertiary)' }}>Add relation</div>
              <div className="flex flex-wrap gap-2">
                <select aria-label="Relation target" value={relationTarget} onChange={(event) => setRelationTarget(event.target.value)} className="min-w-[130px] flex-1 rounded-lg border px-2 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}><option value="">Choose target</option>{data.entities.filter((entity) => entity.id !== selected.id).map((entity) => <option key={entity.id} value={entity.id}>{entity.name}</option>)}</select>
                <input aria-label="Relation type" value={relationType} onChange={(event) => setRelationType(event.target.value)} className="w-36 rounded-lg border px-2 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }} />
                <select aria-label="Relation polarity" value={relationPolarity} onChange={(event) => setRelationPolarity(event.target.value as 'affirmed' | 'negated')} className="rounded-lg border px-2 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}><option value="affirmed">affirmed</option><option value="negated">negated</option></select>
                <input type="date" aria-label="Relation evidence valid from" value={relationValidFrom} onChange={(event) => setRelationValidFrom(event.target.value)} className="rounded-lg border px-2 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }} />
                <input type="date" aria-label="Relation evidence valid until" value={relationValidUntil} onChange={(event) => setRelationValidUntil(event.target.value)} className="rounded-lg border px-2 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }} />
                <input aria-label="Evidence quote" value={relationQuote} onChange={(event) => setRelationQuote(event.target.value)} placeholder="Evidence quote (optional)" className="min-w-[150px] flex-1 rounded-lg border px-2 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }} />
                <input aria-label="Evidence source URL" value={relationSourceUrl} onChange={(event) => setRelationSourceUrl(event.target.value)} placeholder="Source URL (optional)" className="min-w-[150px] flex-1 rounded-lg border px-2 py-2 text-sm" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }} />
                <button onClick={() => void addRelation()} disabled={!relationTarget} className="rounded-lg px-3 py-2 text-sm disabled:opacity-40" style={{ background: 'var(--color-accent)', color: 'white' }}>Connect</button>
              </div>
            </div>
            </div>}
          </section>
        </div>
      </div>
      <GraphEpisodeDialog episodeId={episodeId} onClose={() => setEpisodeId(null)} returnFocus={episodeTrigger} />
    </div>
  );
}
