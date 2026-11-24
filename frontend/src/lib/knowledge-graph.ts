export type GraphPanStart = {
  clientX: number;
  clientY: number;
  x: number;
  y: number;
};

export type GraphPan = { x: number; y: number };

type GraphPoint = { x: number; y: number };
type GraphRelationEndpoints = { id: string; source: string; target: string };

const edgeNodeClearance = 14;

function movePoint(point: GraphPoint, direction: GraphPoint, distance: number): GraphPoint {
  return { x: point.x + direction.x * distance, y: point.y + direction.y * distance };
}

function direction(from: GraphPoint, to: GraphPoint): GraphPoint {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy);
  return length ? { x: dx / length, y: dy / length } : { x: 1, y: 0 };
}

function pointCoordinates(point: GraphPoint): string {
  return point.x + ' ' + point.y;
}

function arrowPoints(tangent: GraphPoint, tip: GraphPoint): string {
  const base = movePoint(tip, tangent, -9);
  const normal = { x: -tangent.y, y: tangent.x };
  return [tip, movePoint(base, normal, 3.5), movePoint(base, normal, -3.5)]
    .map(pointCoordinates).join(' ');
}

function uprightAngle(vector: GraphPoint): number {
  const angle = Math.atan2(vector.y, vector.x) * 180 / Math.PI;
  if (angle > 90) return angle - 180;
  if (angle < -90) return angle + 180;
  return angle;
}

function relationCurve(
  source: GraphPoint,
  target: GraphPoint,
  canonicalOrder: boolean,
  lane: number,
  spacing: number,
) {
  const along = direction(source, target);
  // The perpendicular is based on the unordered pair, so reversing the arrow
  // changes its terminal point without mirroring it onto a sibling lane.
  const normal = canonicalOrder ? { x: -along.y, y: along.x } : { x: along.y, y: -along.x };
  let bend = lane * spacing;
  if (Math.hypot(source.x - target.x, source.y - target.y) < 2 * edgeNodeClearance + 8) {
    // Bow edges away from crowded nodes before clipping the ends.
    bend += lane < 0 ? -32 : 32;
  }
  const midpoint = { x: (source.x + target.x) / 2, y: (source.y + target.y) / 2 };
  const control = movePoint(midpoint, normal, bend);
  const start = movePoint(source, direction(source, control), edgeNodeClearance);
  const tangent = direction(control, target);
  const end = movePoint(target, tangent, -edgeNodeClearance);
  return {
    path: 'M ' + pointCoordinates(start) + ' Q ' + pointCoordinates(control) + ' ' + pointCoordinates(end),
    arrow: arrowPoints(tangent, end),
    label: {
      x: (start.x + 2 * control.x + end.x) / 4,
      y: (start.y + 2 * control.y + end.y) / 4,
    },
    labelAngle: uprightAngle(direction(start, end)),
    hitWidth: Math.max(4, Math.min(14, spacing * 0.42)),
  };
}

function relationLoop(point: GraphPoint, center: GraphPoint, lane: number, count: number) {
  const towardCenter = direction(point, center);
  const angle = Math.atan2(towardCenter.y, towardCenter.x)
    + lane * Math.min(0.65, 5 / Math.max(1, count - 1));
  const outward = { x: Math.cos(angle), y: Math.sin(angle) };
  const across = { x: outward.y, y: -outward.x };
  const rim = movePoint(point, outward, edgeNodeClearance / 2);
  const start = movePoint(rim, across, -edgeNodeClearance * Math.sqrt(3) / 2);
  const end = movePoint(rim, across, edgeNodeClearance * Math.sqrt(3) / 2);
  const reach = movePoint(point, outward, 64 + 2 * Math.min(12, Math.abs(lane)));
  const left = movePoint(reach, across, -28);
  const right = movePoint(reach, across, 28);
  return {
    path: 'M ' + pointCoordinates(start) + ' C ' + [left, right, end].map(pointCoordinates).join(' '),
    arrow: arrowPoints(direction(right, end), end),
    label: {
      x: (start.x + 3 * left.x + 3 * right.x + end.x) / 8,
      y: (start.y + 3 * left.y + 3 * right.y + end.y) / 8,
    },
    labelAngle: uprightAngle(across),
    hitWidth: count < 5 ? 14 : 8,
  };
}

/** Lay out the stored relations once per set of visible node positions. */
export function graphRelationGeometry<T extends GraphRelationEndpoints>(
  relations: readonly T[],
  positions: ReadonlyMap<string, GraphPoint>,
  center: GraphPoint,
) {
  const pairs = new Map<string, T[]>();
  for (const relation of relations) {
    if (!positions.has(relation.source) || !positions.has(relation.target)) continue;
    const key = JSON.stringify([relation.source, relation.target].sort());
    const pair = pairs.get(key);
    if (pair) pair.push(relation);
    else pairs.set(key, [relation]);
  }

  return Array.from(pairs.values()).flatMap((pair) => {
    pair.sort((left, right) => left.id < right.id ? -1 : left.id > right.id ? 1 : 0);
    const middle = (pair.length - 1) / 2;
    const spacing = Math.min(42, 380 / Math.max(pair.length - 1, 1));
    return pair.map((relation, index) => {
      const source = positions.get(relation.source)!;
      const lane = index - middle;
      const geometry = relation.source === relation.target
        ? relationLoop(source, center, lane, pair.length)
        : relationCurve(source, positions.get(relation.target)!, relation.source < relation.target, lane, spacing);
      return { relation, ...geometry };
    });
  });
}

export function safeExternalUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim()) return null;
  try {
    const url = new URL(value.trim());
    if (!['http:', 'https:'].includes(url.protocol) || !url.hostname || url.username || url.password) return null;
    return url.href;
  } catch {
    return null;
  }
}

export function evidenceValidityLabel(
  validFrom?: string,
  validUntil?: string,
  now = new Date(),
): string {
  if (!validFrom && !validUntil) return '';
  const year = String(now.getFullYear()).padStart(4, '0');
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const today = `${year}-${month}-${day}`;
  const period = `${validFrom || 'start unspecified'} through ${validUntil || 'end unspecified'}`;
  if (validUntil && validUntil < today) return `Historical validity: ${period}`;
  if (validFrom && validFrom > today) return `Future validity: ${period}`;
  return `Stated validity: ${period}`;
}

export function graphPanFromPointer(
  start: GraphPanStart,
  clientX: number,
  clientY: number,
  viewportWidth: number,
  viewportHeight: number,
): GraphPan {
  if (viewportWidth <= 0 || viewportHeight <= 0) return { x: start.x, y: start.y };
  return {
    x: start.x + ((clientX - start.clientX) * 1000) / viewportWidth,
    y: start.y + ((clientY - start.clientY) * 620) / viewportHeight,
  };
}


/** Request either a global search or one page around a specific entity. */
export function graphViewPath(query: string, focusId: string, beforeId: number | null): string {
  const params = new URLSearchParams({ limit: focusId ? '32' : '160' });
  if (focusId) {
    params.set('focus_id', focusId);
    if (beforeId !== null) params.set('before_relation_id', String(beforeId));
  } else if (query.trim()) {
    params.set('query', query.trim());
  }
  return '/v1/memory/graph?' + params;
}
