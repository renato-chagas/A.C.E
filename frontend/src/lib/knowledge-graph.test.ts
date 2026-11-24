import { describe, expect, it } from 'vitest';

import { evidenceValidityLabel, graphRelationGeometry, safeExternalUrl } from './knowledge-graph';

describe('evidenceValidityLabel', () => {
  const now = new Date(2026, 10, 18, 23, 55);

  it('classifies past and future evidence by the local date', () => {
    expect(evidenceValidityLabel('2026-11-10', '2026-11-17', now)).toBe(
      'Historical validity: 2026-11-10 through 2026-11-17',
    );
    expect(evidenceValidityLabel('2026-11-19', undefined, now)).toBe(
      'Future validity: 2026-11-19 through end unspecified',
    );
  });

  it('includes the boundary dates and leaves undated evidence unlabeled', () => {
    expect(evidenceValidityLabel('2026-11-18', '2026-11-18', now)).toBe(
      'Stated validity: 2026-11-18 through 2026-11-18',
    );
    expect(evidenceValidityLabel(undefined, '2026-11-18', now)).toBe(
      'Stated validity: start unspecified through 2026-11-18',
    );
    expect(evidenceValidityLabel(undefined, '', now)).toBe('');
  });
});

describe('safeExternalUrl', () => {
  it('accepts ordinary HTTP and HTTPS sources', () => {
    expect(safeExternalUrl('https://example.com/source')).toBe(
      'https://example.com/source',
    );
    expect(safeExternalUrl('http://example.com/source')).toBe(
      'http://example.com/source',
    );
  });

  it('rejects executable, local-file, credentialed, and malformed URLs', () => {
    expect(safeExternalUrl('javascript:alert(1)')).toBeNull();
    expect(safeExternalUrl('data:text/html,<script>alert(1)</script>')).toBeNull();
    expect(safeExternalUrl('file:///C:/Users/renat/secret.txt')).toBeNull();
    expect(safeExternalUrl('https://user:password@example.com/')).toBeNull();
    expect(safeExternalUrl('not a URL')).toBeNull();
    expect(safeExternalUrl('')).toBeNull();
  });
});

describe('graphRelationGeometry', () => {
  const center = { x: 160, y: 100 };
  const positions = new Map([
    ['ace', { x: 40, y: 100 }],
    ['jarvis', { x: 280, y: 100 }],
  ]);
  const claims = [
    { id: 'a', source: 'ace', target: 'jarvis' },
    { id: 'b', source: 'ace', target: 'jarvis' },
    { id: 'c', source: 'jarvis', target: 'ace' },
  ];

  it('draws parallel and opposing claims in stable, separately labeled lanes', () => {
    const geometry = graphRelationGeometry(claims, positions, center);
    expect(graphRelationGeometry([...claims].reverse(), positions, center)).toEqual(geometry);
    expect(new Set(geometry.map((edge) => edge.path)).size).toBe(claims.length);
    expect(new Set(geometry.map((edge) => edge.label.y)).size).toBe(claims.length);
    expect(geometry[0].label.y).toBeLessThan(center.y);
    expect(geometry[2].label.y).toBeGreaterThan(center.y);
  });

  it('points each arrow toward the correct endpoint outside its node', () => {
    const geometry = graphRelationGeometry(claims, positions, center);
    for (const edge of geometry) {
      const [tipX, tipY, base1X, base1Y, base2X, base2Y] = edge.arrow.split(' ').map(Number);
      const target = positions.get(edge.relation.target)!;
      const distance = Math.hypot(tipX - target.x, tipY - target.y);
      const base = { x: (base1X + base2X) / 2, y: (base1Y + base2Y) / 2 };
      expect(distance).toBeCloseTo(14);
      expect(Math.hypot(base.x - target.x, base.y - target.y)).toBeGreaterThan(distance);
      expect(edge.labelAngle).toBeGreaterThanOrEqual(-90);
      expect(edge.labelAngle).toBeLessThanOrEqual(90);
    }
  });

  it('draws separate self loops and omits claims whose endpoints are hidden', () => {
    const geometry = graphRelationGeometry([
      { id: 'self-a', source: 'ace', target: 'ace' },
      { id: 'self-b', source: 'ace', target: 'ace' },
      { id: 'hidden', source: 'ace', target: 'not-visible' },
    ], positions, center);
    expect(geometry.map(({ relation }) => relation.id)).toEqual(['self-a', 'self-b']);
    expect(geometry.every(({ path }) => path.includes(' C '))).toBe(true);
    expect(geometry[0].path).not.toBe(geometry[1].path);
    expect(geometry[0].arrow).not.toBe(geometry[1].arrow);
  });
});
