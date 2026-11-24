import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { WebSourceList } from './WebResearchPage';

const source = {
  title: '<fixture-title>Captured title</fixture-title>',
  url: 'https://example.org/research',
  saved_at: '2026-11-23T09:15:00Z',
  text: '<script>private fixture text</script>',
  capture: 'page',
  revision: 'page-revision',
} as const;

describe('saved web provenance', () => {
  it('renders untrusted title and content literally and protects the source link', () => {
    const html = renderToStaticMarkup(<WebSourceList sources={[source]} />);
    expect(html).toContain('&lt;fixture-title&gt;Captured title&lt;/fixture-title&gt;');
    expect(html).toContain('&lt;script&gt;private fixture text&lt;/script&gt;');
    expect(html).not.toContain('<script>private fixture text</script>');
    expect(html).toContain('href="https://example.org/research"');
    expect(html).toContain('rel="noopener noreferrer"');
    expect(html).toContain('dateTime="2026-11-23T09:15:00Z"');
    expect(html).toContain('Page text');
  });

  it('leaves an unsafe legacy URL as plain text and labels a saved snippet', () => {
    const html = renderToStaticMarkup(<WebSourceList sources={[{
      ...source, url: 'javascript:alert(42)', capture: 'snippet',
    }]} />);
    expect(html).toContain('javascript:alert(42)');
    expect(html).not.toContain('href=');
    expect(html).toContain('Search snippet');
  });
});

describe('saved page graph extraction controls', () => {
  it('offers extraction only for a full page with a revision when inference is available', () => {
    const sources = [
      source,
      { ...source, url: 'https://example.org/snippet', capture: 'snippet' as const },
      { ...source, url: 'https://example.org/legacy', revision: '' },
    ];
    const html = renderToStaticMarkup(
      <WebSourceList sources={sources} extraction={{ pending: null, feedback: new Map(), start: () => {} }} />,
    );
    expect(html.match(/Extract to graph/g)).toHaveLength(1);

    const unavailable = renderToStaticMarkup(
      <WebSourceList sources={sources} extraction={{ pending: null, feedback: new Map() }} />,
    );
    expect(unavailable).toContain('Read saved text');
    expect(unavailable).not.toContain('Extract to graph');
  });

  it('reports zero relations on the selected revision', () => {
    const result = { url: source.url, saved_at: source.saved_at, entities_stored: 1, relations_stored: 0 };
    const html = renderToStaticMarkup(<WebSourceList sources={[source]} extraction={{
      pending: null, feedback: new Map([[source.revision, { result }]]),
    }} />);
    expect(html).toContain('role="status"');
    expect(html).toContain('No relations found. Stored 1 entity in the graph.');
  });
});
