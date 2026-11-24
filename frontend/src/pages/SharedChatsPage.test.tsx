import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { SharedChatsPage, SharedChatTranscript } from './SharedChatsPage';

vi.mock('../lib/api', () => ({
  getBase: () => 'http://127.0.0.1:8000',
  isTauri: () => true,
  authHeaders: () => ({}),
}));

describe('shared chat transcript', () => {
  it('keeps model and user text inert', () => {
    const html = renderToStaticMarkup(
      <SharedChatTranscript turns={[['<script>alert(1)</script>', '[link](https://remote.example)']]} latest={null} />,
    );
    expect(html).toContain('&lt;script&gt;');
    expect(html).toContain('[link](https://remote.example)');
    expect(html).not.toContain('<script');
    expect(html).not.toContain('<a ');
  });

  it('shows an unsaved reply without dropping the saved transcript', () => {
    const html = renderToStaticMarkup(
      <SharedChatTranscript
        turns={[['Earlier message', 'Saved reply']]}
        latest={{ user: 'New message', answer: 'One-time reply', saved: false }}
      />,
    );
    expect(html).toContain('Saved reply');
    expect(html).toContain('One-time reply');
    expect(html).toContain('Unsaved reply');
  });

  it('shows the full latest answer while keeping one exchange per completed turn', () => {
    const html = renderToStaticMarkup(
      <SharedChatTranscript
        turns={[['A question', 'Saved excerpt']]}
        latest={{ user: 'A question', answer: 'Full reply', saved: true }}
      />,
    );
    expect(html).toContain('Full reply');
    expect(html).not.toContain('Saved excerpt');
    expect(html).not.toContain('Unsaved reply');
  });

  it('offers opening or creating a local chat in services-only mode', () => {
    const html = renderToStaticMarkup(<SharedChatsPage localServicesOnly />);
    expect(html).toContain('Open or create');
    expect(html).not.toContain('<textarea');
  });
});
