import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import { CodeImprovementDetail, CodeImprovementsPage } from './CodeImprovementsPage';
import type { CodeImprovementProposal } from '../lib/code-improvements';

vi.mock('../lib/api', () => ({
  getBase: () => 'http://127.0.0.1:8000',
  isTauri: () => true,
  authHeaders: () => ({}),
}));

const proposal: CodeImprovementProposal = {
  proposal_id: '123abc456def',
  title: 'Improve ACE',
  status: 'proposed',
  files: ['src/ace.py'],
  created_at: 1_700_000_000,
  rollback_available: false,
  repository: 'A:/fixture',
  patch: '<script>alert(1)</script>',
  validation: '<img src="https://untrusted.example" />',
  validation_receipt: '',
};

describe('code improvement proposal review', () => {
  it('renders patch and report as literal text and requires no model on entry', () => {
    const page = renderToStaticMarkup(<CodeImprovementsPage />);
    expect(page).toContain('Code improvements');

    const detail = renderToStaticMarkup(<CodeImprovementDetail proposal={proposal} />);
    expect(detail).toContain('A:/fixture');
    expect(detail).toContain('src/ace.py');
    expect(detail).toContain('&lt;script&gt;');
    expect(detail).toContain('&lt;img');
    expect(detail).not.toContain('<script');
    expect(detail).not.toContain('<img');
    expect(detail).not.toContain('<form');
  });

  it.each([
    ['proposed', '', false, 'Proposed', 'validate'],
    ['validated', '', false, 'Validated', 'validate'],
    ['approved', '', false, 'Approved', 'validate'],
    ['validated', 'receipt', false, 'Validated', 'apply'],
    ['approved', 'receipt', false, 'Approved', 'apply'],
    ['applying', 'receipt', false, 'Applying', null],
    ['applied', 'receipt', false, 'Applied', null],
    ['applied', 'receipt', true, 'Applied', 'rollback'],
    ['rolling_back', 'receipt', true, 'Rollback in progress', 'rollback'],
    ['rolled_back', 'receipt', true, 'Rolled back', null],
    ['rejected', '', false, 'Rejected', null],
  ])('shows only the relevant terminal commands when status is %s', (status, receipt, available, label, action) => {
    const html = renderToStaticMarkup(
      <CodeImprovementDetail proposal={{
        ...proposal,
        status,
        validation_receipt: receipt,
        rollback_available: available,
      }} />,
    );
    expect(html).toContain(label);
    expect(html).toContain('jarvis improve show 123abc456def');
    for (const command of ['validate', 'apply', 'rollback']) {
      const commandText = 'jarvis improve ' + command + ' 123abc456def';
      if (command === action) expect(html).toContain(commandText);
      else expect(html).not.toContain(commandText);
    }
  });
});
