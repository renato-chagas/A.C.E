import { authHeaders } from './api';
import { resolveLoopbackApiOrigin } from './loopback-api';

export type CodeImprovementSummary = {
  proposal_id: string;
  title: string;
  status: string;
  created_at: number;
  files: string[];
  rollback_available: boolean;
};

export type CodeImprovementProposal = CodeImprovementSummary & {
  repository: string;
  patch: string;
  validation: string;
  validation_receipt: string;
};

export type CodeImprovementDraft = {
  repository: string;
  task: string;
  files: string[];
};

const proposalIdPattern = /^[0-9a-f]{12}$/;

function isSummary(value: unknown): value is CodeImprovementSummary {
  if (!value || typeof value !== 'object') return false;
  const item = value as Partial<CodeImprovementSummary>;
  return typeof item.proposal_id === 'string' && proposalIdPattern.test(item.proposal_id) &&
    typeof item.title === 'string' &&
    typeof item.status === 'string' &&
    typeof item.created_at === 'number' && Number.isFinite(item.created_at) &&
    Array.isArray(item.files) && item.files.every((file) => typeof file === 'string') &&
    typeof item.rollback_available === 'boolean';
}

function readProposal(body: unknown, expectedId?: string): CodeImprovementProposal {
  if (!isSummary(body) || (expectedId && body.proposal_id !== expectedId)) {
    throw new Error('ACE returned an invalid proposal.');
  }
  const { repository, patch, validation, validation_receipt } = body as Partial<CodeImprovementProposal>;
  if (
    typeof repository !== 'string' || typeof patch !== 'string' ||
    typeof validation !== 'string' || typeof validation_receipt !== 'string'
  ) {
    throw new Error('ACE returned an invalid proposal.');
  }
  return { ...body, repository, patch, validation, validation_receipt };
}

export function createCodeImprovementsClient() {
  const origin = resolveLoopbackApiOrigin();
  if (!origin) return null;
  const endpoint = origin + '/v1/code-improvements';

  async function request(path: string, signal: AbortSignal, draft?: CodeImprovementDraft): Promise<unknown> {
    let response: Response;
    try {
      response = await fetch(endpoint + path, {
        method: draft ? 'POST' : 'GET',
        headers: authHeaders(draft ? { 'Content-Type': 'application/json' } : undefined),
        body: draft ? JSON.stringify(draft) : undefined,
        cache: 'no-store',
        credentials: 'omit',
        redirect: 'error',
        referrerPolicy: 'no-referrer',
        signal,
      });
    } catch (error) {
      if (signal.aborted) throw error;
      throw new Error('Could not reach the local ACE server.');
    }
    if (response.status === 401) throw new Error('The local ACE connection needs a valid API key.');
    if (response.status === 403) throw new Error('Code improvements require direct access from this computer.');
    if (draft && !response.ok) {
      const errors: Readonly<Record<number, string>> = {
        404: 'Update the local ACE backend to generate proposals here.',
        405: 'Update the local ACE backend to generate proposals here.',
        409: 'A proposal is already being generated. Refresh in a moment.',
        422: 'ACE could not use this request. Check the Git repository, selected files, and instructions.',
        503: 'Proposal generation is unavailable. Select a running local model and check that Git is installed.',
      };
      throw new Error(errors[response.status] ?? 'Could not generate the proposal.');
    }
    if (response.status === 404) throw new Error('This proposal is unavailable. Refresh the list to check.');
    if (!response.ok) throw new Error('Could not read code improvements from ACE.');
    return response.json().catch(() => { throw new Error('ACE returned an invalid proposal response.'); });
  }

  return {
    async list(signal: AbortSignal): Promise<CodeImprovementSummary[]> {
      const body = await request('?limit=50', signal) as { proposals?: unknown } | null;
      if (!Array.isArray(body?.proposals) || !body.proposals.every(isSummary)) {
        throw new Error('ACE returned an invalid proposal list.');
      }
      return body.proposals;
    },
    async detail(id: string, signal: AbortSignal): Promise<CodeImprovementProposal> {
      if (!proposalIdPattern.test(id)) throw new Error('Invalid proposal ID.');
      return readProposal(await request('/' + id, signal), id);
    },
    async generate(draft: CodeImprovementDraft, signal: AbortSignal): Promise<CodeImprovementProposal> {
      return readProposal(await request('', signal, draft));
    },
  };
}
