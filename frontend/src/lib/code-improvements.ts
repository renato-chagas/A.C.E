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

export function createCodeImprovementsClient() {
  const origin = resolveLoopbackApiOrigin();
  if (!origin) return null;
  const endpoint = origin + '/v1/code-improvements';

  async function get(path: string, signal: AbortSignal): Promise<unknown> {
    let response: Response;
    try {
      response = await fetch(endpoint + path, {
        method: 'GET',
        headers: authHeaders(),
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
    if (response.status === 404) throw new Error('This proposal is unavailable. Refresh the list to check.');
    if (!response.ok) throw new Error('Could not read code improvements from ACE.');
    return response.json().catch(() => { throw new Error('ACE returned an invalid proposal response.'); });
  }

  return {
    async list(signal: AbortSignal): Promise<CodeImprovementSummary[]> {
      const body = await get('?limit=50', signal) as { proposals?: unknown } | null;
      if (!Array.isArray(body?.proposals) || !body.proposals.every(isSummary)) {
        throw new Error('ACE returned an invalid proposal list.');
      }
      return body.proposals;
    },
    async detail(id: string, signal: AbortSignal): Promise<CodeImprovementProposal> {
      if (!proposalIdPattern.test(id)) throw new Error('Invalid proposal ID.');
      const body = await get('/' + id, signal);
      if (!isSummary(body) || body.proposal_id !== id) {
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
    },
  };
}
