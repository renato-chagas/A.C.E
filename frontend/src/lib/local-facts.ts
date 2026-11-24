import { authHeaders, getBase } from './api';

export type MemoryFact = {
  text: string;
  source: string;
  trust: string;
  superseded: boolean;
  trusted_for_recall: boolean;
};

async function requestFacts(path: string, options: RequestInit): Promise<Response> {
  // Use a direct local address for private text, including in browser preview
  // where the generic same-origin proxy can point at a different server.
  let endpoint: URL;
  try {
    endpoint = new URL(getBase() || 'http://127.0.0.1:8000');
  } catch {
    throw new Error('Memory Facts requires a local ACE API URL.');
  }
  if (
    !['http:', 'https:'].includes(endpoint.protocol) ||
    endpoint.username ||
    endpoint.password ||
    endpoint.search ||
    endpoint.hash ||
    !(endpoint.hostname === 'localhost' ||
      endpoint.hostname === '[::1]' ||
      /^127\.(?:\d+\.){2}\d+$/.test(endpoint.hostname))
  ) {
    throw new Error('Memory Facts requires a local ACE API URL.');
  }
  endpoint.pathname = endpoint.pathname.replace(/\/+$/, '') + path;

  const response = await fetch(endpoint.href, {
    ...options,
    cache: 'no-store',
    redirect: 'error',
    referrerPolicy: 'no-referrer',
  }).catch((error) => {
    if (options.signal?.aborted) throw error;
    throw new Error('Could not reach the local ACE server.');
  });
  switch (response.status) {
    case 401:
      throw new Error('The local ACE connection needs a valid API key.');
    case 403:
      throw new Error('Memory Facts is only available directly from this computer.');
    case 409:
      throw new Error('The saved fact could not be revised. Refresh the list and check the current wording.');
    case 422:
      throw new Error('The local scanners could not clear this correction.');
    case 501:
      throw new Error('The configured memory backend does not support local fact review.');
    default:
      if (!response.ok) throw new Error('The local ACE server could not complete this request.');
  }
  return response;
}

export async function fetchMemoryFacts(signal: AbortSignal): Promise<MemoryFact[]> {
  const response = await requestFacts('/v1/memory/facts', { headers: authHeaders(), signal });
  const body = await response.json().catch(() => null);
  if (!Array.isArray(body?.facts) || !body.facts.every((fact: Partial<MemoryFact> | null) =>
    fact &&
    typeof fact.text === 'string' &&
    typeof fact.source === 'string' &&
    typeof fact.trust === 'string' &&
    typeof fact.superseded === 'boolean' &&
    typeof fact.trusted_for_recall === 'boolean'
  )) {
    throw new Error('ACE returned an invalid fact list.');
  }
  return body.facts;
}

export async function reviseMemoryFact(
  existingFact: string,
  correction: string,
  signal: AbortSignal,
): Promise<void> {
  await requestFacts('/v1/memory/facts/revise', {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ existing_fact: existingFact, correction }),
    signal,
  });
}
