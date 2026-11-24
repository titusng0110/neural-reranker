import fs from 'node:fs';

/**
 * Shared embedding client + cosine similarity helpers.
 *
 * The embedding dimension is centralized in EMBEDDING_DIM below. When you
 * switch embedding models, change it here and re-run embed_corpus.ts /
 * embed_training.ts so every stored vector uses the same dimension.
 */

/** Dimension produced by the embedding model served on port 9000. */
export const EMBEDDING_DIM = 1024;

const EMBEDDING_PORT = 9000;

/**
 * llama-server OpenAI-compatible embeddings endpoint.
 * Override with EMBEDDING_SERVER_URL when the server runs somewhere else.
 */
export const EMBEDDING_SERVER_URL =
  process.env.EMBEDDING_SERVER_URL || `http://127.0.0.1:${EMBEDDING_PORT}/v1/embeddings`;

/**
 * Task prefixes applied before embedding.
 *
 * Experiment: embed queries and documents raw, with no instruction/prompt.
 * Previous query prompt (harrier-oss-v1-0.6b web_search_query):
 *   'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: '
 */
export const QUERY_PREFIX = '';
export const DOCUMENT_PREFIX = '';

export type EmbeddingKind = 'query' | 'document';

interface LlamaEmbeddingResponse {
  data: { index: number; embedding: number[] }[];
}

export function formatEmbeddingInput(text: string, kind: EmbeddingKind): string {
  return (kind === 'query' ? QUERY_PREFIX : DOCUMENT_PREFIX) + text;
}

export function l2Normalize(vector: ArrayLike<number>): Float32Array {
  let sumSq = 0.0;
  for (let i = 0; i < vector.length; i++) {
    const v = vector[i]!;
    sumSq += v * v;
  }

  const invNorm = 1.0 / (Math.sqrt(sumSq) || 1e-12);
  const out = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i++) {
    out[i] = vector[i]! * invNorm;
  }
  return out;
}

export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) {
    throw new Error(`Cannot compare vectors of different dimensions: ${a.length} vs ${b.length}.`);
  }

  let dot = 0.0;
  let normA = 0.0;
  let normB = 0.0;

  for (let i = 0; i < a.length; i++) {
    const av = a[i]!;
    const bv = b[i]!;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }

  return dot / (Math.sqrt(normA * normB) || 1e-12);
}

/**
 * Cosine similarity between a query vector and one row of a packed matrix
 * ([vector0, vector1, ...]). Avoids allocating a slice per document.
 */
export function cosineSimilarityAt(
  query: ArrayLike<number>,
  matrix: ArrayLike<number>,
  offset: number
): number {
  const dim = query.length;
  if (offset < 0 || offset + dim > matrix.length) {
    throw new RangeError(
      `Vector at offset ${offset} (dim ${dim}) is outside matrix of length ${matrix.length}.`
    );
  }

  let dot = 0.0;
  let normQuery = 0.0;
  let normDoc = 0.0;

  for (let i = 0; i < dim; i++) {
    const q = query[i]!;
    const d = matrix[offset + i]!;
    dot += q * d;
    normQuery += q * q;
    normDoc += d * d;
  }

  return dot / (Math.sqrt(normQuery * normDoc) || 1e-12);
}

/**
 * Reads the WSL default gateway. In NAT mode that is the Windows host, so
 * llama-server is reachable there even when 127.0.0.1 only points at WSL.
 */
function getDefaultGateway(): string | null {
  try {
    const lines = fs.readFileSync('/proc/net/route', 'utf8').split('\n');
    for (let i = 1; i < lines.length; i++) {
      const fields = lines[i]!.trim().split(/\s+/);
      if (fields.length < 4) continue;

      const destination = fields[1]!;
      const gateway = fields[2]!;
      const flags = parseInt(fields[3]!, 16);

      if (destination === '00000000' && (flags & 0x2) !== 0) {
        const value = parseInt(gateway, 16);
        const a = value & 0xff;
        const b = (value >>> 8) & 0xff;
        const c = (value >>> 16) & 0xff;
        const d = (value >>> 24) & 0xff;
        return `${a}.${b}.${c}.${d}`;
      }
    }
  } catch {
    // Not Linux (or /proc is unavailable); only localhost will be tried.
  }
  return null;
}

let resolvedServerUrl: string | null = null;

function getServerUrls(): string[] {
  const envUrl = process.env.EMBEDDING_SERVER_URL;
  if (envUrl) return [envUrl];

  const urls = [EMBEDDING_SERVER_URL];
  const gateway = getDefaultGateway();
  if (gateway) {
    const gatewayUrl = `http://${gateway}:${EMBEDDING_PORT}/v1/embeddings`;
    if (!urls.includes(gatewayUrl)) urls.push(gatewayUrl);
  }
  return urls;
}

/**
 * Fetches one or more embeddings, validates them against EMBEDDING_DIM, and
 * returns L2-normalized Float32Arrays in the same order as the input texts.
 *
 * Endpoints are tried in order (localhost, then the WSL gateway) and the
 * first working one is cached for subsequent calls.
 */
export async function fetchEmbeddings(
  texts: string | string[],
  kind: EmbeddingKind = 'document'
): Promise<Float32Array[]> {
  const input = Array.isArray(texts) ? texts : [texts];
  if (input.length === 0) return [];

  const requestBody = JSON.stringify({
    input: input.map((text) => formatEmbeddingInput(text, kind))
  });
  const serverUrls = resolvedServerUrl ? [resolvedServerUrl] : getServerUrls();
  const failures: string[] = [];

  for (const serverUrl of serverUrls) {
    try {
      const res = await fetch(serverUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: requestBody
      });

      if (!res.ok) {
        failures.push(`${serverUrl}: HTTP ${res.status} ${await res.text()}`);
        continue;
      }

      const data = (await res.json()) as LlamaEmbeddingResponse;
      if (!data || !Array.isArray(data.data)) {
        failures.push(`${serverUrl}: invalid response ${JSON.stringify(data)}`);
        continue;
      }

      const items = [...data.data].sort((a, b) => a.index - b.index);
      if (items.length !== input.length) {
        failures.push(
          `${serverUrl}: returned ${items.length} vectors for ${input.length} inputs`
        );
        continue;
      }

      const vectors = items.map((item, i) => {
        const embedding = item.embedding;
        if (!Array.isArray(embedding)) {
          throw new Error(`Invalid embedding at index ${i}.`);
        }
        if (embedding.length !== EMBEDDING_DIM) {
          throw new Error(
            `Embedding dimension mismatch: server returned ${embedding.length}, ` +
            `but EMBEDDING_DIM is ${EMBEDDING_DIM}. Update EMBEDDING_DIM in ` +
            `cosine_similarity.ts, or start the matching model on port 9000.`
          );
        }
        return l2Normalize(embedding);
      });

      resolvedServerUrl = serverUrl;
      return vectors;
    } catch (error) {
      failures.push(
        `${serverUrl}: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  throw new Error(`Could not fetch embeddings from any endpoint:\n${failures.join('\n')}`);
}

/**
 * Convenience wrapper for a single embedding. Returns null (and warns) on
 * failure so callers like SearchEngine can gracefully fall back to lexical
 * scoring when the embedding server is unavailable.
 */
export async function fetchEmbedding(
  text: string,
  kind: EmbeddingKind = 'query'
): Promise<Float32Array | null> {
  try {
    const [embedding] = await fetchEmbeddings(text, kind);
    return embedding ?? null;
  } catch (error) {
    console.warn(
      `Embedding request failed (${EMBEDDING_SERVER_URL}): ` +
      (error instanceof Error ? error.message : String(error))
    );
    return null;
  }
}
