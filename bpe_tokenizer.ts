import fs from 'node:fs';

const BINARY_PATH = './bpe_vocab.bin';

if (!fs.existsSync(BINARY_PATH)) {
  throw new Error(`Cannot find "${BINARY_PATH}". Run "tsx build_vocab.ts" first.`);
}

const buffer = fs.readFileSync(BINARY_PATH);

// Fast unpack on startup
const tokenCount = buffer.readUInt32LE(0);
const RANKS = new Map<string, number>();

let offset = 4;
for (let i = 0; i < tokenCount; i++) {
  const rank = buffer.readUIntBE(offset, 3);
  const len = buffer[offset + 3]!;
  offset += 4;

  const str = buffer.toString('utf-8', offset, offset + len);
  offset += len;

  RANKS.set(str, rank);
}

// ============================================================================
// ALLOCATION-FREE SCRATCHPADS (CONCURRENCY-SAFE FACTORY)
// ============================================================================
export const MAX_WORD_LEN = 256;

export interface BPEScratchpad {
  maxWordLen: number;
  prev: Int32Array;
  next: Int32Array;
  parts: (string | null)[];
  wordRegex: RegExp;
}

export function createBPEScratchpad(maxWordLen: number = MAX_WORD_LEN): BPEScratchpad {
  return {
    maxWordLen,
    prev: new Int32Array(maxWordLen),
    next: new Int32Array(maxWordLen),
    parts: new Array<string | null>(maxWordLen).fill(null),
    wordRegex: /[\p{L}\p{M}\p{N}]+/gu
  };
}

const DEFAULT_SCRATCHPAD: BPEScratchpad = createBPEScratchpad();

/**
 * Tokenizes a single word directly into an existing Set or Array.
 * Zero array allocations. Zero TypedArray allocations.
 */
function tokenizeWordInto(
  word: string,
  outCollection: Set<string> | string[],
  ctx: BPEScratchpad = DEFAULT_SCRATCHPAD
): void {
  const { maxWordLen, prev, next, parts } = ctx;
  let n = 0;

  for (const ch of word) {
    if (n >= maxWordLen) break;
    parts[n] = ch;
    prev[n] = n - 1;
    next[n] = n + 1;
    n++;
  }

  if (n === 0) return;
  if (n === 1) {
    if ('add' in outCollection) {
      outCollection.add(word);
    } else {
      outCollection.push(word);
    }
    parts[0] = null;
    return;
  }
  next[n - 1] = -1;

  // BPE Merge Loop using O(1) linked-list pointer updates
  while (true) {
    let minRank = Infinity;
    let bestI = -1;

    let i = 0;
    while (i !== -1 && next[i] !== -1) {
      const j = next[i]!;
      const pair = (parts[i] as string) + (parts[j] as string);
      const rank = RANKS.get(pair);

      if (rank !== undefined && rank < minRank) {
        minRank = rank;
        bestI = i;
      }
      i = j;
    }

    if (bestI === -1) break;

    // Merge bestI and next[bestI] in-place
    const r = next[bestI]!;
    parts[bestI] = (parts[bestI] as string) + (parts[r] as string);

    // Unlink node 'r'
    const nextR = next[r]!;
    next[bestI] = nextR;
    if (nextR !== -1) {
      prev[nextR] = bestI;
    }
  }

  // Stream output directly into destination container & clean up references
  let curr = 0;
  while (curr !== -1) {
    const val = parts[curr] as string;
    if ('add' in outCollection) {
      outCollection.add(val);
    } else {
      outCollection.push(val);
    }
    parts[curr] = null;
    curr = next[curr]!;
  }
}

export function tokenizeWord(word: string, ctx: BPEScratchpad = DEFAULT_SCRATCHPAD): string[] {
  const result: string[] = [];
  tokenizeWordInto(word.normalize('NFC').toLowerCase().trim(), result, ctx);
  return result;
}

export function tokenize(text: string, ctx: BPEScratchpad = DEFAULT_SCRATCHPAD): string[] {
  if (!text) return [];

  const clean = /^[\x00-\x7F]*$/.test(text)
    ? text.toLowerCase().trim()
    : text.normalize('NFC').toLowerCase().trim();

  const tokenSet = new Set<string>();
  const regex = ctx.wordRegex;
  regex.lastIndex = 0;

  let match: RegExpExecArray | null;
  while ((match = regex.exec(clean)) !== null) {
    tokenizeWordInto(match[0], tokenSet, ctx);
  }

  return Array.from(tokenSet);
}
