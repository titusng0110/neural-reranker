import fs from 'node:fs';
import { tokenize as tokenizeBPE, tokenizeWord as tokenizeWordBPE } from './bpe_tokenizer.ts';
import { tokenize as tokenizeNGram } from './ngram_tokenizer.ts';
import {
  scorePassage,
  computeVibiness,
  FUNCTION_WORDS
} from './scoring.ts';

// Preserves atomic section codes (109A, 14A.35, 8.08) and parentheticals ((1), (a))
const STATUTORY_SPLIT_REGEX = /(\([a-zA-Z0-9]+\)|[0-9]+[a-zA-Z]*(?:\.[0-9]+[a-zA-Z]*)?|[\p{L}\p{M}]+|[.!?\n]+|[\p{P}\p{S}])/gu;
const LLAMA_SERVER_URL = 'http://127.0.0.1:9000/v1/embeddings';

function extractQuerySymbolShingles(queryText: string): string[] {
  const rawSymbols = queryText.match(/\([a-zA-Z0-9]+\)|[0-9]+[a-zA-Z]*(?:\.[0-9]+[a-zA-Z]*)?/gu) || [];
  if (rawSymbols.length === 0) return [];

  const cleanSymbols = rawSymbols.map((s) => s.toLowerCase());
  const shingles = new Set<string>();

  for (let sIdx = 0; sIdx < cleanSymbols.length; sIdx++) {
    let compound = '';
    for (let k = 0; k < 3 && (sIdx + k) < cleanSymbols.length; k++) {
      compound += cleanSymbols[sIdx + k]!;
      shingles.add(compound);
    }
  }

  return Array.from(shingles);
}

export interface IndexedDocument {
  id: string;
  text: string;
  cleanText: string;
  words: string[];
  bpeTokenMap: Map<string, number[]>;
  bpeTfMap: Map<string, number>;
  docBpeSet: Set<string>;
  docSymbolSet: Set<string>;
  docNgramSet: Set<string>;
  punctuationIndices: Set<number>;
  docProseQuality: number;
}

export interface SearchResult {
  id: string;
  text: string;
  score: number;
  probability: number;
  denseSim: number;
  features: number[];
}

interface PreparedQuery {
  queryClean: string;
  queryWords: string[];
  querySymbols: string[];
  uniqueBpeTokens: string[];
  tokenToUniqueId: Map<string, number>;
  uniqueBpeCount: number;
  queryNgrams: string[];
  querySymbolShingles: string[];
  salientPhrases: string[];
  bpeQueryIdfSum: number;
  ngramQueryIdfSum: number;
  maxIdfToken: string;
  medianIdf: number;
  maxQueryBm25: number;
  queryVibiness: number;
  queryNumericRatio: number;
  fullCitationQuery: string;
  hasFullCitation: boolean;
}

interface ScoredCandidate {
  docIdx: number;
  score: number;
  denseSim: number;
}

interface WindowHit {
  pos: number;
  termId: number;
}

export interface SearchScratchpad {
  featureBuffer: Float32Array;
  windowHits: WindowHit[];
  termCounts: Int32Array;
  matchedBpeTokens: Set<string>;
}

export function createSearchScratchpad(maxTerms: number = 64): SearchScratchpad {
  return {
    featureBuffer: new Float32Array(18),
    windowHits: [],
    termCounts: new Int32Array(Math.max(64, maxTerms)),
    matchedBpeTokens: new Set<string>()
  };
}

export class SearchEngine {
  documents: IndexedDocument[] = [];
  docCount: number = 0;
  totalCorpusWords: number = 0;
  avgdl: number = 0.0;

  bpeInvertedIndex: Map<string, Set<number>> = new Map();
  bpeDocFreq: Map<string, number> = new Map();
  ngramDocFreq: Map<string, number> = new Map();

  denseEmbeddings: Float32Array | null = null;
  denseDim: number = 0;

  loadDenseEmbeddings(binPath: string): void {
    if (!fs.existsSync(binPath)) {
      console.warn(`Warning: Dense embedding file "${binPath}" not found.`);
      return;
    }

    const buf = fs.readFileSync(binPath);
    const count = buf.readUInt32LE(0);
    const dim = buf.readUInt32LE(4);

    this.denseEmbeddings = new Float32Array(buf.buffer, buf.byteOffset + 8, count * dim);
    this.denseDim = dim;
  }

  addDocument(id: string, text: string): void {
    const docIdx = this.documents.length;

    let punctCount = 0;
    let letterCount = 0;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i]!;
      if (/[\p{P}\p{S}]/u.test(ch)) punctCount++;
      else if (/[\p{L}\p{N}]/u.test(ch)) letterCount++;
    }
    const punctRatio = punctCount / Math.max(1, letterCount);
    const docProseQuality = 1.0 / (1.0 + 6.0 * punctRatio);

    const words: string[] = [];
    const rawSymbols: { token: string; pos: number }[] = [];
    const punctuationIndices = new Set<number>();
    let wordCounter = 0;

    let match: RegExpExecArray | null;
    STATUTORY_SPLIT_REGEX.lastIndex = 0;
    while ((match = STATUTORY_SPLIT_REGEX.exec(text)) !== null) {
      const token = match[0]!;

      if (/[.!?\n]/.test(token)) {
        if (wordCounter > 0) punctuationIndices.add(wordCounter - 1);
      } else if (token.startsWith('(') && token.endsWith(')')) {
        rawSymbols.push({ token, pos: wordCounter });
      } else if (/^[0-9]+[a-zA-Z]*(?:\.[0-9]+[a-zA-Z]*)?$/.test(token)) {
        rawSymbols.push({ token, pos: wordCounter });
        words.push(token);
        wordCounter++;
      } else if (/^[\p{L}\p{M}]+$/u.test(token)) {
        words.push(token);
        wordCounter++;
      }
    }

    const bpeTokenMap = new Map<string, number[]>();
    const bpeTfMap = new Map<string, number>();
    const docBpeSet = new Set<string>();

    for (let pos = 0; pos < words.length; pos++) {
      const subwords = tokenizeWordBPE(words[pos]!);
      for (let s = 0; s < subwords.length; s++) {
        const sub = subwords[s]!;
        docBpeSet.add(sub);
        bpeTfMap.set(sub, (bpeTfMap.get(sub) || 0) + 1);
        let list = bpeTokenMap.get(sub);
        if (!list) {
          list = [];
          bpeTokenMap.set(sub, list);
        }
        list.push(pos);
      }
    }

    const docSymbolSet = new Set<string>();
    for (let sIdx = 0; sIdx < rawSymbols.length; sIdx++) {
      const basePos = rawSymbols[sIdx]!.pos;
      const windowTokens: string[] = [];

      for (let k = 0; k < 3 && (sIdx + k) < rawSymbols.length; k++) {
        const item = rawSymbols[sIdx + k]!;
        if (Math.abs(item.pos - basePos) <= 1) {
          windowTokens.push(item.token.toLowerCase());
        } else {
          break;
        }
      }

      let compound = '';
      for (let w = 0; w < windowTokens.length; w++) {
        compound += windowTokens[w]!;
        docSymbolSet.add(compound);
        docBpeSet.add(compound);
        bpeTfMap.set(compound, (bpeTfMap.get(compound) || 0) + 1);

        let list = bpeTokenMap.get(compound);
        if (!list) {
          list = [];
          bpeTokenMap.set(compound, list);
        }
        list.push(basePos);
      }
    }

    const docNgramSet = new Set<string>(tokenizeNGram(text));

    for (const token of docBpeSet) {
      this.bpeDocFreq.set(token, (this.bpeDocFreq.get(token) || 0) + 1);
      let posting = this.bpeInvertedIndex.get(token);
      if (!posting) {
        posting = new Set();
        this.bpeInvertedIndex.set(token, posting);
      }
      posting.add(docIdx);
    }

    for (const gram of docNgramSet) {
      this.ngramDocFreq.set(gram, (this.ngramDocFreq.get(gram) || 0) + 1);
    }

    this.totalCorpusWords += words.length;
    this.documents.push({
      id,
      text,
      cleanText: text.toLowerCase(),
      words,
      bpeTokenMap,
      bpeTfMap,
      docBpeSet,
      docSymbolSet,
      docNgramSet,
      punctuationIndices,
      docProseQuality
    });

    this.docCount = this.documents.length;
    this.avgdl = this.totalCorpusWords / this.docCount;
  }

  getIdf(token: string, isNgram: boolean = false): number {
    const freqTable = isNgram ? this.ngramDocFreq : this.bpeDocFreq;
    const df = freqTable.get(token) || 0;
    return Math.log(1.0 + (this.docCount - df + 0.5) / (df + 0.5));
  }

  prepareQuery(queryText: string): PreparedQuery {
    const queryClean = queryText.toLowerCase().trim();

    const rawSymbols = queryClean.match(/\([a-zA-Z0-9]+\)|[0-9]+[a-zA-Z]*(?:\.[0-9]+[a-zA-Z]*)?/gu) || [];
    const querySymbols = rawSymbols.map((s) => s.toLowerCase());
    const queryWords = queryClean.match(/[\p{L}\p{M}]+/gu) || [];

    const rawBpeTokens = tokenizeBPE(queryText);
    const tokenToUniqueId = new Map<string, number>();
    const uniqueBpeTokens: string[] = [];

    for (let i = 0; i < rawBpeTokens.length; i++) {
      const t = rawBpeTokens[i]!;
      if (!tokenToUniqueId.has(t)) {
        tokenToUniqueId.set(t, uniqueBpeTokens.length);
        uniqueBpeTokens.push(t);
      }
    }

    const queryNgrams = tokenizeNGram(queryText);

    // Feature #11: Require at least one content word in salient n-gram phrases
    const salientPhrases: string[] = [];
    for (let i = 0; i < queryWords.length - 1; i++) {
      const w1 = queryWords[i]!;
      const w2 = queryWords[i + 1]!;
      if (!FUNCTION_WORDS.has(w1) || !FUNCTION_WORDS.has(w2)) {
        salientPhrases.push(w1 + ' ' + w2);
        if (i < queryWords.length - 2) {
          salientPhrases.push(w1 + ' ' + w2 + ' ' + queryWords[i + 2]!);
        }
      }
    }

    let maxIdfToken = uniqueBpeTokens[0] || '';
    let maxIdfVal = -1;
    let bpeQueryIdfSum = 0;
    const idfList: number[] = [];

    for (let i = 0; i < uniqueBpeTokens.length; i++) {
      const t = uniqueBpeTokens[i]!;
      const idf = this.getIdf(t, false);
      bpeQueryIdfSum += idf;
      idfList.push(idf);
      if (idf > maxIdfVal) {
        maxIdfVal = idf;
        maxIdfToken = t;
      }
    }

    idfList.sort((a, b) => a - b);
    const medianIdf = idfList.length > 0 ? idfList[Math.floor(idfList.length / 2)]! : 0.0;

    let ngramQueryIdfSum = 0;
    for (let i = 0; i < queryNgrams.length; i++) {
      ngramQueryIdfSum += this.getIdf(queryNgrams[i]!, true);
    }

    const k1 = 1.2;
    const maxQueryBm25 = bpeQueryIdfSum * (k1 + 1);
    const queryVibiness = computeVibiness(queryWords);
    const totalTokens = queryWords.length + querySymbols.length;
    const queryNumericRatio = totalTokens > 0 ? querySymbols.length / totalTokens : 0.0;

    const fullCitationQuery = querySymbols.join('');
    const hasFullCitation = fullCitationQuery.length >= 2;
    const querySymbolShingles = extractQuerySymbolShingles(queryText);

    return {
      queryClean,
      queryWords,
      querySymbols,
      uniqueBpeTokens,
      tokenToUniqueId,
      uniqueBpeCount: uniqueBpeTokens.length,
      queryNgrams,
      querySymbolShingles,
      salientPhrases,
      bpeQueryIdfSum,
      ngramQueryIdfSum,
      maxIdfToken,
      medianIdf,
      maxQueryBm25,
      queryVibiness,
      queryNumericRatio,
      fullCitationQuery,
      hasFullCitation
    };
  }

  // Pure in-place calculation: 0 heap allocations
  computeFeaturesInto(
    prep: PreparedQuery,
    doc: IndexedDocument,
    docIdx: number,
    queryEmbedding: Float32Array | null,
    precomputedDenseSim: number | null,
    scratch: SearchScratchpad
  ): void {
    const { featureBuffer, windowHits, termCounts, matchedBpeTokens } = scratch;

    windowHits.length = 0;
    matchedBpeTokens.clear();

    const {
      queryClean, querySymbols, uniqueBpeTokens, uniqueBpeCount,
      queryNgrams, salientPhrases, bpeQueryIdfSum, ngramQueryIdfSum,
      maxIdfToken, medianIdf, maxQueryBm25, queryVibiness,
      queryNumericRatio, fullCitationQuery, hasFullCitation
    } = prep;

    const k1 = 1.2;
    const b = 0.75;
    const docLenNorm = 1.0 - b + b * (doc.words.length / (this.avgdl || 1.0));

    let bpeHitIdfSum = 0.0;
    let rawBm25Score = 0.0;
    let totalHitCount = 0;
    let firstSalientPos = -1;

    for (let termId = 0; termId < uniqueBpeTokens.length; termId++) {
      const t = uniqueBpeTokens[termId]!;
      if (doc.docBpeSet.has(t)) {
        const idf = this.getIdf(t, false);
        bpeHitIdfSum += idf;
        matchedBpeTokens.add(t);

        const positions = doc.bpeTokenMap.get(t);
        if (positions) {
          const isContent = !FUNCTION_WORDS.has(t) && idf >= medianIdf;
          for (let p = 0; p < positions.length; p++) {
            const pos = positions[p]!;
            windowHits.push({ pos, termId });

            if (isContent && (firstSalientPos === -1 || pos < firstSalientPos)) {
              firstSalientPos = pos;
            }
          }
        }

        const tf = doc.bpeTfMap.get(t) || 0;
        totalHitCount += tf;
        rawBm25Score += idf * ((tf * (k1 + 1)) / (tf + k1 * docLenNorm));
      }
    }

    let ngramHitIdfSum = 0.0;
    for (let i = 0; i < queryNgrams.length; i++) {
      const g = queryNgrams[i]!;
      if (doc.docNgramSet.has(g)) {
        ngramHitIdfSum += this.getIdf(g, true);
      }
    }

    // =========================================================================
    // FEATURE #3 & #4: MINIMUM COVERING WINDOW (BUG-FREE TWO-POINTER SCAN)
    // =========================================================================
    let minCoveringSpan = Infinity;
    let bestL = 0;
    let bestR = 0;
    const K = matchedBpeTokens.size;

    if (K >= 2 && windowHits.length >= K) {
      windowHits.sort((a, b) => a.pos - b.pos);
      termCounts.fill(0);

      let distinctCount = 0;
      let left = 0;

      for (let right = 0; right < windowHits.length; right++) {
        const tRight = windowHits[right]!.termId;
        const cRight = termCounts[tRight]!;
        if (cRight === 0) distinctCount++;
        termCounts[tRight] = cRight + 1;

        while (distinctCount === K) {
          const span = windowHits[right]!.pos - windowHits[left]!.pos + 1;
          if (span < minCoveringSpan) {
            minCoveringSpan = span;
            bestL = windowHits[left]!.pos;
            bestR = windowHits[right]!.pos;
          }

          const tLeft = windowHits[left]!.termId;
          const cLeft = termCounts[tLeft]!;
          termCounts[tLeft] = cLeft - 1;
          if (cLeft === 1) distinctCount--;
          left++;
        }
      }
    }

    let sentenceTerminals = 0;
    if (minCoveringSpan !== Infinity && bestR > bestL) {
      for (let p = bestL; p < bestR; p++) {
        if (doc.punctuationIndices.has(p)) sentenceTerminals++;
      }
    }

    // Feature #6: Order Monotonicity
    let isOrdered = false;
    if (matchedBpeTokens.size >= 2) {
      let lastPos = -1;
      let ordered = true;
      for (let i = 0; i < uniqueBpeTokens.length; i++) {
        const t = uniqueBpeTokens[i]!;
        const positions = doc.bpeTokenMap.get(t);
        if (positions && positions.length > 0) {
          let found = false;
          for (let k = 0; k < positions.length; k++) {
            const p = positions[k]!;
            if (p > lastPos) {
              lastPos = p;
              found = true;
              break;
            }
          }
          if (!found) { ordered = false; break; }
        }
      }
      isOrdered = ordered;
    }

    // Citation and Symbol Coverage
    let symbolMatchedCount = 0;
    for (let i = 0; i < querySymbols.length; i++) {
      if (doc.docSymbolSet.has(querySymbols[i]!)) symbolMatchedCount++;
    }

    const hasSalientSymbol = querySymbols.length > 0 && doc.docSymbolSet.has(querySymbols[0]!);
    const hasExactCitation = hasFullCitation && doc.docSymbolSet.has(fullCitationQuery);

    // Feature #11: Salient Collocation Matching
    let phraseMatchScore = 0.0;
    if (queryClean.length > 0 && doc.cleanText.includes(queryClean)) {
      phraseMatchScore = 1.0;
    } else if (salientPhrases.length > 0) {
      let matchedCount = 0;
      for (let i = 0; i < salientPhrases.length; i++) {
        if (doc.cleanText.includes(salientPhrases[i]!)) matchedCount++;
      }
      phraseMatchScore = Math.min(1.0, matchedCount / Math.min(3, salientPhrases.length));
    }

    // Dense Latent Similarity
    let denseCosineSimilarity = 0.0;
    if (precomputedDenseSim !== null) {
      denseCosineSimilarity = precomputedDenseSim;
    } else if (queryEmbedding && this.denseEmbeddings) {
      const dim = this.denseDim;
      const offset = docIdx * dim;
      let dot = 0.0;
      for (let i = 0; i < dim; i++) {
        dot += queryEmbedding[i]! * this.denseEmbeddings[offset + i]!;
      }
      denseCosineSimilarity = dot;
    }

    // Direct output writes to Float32Array (Eliminated 24-property object allocation)
    featureBuffer[0] = bpeQueryIdfSum > 0 ? Math.min(1.0, bpeHitIdfSum / bpeQueryIdfSum) : 0.0;
    featureBuffer[1] = ngramQueryIdfSum > 0 ? Math.min(1.0, ngramHitIdfSum / ngramQueryIdfSum) : 0.0;
    featureBuffer[2] = uniqueBpeCount > 0 ? Math.min(1.0, matchedBpeTokens.size / uniqueBpeCount) : 0.0;
    featureBuffer[3] = (minCoveringSpan !== Infinity && minCoveringSpan >= K && K >= 2)
      ? Math.min(1.0, K / minCoveringSpan)
      : 0.0;
    featureBuffer[4] = (minCoveringSpan !== Infinity && K >= 2) ? 1.0 / (1.0 + sentenceTerminals) : 0.0;
    featureBuffer[5] = firstSalientPos >= 0 ? 1.0 / (1.0 + 0.05 * firstSalientPos) : 0.0;
    featureBuffer[6] = isOrdered ? 1.0 : 0.0;
    featureBuffer[7] = Math.max(0.0, Math.min(1.0, denseCosineSimilarity));
    featureBuffer[8] = maxQueryBm25 > 0 ? Math.min(1.0, rawBm25Score / maxQueryBm25) : 0.0;
    featureBuffer[9] = doc.docBpeSet.has(maxIdfToken) ? 1.0 : 0.0;
    featureBuffer[10] = doc.words.length > 0 ? Math.min(1.0, totalHitCount / doc.words.length) : 0.0;
    featureBuffer[11] = phraseMatchScore;
    featureBuffer[12] = queryVibiness;
    featureBuffer[13] = queryNumericRatio;
    featureBuffer[14] = doc.docProseQuality;
    featureBuffer[15] = querySymbols.length > 0 ? Math.min(1.0, symbolMatchedCount / querySymbols.length) : 0.0;
    featureBuffer[16] = hasSalientSymbol ? 1.0 : 0.0;
    featureBuffer[17] = hasExactCitation ? 1.0 : 0.0;
  }

  // Safe wrapper for training scripts
  computeFeatures(
    queryText: string,
    docIdx: number,
    queryEmbedding: Float32Array | null = null,
    precomputedDenseSim: number | null = null
  ): Float32Array {
    const prep = this.prepareQuery(queryText);
    const doc = this.documents[docIdx];
    if (!doc) return new Float32Array(18);

    const scratch = createSearchScratchpad(prep.uniqueBpeCount);
    this.computeFeaturesInto(prep, doc, docIdx, queryEmbedding, precomputedDenseSim, scratch);

    const copy = new Float32Array(18);
    copy.set(scratch.featureBuffer);
    return copy;
  }

  async fetchQueryEmbedding(queryText: string): Promise<Float32Array | null> {
    try {
      const res = await fetch(LLAMA_SERVER_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: [`Query: ${queryText}`] }),
      });
      if (!res.ok) return null;
      const data = (await res.json()) as { data: { embedding: number[] }[] };
      const raw = data.data[0]!.embedding;

      let sum = 0.0;
      for (let i = 0; i < raw.length; i++) sum += raw[i]! * raw[i]!;
      const norm = Math.sqrt(sum) || 1e-12;
      const normalized = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i++) normalized[i] = raw[i]! / norm;

      return normalized;
    } catch {
      return null;
    }
  }

    async search(
    queryText: string,
    topK: number = 5,
    queryEmbedding: Float32Array | null = null
  ): Promise<SearchResult[]> {
    if (!queryText || this.docCount === 0) return [];

    // 1. Fetch dense query embedding if available and not supplied
    if (!queryEmbedding && this.denseEmbeddings) {
      queryEmbedding = await this.fetchQueryEmbedding(queryText);
    }

    // 2. Prepare query metrics (IDFs, tokens, vibiness, symbols)
    const prep = this.prepareQuery(queryText);

    // 3. Allocate scratchpad ONCE per query (Zero-GC, thread-reentrant)
    const scratch = createSearchScratchpad(prep.uniqueBpeCount);

    // Pre-allocate candidates array for the entire population
    const candidates: ScoredCandidate[] = new Array(this.docCount);

    // =========================================================================
    // EXHAUSTIVE FULL-POPULATION SCORING (Zero-GC Hot Loop)
    // =========================================================================
    for (let docIdx = 0; docIdx < this.docCount; docIdx++) {
      const doc = this.documents[docIdx]!;

      // In-place feature extraction directly into scratchpad Float32Array
      this.computeFeaturesInto(
        prep,
        doc,
        docIdx,
        queryEmbedding,
        null,
        scratch
      );

      const f = scratch.featureBuffer;
      const score = scorePassage(f);

      candidates[docIdx] = {
        docIdx,
        score,
        denseSim: f[7]!
      };
    }

    // Sort the entire population by SwiGLU score descending
    candidates.sort((a, b) => b.score - a.score);
    const finalCount = Math.min(topK, candidates.length);

    // Reconstruct features only for the top-K winners
    const results: SearchResult[] = new Array(finalCount);
    for (let i = 0; i < finalCount; i++) {
      const c = candidates[i]!;
      const doc = this.documents[c.docIdx]!;

      this.computeFeaturesInto(
        prep,
        doc,
        c.docIdx,
        queryEmbedding,
        null,
        scratch
      );

      results[i] = {
        id: doc.id,
        text: doc.text,
        score: c.score,
        probability: c.score,
        denseSim: c.denseSim,
        features: Array.from(scratch.featureBuffer)
      };
    }

    return results;
  }

}
