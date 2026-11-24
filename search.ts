import fs from 'node:fs';
import { tokenize as tokenizeBPE, tokenizeWord as tokenizeWordBPE } from './bpe_tokenizer.ts';
import { tokenize as tokenizeNGram } from './ngram_tokenizer.ts';
import {
  scorePassage,
  computeVibiness,
  FUNCTION_WORDS
} from './scoring.ts';
import { NUM_FEATURES } from './weights_loader.ts';

const LLAMA_SERVER_URL = 'http://127.0.0.1:9000/v1/embeddings';

/**
 * Preprocessing rule:
 * 1. 3 or more spaces become 2 spaces.
 * 2. 3 or more newlines become 2 newlines.
 */
export function preprocessText(text: string): string {
  if (!text) return '';
  return text
    .replace(/[^\S\r\n]{3,}/g, '  ')
    .replace(/(?:\r?\n){3,}/g, '\n\n');
}

// Splits tokens preserving atomic statutory symbols & parentheticals
const BASE_TOKEN_REGEX = /(\([a-zA-Z0-9]+\)|[0-9]+[a-zA-Z]*(?:\.[0-9]+[a-zA-Z]*)?|[\p{L}\p{M}\p{N}]+|[.!?\n]+|[^\s\p{L}\p{M}\p{N}])/gu;
const PURE_LANGUAGE_REGEX = /^[\p{L}\p{M}]+$/u;
const STATUTORY_SYMBOL_REGEX = /\([a-zA-Z0-9]+\)|[0-9]+[a-zA-Z]*(?:\.[0-9]+[a-zA-Z]*)?/gu;

export interface IndexedDocument {
  id: string;
  rawText: string;
  cleanText: string;
  wordsT1: string[];
  docLengthT1: number;
  docLengthT2: number;
  docLengthT3: number;

  // Tokenizer 1 (Universal BPE)
  bpeTfMap1: Map<string, number>;
  bpePosMap1: Map<string, number[]>;
  docBpeSet1: Set<string>;

  // Tokenizer 2 (Symbol & Alphanumerics BPE)
  bpeTfMap2: Map<string, number>;
  docBpeSet2: Set<string>;
  docSymbolShingles2: Set<string>;

  // Tokenizer 3 (Language Multilingual N-Gram)
  ngramTfMap3: Map<string, number>;
  docNgramSet3: Set<string>;

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
  rawQuery: string;
  cleanQuery: string;
  queryWords: string[];
  querySymbols: string[];

  // Tokenizer 1
  uniqueBpe1: string[];
  tokenToId1: Map<string, number>;
  maxIdfToken1: string;
  medianIdf1: number;
  maxQueryBm25_1: number;

  // Tokenizer 2
  uniqueBpe2: string[];
  maxIdfToken2: string;
  maxQueryBm25_2: number;
  fullCitationQuery2: string;
  hasFullCitation2: boolean;

  // Tokenizer 3
  uniqueNgrams3: string[];
  maxQueryBm25_3: number;

  salientPhrases: string[];
  queryVibiness: number;
  queryNumericRatio: number;
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
  matchedTokens1: Set<string>;
}

export function createSearchScratchpad(maxTerms: number = 64): SearchScratchpad {
  return {
    featureBuffer: new Float32Array(NUM_FEATURES),
    windowHits: [],
    termCounts: new Int32Array(Math.max(64, maxTerms)),
    matchedTokens1: new Set<string>()
  };
}

export class SearchEngine {
  documents: IndexedDocument[] = [];
  docCount: number = 0;

  totalWordsT1: number = 0;
  totalWordsT2: number = 0;
  totalWordsT3: number = 0;

  avgdlT1: number = 0.0;
  avgdlT2: number = 0.0;
  avgdlT3: number = 0.0;

  docFreq1: Map<string, number> = new Map();
  docFreq2: Map<string, number> = new Map();
  docFreq3: Map<string, number> = new Map();

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
    const preprocessed = preprocessText(text);
    const docIdx = this.documents.length;

    // Prose quality computation
    let punctCount = 0;
    let letterCount = 0;
    for (let i = 0; i < preprocessed.length; i++) {
      const ch = preprocessed[i]!;
      if (/[\p{P}\p{S}]/u.test(ch)) punctCount++;
      else if (/[\p{L}\p{N}]/u.test(ch)) letterCount++;
    }
    const punctRatio = punctCount / Math.max(1, letterCount);
    const docProseQuality = 1.0 / (1.0 + 6.0 * punctRatio);

    const wordsT1: string[] = [];
    const punctuationIndices = new Set<number>();
    const rawSymbolTokens: { token: string; pos: number }[] = [];

    // Storage for Tokenizer 2 (mixed alphanumerics, symbols, codes)
    const tokensT2: string[] = [];
    // Storage for Tokenizer 3 (pure language words across all scripts)
    const tokensT3: string[] = [];

    let match: RegExpExecArray | null;
    BASE_TOKEN_REGEX.lastIndex = 0;
    let wordCounter = 0;

    while ((match = BASE_TOKEN_REGEX.exec(preprocessed)) !== null) {
      const token = match[0]!;

      if (/[.!?\n]/.test(token)) {
        if (wordCounter > 0) punctuationIndices.add(wordCounter - 1);
        continue;
      }

      const isPureLang = PURE_LANGUAGE_REGEX.test(token);

      // Tokenizer 1 accepts all word tokens
      wordsT1.push(token);

      // Tokenizer 2: ONLY remove if token is pure language. Keep if it has numbers/symbols/mixed alphanumerics
      if (!isPureLang) {
        tokensT2.push(token);
      }

      // Tokenizer 3: Language only (includes non-English alphabets/scripts)
      if (/[\p{L}\p{M}]/u.test(token) && !/^[0-9]+$/.test(token)) {
        tokensT3.push(token);
      }

      if (token.startsWith('(') && token.endsWith(')')) {
        rawSymbolTokens.push({ token, pos: wordCounter });
      } else if (/^[0-9]+[a-zA-Z]*(?:\.[0-9]+[a-zA-Z]*)?$/.test(token)) {
        rawSymbolTokens.push({ token, pos: wordCounter });
      }

      wordCounter++;
    }

    // 1. TOKENIZER 1: Full Universal BPE
    const bpeTfMap1 = new Map<string, number>();
    const bpePosMap1 = new Map<string, number[]>();
    const docBpeSet1 = new Set<string>();

    for (let pos = 0; pos < wordsT1.length; pos++) {
      const subwords = tokenizeWordBPE(wordsT1[pos]!);
      for (let s = 0; s < subwords.length; s++) {
        const sub = subwords[s]!;
        docBpeSet1.add(sub);
        bpeTfMap1.set(sub, (bpeTfMap1.get(sub) || 0) + 1);
        let list = bpePosMap1.get(sub);
        if (!list) {
          list = [];
          bpePosMap1.set(sub, list);
        }
        list.push(pos);
      }
    }

    // 2. TOKENIZER 2: Symbol & Alphanumeric BPE
    const bpeTfMap2 = new Map<string, number>();
    const docBpeSet2 = new Set<string>();
    const docSymbolShingles2 = new Set<string>();

    for (let i = 0; i < tokensT2.length; i++) {
      const subwords = tokenizeWordBPE(tokensT2[i]!);
      for (let s = 0; s < subwords.length; s++) {
        const sub = subwords[s]!;
        docBpeSet2.add(sub);
        bpeTfMap2.set(sub, (bpeTfMap2.get(sub) || 0) + 1);
      }
    }

    // Build contiguous statutory shingles (e.g. 109a(1)(a))
    for (let sIdx = 0; sIdx < rawSymbolTokens.length; sIdx++) {
      const basePos = rawSymbolTokens[sIdx]!.pos;
      const windowTokens: string[] = [];

      for (let k = 0; k < 3 && (sIdx + k) < rawSymbolTokens.length; k++) {
        const item = rawSymbolTokens[sIdx + k]!;
        if (Math.abs(item.pos - basePos) <= 1) {
          windowTokens.push(item.token.toLowerCase());
        } else {
          break;
        }
      }

      let compound = '';
      for (let w = 0; w < windowTokens.length; w++) {
        compound += windowTokens[w]!;
        docSymbolShingles2.add(compound);
        docBpeSet2.add(compound);
        bpeTfMap2.set(compound, (bpeTfMap2.get(compound) || 0) + 1);
      }
    }

    // 3. TOKENIZER 3: Language Multilingual N-Gram
    const ngramTfMap3 = new Map<string, number>();
    const docNgramSet3 = new Set<string>();

    for (let i = 0; i < tokensT3.length; i++) {
      const ngrams = tokenizeNGram(tokensT3[i]!);
      for (let g = 0; g < ngrams.length; g++) {
        const gram = ngrams[g]!;
        docNgramSet3.add(gram);
        ngramTfMap3.set(gram, (ngramTfMap3.get(gram) || 0) + 1);
      }
    }

    // Update global document frequencies
    for (const token of docBpeSet1) {
      this.docFreq1.set(token, (this.docFreq1.get(token) || 0) + 1);
    }
    for (const token of docBpeSet2) {
      this.docFreq2.set(token, (this.docFreq2.get(token) || 0) + 1);
    }
    for (const gram of docNgramSet3) {
      this.docFreq3.set(gram, (this.docFreq3.get(gram) || 0) + 1);
    }

    const docLengthT1 = Math.max(1, wordsT1.length);
    const docLengthT2 = Math.max(1, tokensT2.length);
    const docLengthT3 = Math.max(1, docNgramSet3.size);

    this.totalWordsT1 += docLengthT1;
    this.totalWordsT2 += docLengthT2;
    this.totalWordsT3 += docLengthT3;

    this.documents.push({
      id,
      rawText: text,
      cleanText: preprocessed.toLowerCase(),
      wordsT1,
      docLengthT1,
      docLengthT2,
      docLengthT3,
      bpeTfMap1,
      bpePosMap1,
      docBpeSet1,
      bpeTfMap2,
      docBpeSet2,
      docSymbolShingles2,
      ngramTfMap3,
      docNgramSet3,
      punctuationIndices,
      docProseQuality
    });

    this.docCount = this.documents.length;
    this.avgdlT1 = this.totalWordsT1 / this.docCount;
    this.avgdlT2 = this.totalWordsT2 / this.docCount;
    this.avgdlT3 = this.totalWordsT3 / this.docCount;
  }

  getIdf(token: string, tokenizerType: 1 | 2 | 3): number {
    const table =
      tokenizerType === 1 ? this.docFreq1 :
      tokenizerType === 2 ? this.docFreq2 : this.docFreq3;

    const df = table.get(token) || 0;
    return Math.log(1.0 + (this.docCount - df + 0.5) / (df + 0.5));
  }

  prepareQuery(queryText: string): PreparedQuery {
    const preprocessed = preprocessText(queryText);
    const cleanQuery = preprocessed.toLowerCase().trim();

    const rawSymbols = cleanQuery.match(STATUTORY_SYMBOL_REGEX) || [];
    const querySymbols = rawSymbols.map((s) => s.toLowerCase());
    const queryWords = cleanQuery.match(/[\p{L}\p{M}]+/gu) || [];

    // Split base tokens for query
    const baseTokens: string[] = [];
    let match: RegExpExecArray | null;
    BASE_TOKEN_REGEX.lastIndex = 0;
    while ((match = BASE_TOKEN_REGEX.exec(cleanQuery)) !== null) {
      const tok = match[0]!;
      if (!/[.!?\n]/.test(tok)) baseTokens.push(tok);
    }

    // 1. Query Tokenizer 1 (Universal BPE)
    const rawBpe1 = tokenizeBPE(cleanQuery);
    const uniqueBpe1: string[] = [];
    const tokenToId1 = new Map<string, number>();

    for (let i = 0; i < rawBpe1.length; i++) {
      const t = rawBpe1[i]!;
      if (!tokenToId1.has(t)) {
        tokenToId1.set(t, uniqueBpe1.length);
        uniqueBpe1.push(t);
      }
    }

    let maxIdfToken1 = uniqueBpe1[0] || '';
    let maxIdfVal1 = -1;
    let bpeQueryIdfSum1 = 0;
    const idfList1: number[] = [];

    for (let i = 0; i < uniqueBpe1.length; i++) {
      const t = uniqueBpe1[i]!;
      const idf = this.getIdf(t, 1);
      bpeQueryIdfSum1 += idf;
      idfList1.push(idf);
      if (idf > maxIdfVal1) {
        maxIdfVal1 = idf;
        maxIdfToken1 = t;
      }
    }

    idfList1.sort((a, b) => a - b);
    const medianIdf1 = idfList1.length > 0 ? idfList1[Math.floor(idfList1.length / 2)]! : 0.0;
    const k1 = 1.2;
    const maxQueryBm25_1 = bpeQueryIdfSum1 * (k1 + 1);

    // 2. Query Tokenizer 2 (Symbol & Alphanumerics BPE)
    const queryTokensT2 = baseTokens.filter((t) => !PURE_LANGUAGE_REGEX.test(t));
    const uniqueBpe2Set = new Set<string>();

    for (let i = 0; i < queryTokensT2.length; i++) {
      const subwords = tokenizeWordBPE(queryTokensT2[i]!);
      for (let s = 0; s < subwords.length; s++) {
        uniqueBpe2Set.add(subwords[s]!);
      }
    }

    const uniqueBpe2 = Array.from(uniqueBpe2Set);
    let maxIdfToken2 = uniqueBpe2[0] || '';
    let maxIdfVal2 = -1;
    let bpeQueryIdfSum2 = 0;

    for (let i = 0; i < uniqueBpe2.length; i++) {
      const t = uniqueBpe2[i]!;
      const idf = this.getIdf(t, 2);
      bpeQueryIdfSum2 += idf;
      if (idf > maxIdfVal2) {
        maxIdfVal2 = idf;
        maxIdfToken2 = t;
      }
    }
    const maxQueryBm25_2 = bpeQueryIdfSum2 * (k1 + 1);
    const fullCitationQuery2 = querySymbols.join('');
    const hasFullCitation2 = fullCitationQuery2.length >= 2;

    // 3. Query Tokenizer 3 (Language Multilingual N-Gram)
    const queryTokensT3 = baseTokens.filter((t) => /[\p{L}\p{M}]/u.test(t) && !/^[0-9]+$/.test(t));
    const uniqueNgrams3Set = new Set<string>();

    for (let i = 0; i < queryTokensT3.length; i++) {
      const ngrams = tokenizeNGram(queryTokensT3[i]!);
      for (let g = 0; g < ngrams.length; g++) {
        uniqueNgrams3Set.add(ngrams[g]!);
      }
    }

    const uniqueNgrams3 = Array.from(uniqueNgrams3Set);
    let ngramQueryIdfSum3 = 0;
    for (let i = 0; i < uniqueNgrams3.length; i++) {
      ngramQueryIdfSum3 += this.getIdf(uniqueNgrams3[i]!, 3);
    }
    const maxQueryBm25_3 = ngramQueryIdfSum3 * (k1 + 1);

    // Salient collocations (Tokenizer 1 exact match)
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

    const queryVibiness = computeVibiness(queryWords);
    const totalTokens = queryWords.length + querySymbols.length;
    const queryNumericRatio = totalTokens > 0 ? querySymbols.length / totalTokens : 0.0;

    return {
      rawQuery: queryText,
      cleanQuery,
      queryWords,
      querySymbols,
      uniqueBpe1,
      tokenToId1,
      maxIdfToken1,
      medianIdf1,
      maxQueryBm25_1,
      uniqueBpe2,
      maxIdfToken2,
      maxQueryBm25_2,
      fullCitationQuery2,
      hasFullCitation2,
      uniqueNgrams3,
      maxQueryBm25_3,
      salientPhrases,
      queryVibiness,
      queryNumericRatio
    };
  }

  computeFeaturesInto(
    prep: PreparedQuery,
    doc: IndexedDocument,
    docIdx: number,
    queryEmbedding: Float32Array | null,
    precomputedDenseSim: number | null,
    scratch: SearchScratchpad
  ): void {
    const { featureBuffer, windowHits, termCounts, matchedTokens1 } = scratch;
    windowHits.length = 0;
    matchedTokens1.clear();

    const k1 = 1.2;
    const b = 0.75;

    // =========================================================================
    // 1. TOKENIZER 1: BM25, Coverage, Positional hits, Salience
    // =========================================================================
    const lenNorm1 = 1.0 - b + b * (doc.docLengthT1 / (this.avgdlT1 || 1.0));
    let rawBm25_1 = 0.0;
    let totalHitCount1 = 0;
    let firstSalientPos1 = -1;

    for (let i = 0; i < prep.uniqueBpe1.length; i++) {
      const t = prep.uniqueBpe1[i]!;
      if (doc.docBpeSet1.has(t)) {
        matchedTokens1.add(t);
        const idf = this.getIdf(t, 1);
        const tf = doc.bpeTfMap1.get(t) || 0;
        totalHitCount1 += tf;
        rawBm25_1 += idf * ((tf * (k1 + 1)) / (tf + k1 * lenNorm1));

        const positions = doc.bpePosMap1.get(t);
        if (positions) {
          const isContent = !FUNCTION_WORDS.has(t) && idf >= prep.medianIdf1;
          for (let p = 0; p < positions.length; p++) {
            const pos = positions[p]!;
            windowHits.push({ pos, termId: i });
            if (isContent && (firstSalientPos1 === -1 || pos < firstSalientPos1)) {
              firstSalientPos1 = pos;
            }
          }
        }
      }
    }

    const bm25_t1 = prep.maxQueryBm25_1 > 0 ? Math.min(1.0, rawBm25_1 / prep.maxQueryBm25_1) : 0.0;
    const coverage_t1 = prep.uniqueBpe1.length > 0 ? matchedTokens1.size / prep.uniqueBpe1.length : 0.0;
    const salienceGate_t1 = doc.docBpeSet1.has(prep.maxIdfToken1) ? 1.0 : 0.0;

    // =========================================================================
    // 2. TOKENIZER 2: Symbol & Alphanumeric BM25, Coverage, Salience, Exact Citation
    // =========================================================================
    let bm25_t2 = 0.0;
    let coverage_t2 = 0.0;
    let salienceGate_t2 = 0.0;

    if (prep.uniqueBpe2.length > 0) {
      const lenNorm2 = 1.0 - b + b * (doc.docLengthT2 / (this.avgdlT2 || 1.0));
      let rawBm25_2 = 0.0;
      let matchedCount2 = 0;

      for (let i = 0; i < prep.uniqueBpe2.length; i++) {
        const t = prep.uniqueBpe2[i]!;
        if (doc.docBpeSet2.has(t)) {
          matchedCount2++;
          const idf = this.getIdf(t, 2);
          const tf = doc.bpeTfMap2.get(t) || 0;
          rawBm25_2 += idf * ((tf * (k1 + 1)) / (tf + k1 * lenNorm2));
        }
      }

      bm25_t2 = prep.maxQueryBm25_2 > 0 ? Math.min(1.0, rawBm25_2 / prep.maxQueryBm25_2) : 0.0;
      coverage_t2 = matchedCount2 / prep.uniqueBpe2.length;
      salienceGate_t2 = doc.docBpeSet2.has(prep.maxIdfToken2) ? 1.0 : 0.0;
    }

    const exactCitation_t2 =
      prep.hasFullCitation2 && doc.docSymbolShingles2.has(prep.fullCitationQuery2) ? 1.0 : 0.0;

    // =========================================================================
    // 3. TOKENIZER 3: Language N-Gram BM25, Coverage
    // =========================================================================
    let bm25_t3 = 0.0;
    let coverage_t3 = 0.0;

    if (prep.uniqueNgrams3.length > 0) {
      const lenNorm3 = 1.0 - b + b * (doc.docLengthT3 / (this.avgdlT3 || 1.0));
      let rawBm25_3 = 0.0;
      let matchedCount3 = 0;

      for (let i = 0; i < prep.uniqueNgrams3.length; i++) {
        const g = prep.uniqueNgrams3[i]!;
        if (doc.docNgramSet3.has(g)) {
          matchedCount3++;
          const idf = this.getIdf(g, 3);
          const tf = doc.ngramTfMap3.get(g) || 0;
          rawBm25_3 += idf * ((tf * (k1 + 1)) / (tf + k1 * lenNorm3));
        }
      }

      bm25_t3 = prep.maxQueryBm25_3 > 0 ? Math.min(1.0, rawBm25_3 / prep.maxQueryBm25_3) : 0.0;
      coverage_t3 = matchedCount3 / prep.uniqueNgrams3.length;
    }

    // =========================================================================
    // 4. CLAUSE INTEGRITY (Computed internally from minimal covering span)
    // =========================================================================
    let clauseIntegrity = 0.0;
    const K = matchedTokens1.size;

    if (K >= 2 && windowHits.length >= K) {
      windowHits.sort((a, b) => a.pos - b.pos);
      termCounts.fill(0);

      let distinctCount = 0;
      let left = 0;
      let minSpan = Infinity;
      let bestL = 0;
      let bestR = 0;

      for (let right = 0; right < windowHits.length; right++) {
        const tRight = windowHits[right]!.termId;
        const cRight = termCounts[tRight]!;
        if (cRight === 0) distinctCount++;
        termCounts[tRight] = cRight + 1;

        while (distinctCount === K) {
          const span = windowHits[right]!.pos - windowHits[left]!.pos + 1;
          if (span < minSpan) {
            minSpan = span;
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

      if (minSpan !== Infinity && bestR > bestL) {
        let sentenceTerminals = 0;
        for (let p = bestL; p < bestR; p++) {
          if (doc.punctuationIndices.has(p)) sentenceTerminals++;
        }
        clauseIntegrity = 1.0 / (1.0 + sentenceTerminals);
      }
    }

    // =========================================================================
    // 5. EXACT MATCHES, POSITIONS, DENSITY & PRIORS
    // =========================================================================
    let exactPhrase_t1 = 0.0;
    if (prep.cleanQuery.length > 0 && doc.cleanText.includes(prep.cleanQuery)) {
      exactPhrase_t1 = 1.0;
    } else if (prep.salientPhrases.length > 0) {
      let matchedCount = 0;
      for (let i = 0; i < prep.salientPhrases.length; i++) {
        if (doc.cleanText.includes(prep.salientPhrases[i]!)) matchedCount++;
      }
      exactPhrase_t1 = Math.min(1.0, matchedCount / Math.min(3, prep.salientPhrases.length));
    }

    const earlyPositionBias_t1 = firstSalientPos1 >= 0 ? 1.0 / (1.0 + 0.05 * firstSalientPos1) : 0.0;
    const informationDensity = doc.docLengthT1 > 0 ? Math.min(1.0, totalHitCount1 / doc.docLengthT1) : 0.0;

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

    // =========================================================================
    // WRITE THE DEFINITIVE 17-FEATURE VECTOR (ZERO HEAP ALLOCATION)
    // =========================================================================
    featureBuffer[0]  = bm25_t1;
    featureBuffer[1]  = bm25_t2;
    featureBuffer[2]  = bm25_t3;
    featureBuffer[3]  = coverage_t1;
    featureBuffer[4]  = coverage_t2;
    featureBuffer[5]  = coverage_t3;
    featureBuffer[6]  = salienceGate_t1;
    featureBuffer[7]  = salienceGate_t2;
    featureBuffer[8]  = exactPhrase_t1;
    featureBuffer[9]  = exactCitation_t2;
    featureBuffer[10] = clauseIntegrity;
    featureBuffer[11] = earlyPositionBias_t1;
    featureBuffer[12] = informationDensity;
    featureBuffer[13] = Math.max(0.0, Math.min(1.0, denseCosineSimilarity));
    featureBuffer[14] = prep.queryVibiness;
    featureBuffer[15] = prep.queryNumericRatio;
    featureBuffer[16] = doc.docProseQuality;
  }

  computeFeatures(
    queryText: string,
    docIdx: number,
    queryEmbedding: Float32Array | null = null,
    precomputedDenseSim: number | null = null
  ): Float32Array {
    const prep = this.prepareQuery(queryText);
    const doc = this.documents[docIdx];
    if (!doc) return new Float32Array(NUM_FEATURES);

    const scratch = createSearchScratchpad(prep.uniqueBpe1.length);
    this.computeFeaturesInto(prep, doc, docIdx, queryEmbedding, precomputedDenseSim, scratch);

    const copy = new Float32Array(NUM_FEATURES);
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

    if (!queryEmbedding && this.denseEmbeddings) {
      queryEmbedding = await this.fetchQueryEmbedding(queryText);
    }

    const prep = this.prepareQuery(queryText);
    const scratch = createSearchScratchpad(prep.uniqueBpe1.length);
    const candidates: ScoredCandidate[] = new Array(this.docCount);

    for (let docIdx = 0; docIdx < this.docCount; docIdx++) {
      const doc = this.documents[docIdx]!;
      this.computeFeaturesInto(prep, doc, docIdx, queryEmbedding, null, scratch);

      const f = scratch.featureBuffer;
      const score = scorePassage(f);

      candidates[docIdx] = {
        docIdx,
        score,
        denseSim: f[13]!
      };
    }

    candidates.sort((a, b) => b.score - a.score);
    const finalCount = Math.min(topK, candidates.length);

    const results: SearchResult[] = new Array(finalCount);
    for (let i = 0; i < finalCount; i++) {
      const c = candidates[i]!;
      const doc = this.documents[c.docIdx]!;

      this.computeFeaturesInto(prep, doc, c.docIdx, queryEmbedding, null, scratch);

      results[i] = {
        id: doc.id,
        text: doc.rawText,
        score: c.score,
        probability: c.score,
        denseSim: c.denseSim,
        features: Array.from(scratch.featureBuffer)
      };
    }

    return results;
  }
}
