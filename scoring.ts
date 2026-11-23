import fs from 'node:fs';
import { model, NUM_FEATURES, HIDDEN_NEURONS } from './weights_loader.ts';

const {
  W_skip, W_gate, b_gate, W_val, b_val, W_out, b_out,
  temperature = 1.0, calibrationBias = 0.0
} = model;

export const CALIBRATION_TEMPERATURE =
  Number.isFinite(temperature) && temperature > 0 ? temperature : 1.0;
export const CALIBRATION_BIAS =
  Number.isFinite(calibrationBias) ? calibrationBias : 0.0;

// Pure grammatical glue
export const PURE_STOPWORDS = new Set<string>([
  'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'on', 'by',
  'with', 'at', 'from', 'as', 'into', 'that', 'this', 'it', 'its', 'their',
  'than', 'then', 'so', 'such', 'be', 'been', 'being', 'have', 'has', 'had'
]);

// Conversational / Question markers (high vibe)
export const WH_WORDS = new Set<string>([
  'how', 'what', 'why', 'when', 'where', 'who', 'whom', 'whose',
  'can', 'could', 'should', 'would', 'will', 'is', 'are', 'was',
  'were', 'does', 'did', 'do', 'if', 'about', 'think', 'mean', 'meaning', 'happen',
  'happens', 'doing', 'like', 'please', 'thank', 'thanks', 'help', 'lol', 'lmao', 'omg', 'omfg', 'btw', 'idk',
  'i', 'my', 'me', 'he', 'his','she', 'her', 'you'
]);

// Combined non-content words (used to filter out garbage phrases)
export const FUNCTION_WORDS = new Set<string>([
  ...PURE_STOPWORDS,
  ...WH_WORDS
]);

// Binary deserialization of simple_vocab.bin
function loadSimpleVocab(): Set<string> {
  const BIN_PATH = './simple_vocab.bin';
  const JSON_PATH = './simple_vocab.json';

  if (fs.existsSync(BIN_PATH)) {
    const buf = fs.readFileSync(BIN_PATH);
    const count = buf.readUInt32LE(0);
    const vocab = new Set<string>();

    let offset = 4;
    for (let i = 0; i < count; i++) {
      const len = buf.readUInt8(offset);
      offset += 1;
      const word = buf.toString('utf-8', offset, offset + len);
      offset += len;
      vocab.add(word);
    }
    return vocab;
  }

  if (fs.existsSync(JSON_PATH)) {
    return new Set<string>(JSON.parse(fs.readFileSync(JSON_PATH, 'utf-8')));
  }

  return new Set<string>();
}

export const SIMPLE_VOCAB = loadSimpleVocab();

export function computeVibiness(queryWords: string[] | null | undefined): number {
  if (!queryWords || queryWords.length === 0) return 0.0;
  let vibePoints = 0;
  for (let i = 0; i < queryWords.length; i++) {
    const clean = queryWords[i]!.toLowerCase();
    if (WH_WORDS.has(clean)) vibePoints += 1.0;
    else if (SIMPLE_VOCAB.has(clean)) vibePoints += 0.3;
  }
  return Math.min(1.0, vibePoints / queryWords.length);
}

function silu(x: number): number {
  return x / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

export function scorePassage(x: Float32Array | number[]): number {
  // 1. Direct Skip Highway
  let z = b_out;
  for (let i = 0; i < NUM_FEATURES; i++) {
    z += x[i]! * W_skip[i]!;
  }

  // 2. Direct SwiGLU Hidden Layer (Zero RMSNorm)
  for (let j = 0; j < HIDDEN_NEURONS; j++) {
    let gate = b_gate[j]!;
    let val = b_val[j]!;

    for (let i = 0; i < NUM_FEATURES; i++) {
      const xi = x[i]!;
      gate += xi * W_gate[i * HIDDEN_NEURONS + j]!;
      val  += xi * W_val[i * HIDDEN_NEURONS + j]!;
    }

    const h_j = silu(gate) * val;
    z += h_j * W_out[j]!;
  }

  const calibratedLogit = (z + CALIBRATION_BIAS) / CALIBRATION_TEMPERATURE;
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, calibratedLogit))));
}

export function scoreProbability(x: Float32Array | number[]): number {
  return scorePassage(x);
}

export interface FeatureExtractionParams {
  bpeHitIdfSum: number;
  bpeQueryIdfSum: number;
  ngramHitIdfSum: number;
  ngramQueryIdfSum: number;
  matchedWordCount: number;
  uniqueBpeCount: number;
  minCoveringSpan: number;
  sentenceTerminalsInSpan: number;
  firstSalientPos: number;
  isOrdered: boolean;
  denseCosineSimilarity?: number;
  rawBm25Score: number;
  maxQueryBm25: number;
  hasSalientTerm: boolean;
  totalHitCount: number;
  docWordCount: number;
  phraseMatchScore: number;
  queryVibiness: number;
  queryNumericRatio: number;
  docProseQuality: number;
  symbolMatchedCount: number;
  totalQuerySymbols: number;
  hasSalientSymbol: boolean;
  hasExactCitation: boolean;
}

export function extractNormalizedFeatures(params: FeatureExtractionParams): Float32Array {
  const {
    bpeHitIdfSum, bpeQueryIdfSum, ngramHitIdfSum, ngramQueryIdfSum,
    matchedWordCount, uniqueBpeCount, minCoveringSpan, sentenceTerminalsInSpan,
    firstSalientPos, isOrdered, denseCosineSimilarity = 0.0, rawBm25Score,
    maxQueryBm25, hasSalientTerm, totalHitCount, docWordCount, phraseMatchScore,
    queryVibiness, queryNumericRatio, docProseQuality, symbolMatchedCount,
    totalQuerySymbols, hasSalientSymbol, hasExactCitation
  } = params;

  const x = new Float32Array(NUM_FEATURES);

  x[0] = bpeQueryIdfSum > 0 ? Math.min(1.0, bpeHitIdfSum / bpeQueryIdfSum) : 0.0;
  x[1] = ngramQueryIdfSum > 0 ? Math.min(1.0, ngramHitIdfSum / ngramQueryIdfSum) : 0.0;
  x[2] = uniqueBpeCount > 0 ? Math.min(1.0, matchedWordCount / uniqueBpeCount) : 0.0;
  x[3] = (minCoveringSpan !== Infinity && minCoveringSpan >= matchedWordCount && matchedWordCount >= 2)
    ? Math.min(1.0, matchedWordCount / minCoveringSpan)
    : 0.0;
  x[4] = (minCoveringSpan !== Infinity && matchedWordCount >= 2)
    ? 1.0 / (1.0 + sentenceTerminalsInSpan)
    : 0.0;
  x[5] = firstSalientPos >= 0 ? 1.0 / (1.0 + 0.05 * firstSalientPos) : 0.0;
  x[6] = isOrdered ? 1.0 : 0.0;
  x[7] = Math.max(0.0, Math.min(1.0, denseCosineSimilarity));
  x[8] = maxQueryBm25 > 0 ? Math.min(1.0, rawBm25Score / maxQueryBm25) : 0.0;
  x[9] = hasSalientTerm ? 1.0 : 0.0;
  x[10] = docWordCount > 0 ? Math.min(1.0, totalHitCount / docWordCount) : 0.0;
  x[11] = Math.max(0.0, Math.min(1.0, phraseMatchScore));
  x[12] = queryVibiness;
  x[13] = queryNumericRatio;
  x[14] = docProseQuality;
  x[15] = totalQuerySymbols > 0 ? Math.min(1.0, symbolMatchedCount / totalQuerySymbols) : 0.0;
  x[16] = hasSalientSymbol ? 1.0 : 0.0;
  x[17] = hasExactCitation ? 1.0 : 0.0;

  return x;
}
