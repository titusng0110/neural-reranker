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

export const PURE_STOPWORDS = new Set<string>([
  'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'on', 'by',
  'with', 'at', 'from', 'as', 'into', 'that', 'this', 'it', 'its', 'their',
  'than', 'then', 'so', 'such', 'be', 'been', 'being', 'have', 'has', 'had'
]);

export const WH_WORDS = new Set<string>([
  'how', 'what', 'why', 'when', 'where', 'who', 'whom', 'whose',
  'can', 'could', 'should', 'would', 'will', 'is', 'are', 'was',
  'were', 'does', 'did', 'do', 'if', 'about', 'think', 'mean', 'meaning', 'happen',
  'happens', 'doing', 'like', 'please', 'thank', 'thanks', 'help', 'lol', 'lmao', 'omg', 'omfg', 'btw', 'idk',
  'i', 'my', 'me', 'he', 'his','she', 'her', 'you'
]);

export const FUNCTION_WORDS = new Set<string>([
  ...PURE_STOPWORDS,
  ...WH_WORDS
]);

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
  // 1. Skip Highway
  let z = b_out;
  for (let i = 0; i < NUM_FEATURES; i++) {
    z += x[i]! * W_skip[i]!;
  }

  // 2. SwiGLU Hidden Layer
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
  bm25_t1: number;
  bm25_t2: number;
  bm25_t3: number;
  coverage_t1: number;
  coverage_t2: number;
  coverage_t3: number;
  salienceGate_t1: boolean;
  salienceGate_t2: boolean;
  exactPhrase_t1: number;
  exactCitation_t2: boolean;
  clauseIntegrity: number;
  earlyPositionBias_t1: number;
  informationDensity: number;
  denseCosineSimilarity?: number;
  queryVibiness: number;
  queryNumericRatio: number;
  docProseQuality: number;
}

export function extractNormalizedFeatures(params: FeatureExtractionParams): Float32Array {
  const x = new Float32Array(NUM_FEATURES);

  x[0] = Math.max(0.0, Math.min(1.0, params.bm25_t1));
  x[1] = Math.max(0.0, Math.min(1.0, params.bm25_t2));
  x[2] = Math.max(0.0, Math.min(1.0, params.bm25_t3));
  x[3] = Math.max(0.0, Math.min(1.0, params.coverage_t1));
  x[4] = Math.max(0.0, Math.min(1.0, params.coverage_t2));
  x[5] = Math.max(0.0, Math.min(1.0, params.coverage_t3));
  x[6] = params.salienceGate_t1 ? 1.0 : 0.0;
  x[7] = params.salienceGate_t2 ? 1.0 : 0.0;
  x[8] = Math.max(0.0, Math.min(1.0, params.exactPhrase_t1));
  x[9] = params.exactCitation_t2 ? 1.0 : 0.0;
  x[10] = Math.max(0.0, Math.min(1.0, params.clauseIntegrity));
  x[11] = Math.max(0.0, Math.min(1.0, params.earlyPositionBias_t1));
  x[12] = Math.max(0.0, Math.min(1.0, params.informationDensity));
  x[13] = Math.max(0.0, Math.min(1.0, params.denseCosineSimilarity ?? 0.0));
  x[14] = Math.max(0.0, Math.min(1.0, params.queryVibiness));
  x[15] = Math.max(0.0, Math.min(1.0, params.queryNumericRatio));
  x[16] = Math.max(0.0, Math.min(1.0, params.docProseQuality));

  return x;
}
