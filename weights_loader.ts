import fs from 'node:fs';

export const BINARY_PATH = './weights.bin';
export const NUM_FEATURES = 18;
export const HIDDEN_NEURONS = 6;

export const FEATURE_NAMES: readonly string[] = [
  /*  0 */ 'BPE Lexical Ratio',
  /*  1 */ 'N-Gram Fuzzy Ratio',
  /*  2 */ 'Query Coverage',
  /*  3 */ 'Proximity Span',
  /*  4 */ 'Clause Integrity',
  /*  5 */ 'Early Position Bias',
  /*  6 */ 'Order Monotonicity',
  /*  7 */ 'Dense Cosine Similarity',
  /*  8 */ 'Full BM25 Saturation',
  /*  9 */ 'Salience Gate (Max IDF)',
  /* 10 */ 'Information Density',
  /* 11 */ 'Exact Phrase Match',
  /* 12 */ 'Query Vibiness Prior',
  /* 13 */ 'Query Numeric Ratio',
  /* 14 */ 'Document Prose Quality Prior',
  /* 15 */ 'Symbol/Number Coverage',
  /* 16 */ 'Symbol Salience Gate',
  /* 17 */ 'Exact Citation Hierarchy Match'
] as const;

export interface WriteModelWeightsInput {
  W_skip: Float32Array;
  W_gate: Float32Array;
  b_gate: Float32Array;
  W_val: Float32Array;
  b_val: Float32Array;
  W_out: Float32Array;
  b_out: number;
  temperature?: number;
  calibrationBias?: number;
}

export interface ModelWeights {
  NUM_FEATURES: number;
  HIDDEN_NEURONS: number;
  W_skip: Float32Array;
  W_gate: Float32Array;
  b_gate: Float32Array;
  W_val: Float32Array;
  b_val: Float32Array;
  W_out: Float32Array;
  b_out: number;
  temperature: number;
  calibrationBias: number;
  isDefault: boolean;
}

/**
 * Total parameters in SwiGLU + Skip model (18 - 6 - 1):
 *   W_skip          : 18
 *   W_gate          : 18 * 6 = 108
 *   b_gate          : 6
 *   W_val           : 18 * 6 = 108
 *   b_val           : 6
 *   W_out           : 6
 *   b_out           : 1
 *   temperature     : 1
 *   calibrationBias : 1
 * Total = 255 floats = 1,020 bytes (+ 8 bytes header = 1,028 bytes)
 */
export function writeModelWeights(
  modelData: WriteModelWeightsInput,
  filepath: string = BINARY_PATH
): void {
  const {
    W_skip,
    W_gate,
    b_gate,
    W_val,
    b_val,
    W_out,
    b_out,
    temperature = 1.0,
    calibrationBias = 0.0
  } = modelData;

  const safeTemperature =
    Number.isFinite(temperature) && temperature > 0 ? temperature : 1.0;
  const safeCalibrationBias =
    Number.isFinite(calibrationBias) ? calibrationBias : 0.0;

  const totalFloats =
    NUM_FEATURES +
    NUM_FEATURES * HIDDEN_NEURONS * 2 +
    HIDDEN_NEURONS * 3 +
    3;

  const buf = Buffer.alloc(8 + totalFloats * 4); // 8 bytes header + floats

  buf.writeUInt32LE(NUM_FEATURES, 0);
  buf.writeUInt32LE(HIDDEN_NEURONS, 4);

  let offset = 8;
  const writeArray = (arr: Float32Array): void => {
    for (let i = 0; i < arr.length; i++) {
      buf.writeFloatLE(arr[i]!, offset);
      offset += 4;
    }
  };

  writeArray(W_skip);
  writeArray(W_gate);
  writeArray(b_gate);
  writeArray(W_val);
  writeArray(b_val);
  writeArray(W_out);

  buf.writeFloatLE(b_out, offset);
  offset += 4;

  buf.writeFloatLE(safeTemperature, offset);
  offset += 4;
  buf.writeFloatLE(safeCalibrationBias, offset);

  fs.writeFileSync(filepath, buf);
}

export function loadModelWeights(filepath: string = BINARY_PATH): ModelWeights {
  if (!fs.existsSync(filepath)) {
    return {
      NUM_FEATURES,
      HIDDEN_NEURONS,
      W_skip: new Float32Array(NUM_FEATURES).fill(0.2),
      W_gate: new Float32Array(NUM_FEATURES * HIDDEN_NEURONS).fill(0.1),
      b_gate: new Float32Array(HIDDEN_NEURONS).fill(0.0),
      W_val: new Float32Array(NUM_FEATURES * HIDDEN_NEURONS).fill(0.1),
      b_val: new Float32Array(HIDDEN_NEURONS).fill(0.0),
      W_out: new Float32Array(HIDDEN_NEURONS).fill(0.2),
      b_out: -1.5,
      temperature: 1.0,
      calibrationBias: 0.0,
      isDefault: true
    };
  }

  const buf = fs.readFileSync(filepath);
  let offset = 8;

  const readArray = (len: number): Float32Array => {
    const arr = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      arr[i] = buf.readFloatLE(offset);
      offset += 4;
    }
    return arr;
  };

  const W_skip = readArray(NUM_FEATURES);
  const W_gate = readArray(NUM_FEATURES * HIDDEN_NEURONS);
  const b_gate = readArray(HIDDEN_NEURONS);
  const W_val  = readArray(NUM_FEATURES * HIDDEN_NEURONS);
  const b_val  = readArray(HIDDEN_NEURONS);
  const W_out  = readArray(HIDDEN_NEURONS);
  const b_out  = buf.readFloatLE(offset);
  offset += 4;

  let temperature = 1.0;
  let calibrationBias = 0.0;
  if (buf.length >= offset + 8) {
    temperature = buf.readFloatLE(offset);
    calibrationBias = buf.readFloatLE(offset + 4);
    if (!Number.isFinite(temperature) || temperature <= 0) temperature = 1.0;
    if (!Number.isFinite(calibrationBias)) calibrationBias = 0.0;
  }

  return {
    NUM_FEATURES,
    HIDDEN_NEURONS,
    W_skip,
    W_gate,
    b_gate,
    W_val,
    b_val,
    W_out,
    b_out,
    temperature,
    calibrationBias,
    isDefault: false
  };
}

export const model: ModelWeights = loadModelWeights();
