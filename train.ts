import fs from 'node:fs';
import readline from 'node:readline';
import { performance } from 'node:perf_hooks';
import { SearchEngine } from './search.ts';
import { NUM_FEATURES, HIDDEN_NEURONS, writeModelWeights } from './weights_loader.ts';

// ============================================================================
// CONFIGURATION & HYPERPARAMETERS
// ============================================================================
const DATASET_PATH =
  process.argv[2] ||
  (fs.existsSync('./training_data.jsonl')
    ? './training_data.jsonl'
    : './training_dataset.jsonl');

const DENSE_BIN_PATH = './training_embeddings.bin';
const DENSE_JSON_PATH = './training_embeddings.json';

// Feature Dropout: 5% chance to mask each retrieval feature (0..13) during training
const FEATURE_DROPOUT_PROB = 0.05;
const NUM_RETRIEVAL_FEATURES = 14; // Features 0 through 13

// Label Smoothing (Szegedy et al., 2016)
const LABEL_SMOOTH_ALPHA = 0.02; // Soft targets: [0.02, 0.98]

// Master Training Schedule (Full 1,000 Epochs with Cosine Annealing)
const TOTAL_EPOCHS = 1000;
const WARMUP_EPOCHS = 30;
const BASE_LR = 0.025;
const MIN_LR = 0.0005;

// AdamW Hyperparameters (Loshchilov & Hutter, 2019)
const WEIGHT_DECAY = 0.001;
const ADAM_BETA1 = 0.9;
const ADAM_BETA2 = 0.999;
const ADAM_EPS = 1e-8;

// Basin-Hopping Initializer (Wales & Doye, 1997)
const BH_HOPS = 100;
const BH_MINI_EPOCHS = 6;
const BH_PERTURB_SIGMA = 0.07;
const BH_METROPOLIS_TEMP = 0.020;

// ============================================================================
// DETERMINISTIC PSEUDO-RANDOM & GAUSSIAN GENERATOR
// ============================================================================
interface PRNG {
  uniform: () => number;
  normal: (mean?: number, std?: number) => number;
}

function createPRNG(seed = 1337): PRNG {
  let t = seed;
  const uniform = (): number => {
    t = (t + 0x6d2b79f5) | 0;
    let v = Math.imul(t ^ (t >>> 15), t | 1);
    v ^= v + Math.imul(v ^ (v >>> 7), v | 61);
    return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
  };

  let spare: number | null = null;
  const normal = (mean = 0, std = 1): number => {
    if (spare !== null) {
      const val = spare;
      spare = null;
      return mean + val * std;
    }
    let u = 0, v = 0, s = 0;
    do {
      u = uniform() * 2 - 1;
      v = uniform() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const mul = Math.sqrt((-2.0 * Math.log(s)) / s);
    spare = v * mul;
    return mean + u * mul * std;
  };

  return { uniform, normal };
}

// ============================================================================
// DATASET & EMBEDDINGS LOADING
// ============================================================================
interface DenseEmbeddingsStore {
  queryEmbeddings: Float32Array[];
  passageEmbeddings: Float32Array[];
}

function loadTrainingEmbeddings(): DenseEmbeddingsStore | null {
  if (fs.existsSync(DENSE_BIN_PATH)) {
    console.log(`Loading binary dense vectors from "${DENSE_BIN_PATH}" (zero-copy)...`);
    const buf = fs.readFileSync(DENSE_BIN_PATH);
    const countQ = buf.readUInt32LE(0);
    const countP = buf.readUInt32LE(4);
    const dim = buf.readUInt32LE(8);

    const floatBaseOffset = buf.byteOffset + 12;
    const allFloats = new Float32Array(buf.buffer, floatBaseOffset, (countQ + countP) * dim);

    const queryEmbeddings: Float32Array[] = [];
    for (let i = 0; i < countQ; i++) {
      queryEmbeddings.push(allFloats.subarray(i * dim, (i + 1) * dim));
    }

    const passageEmbeddings: Float32Array[] = [];
    const pOffset = countQ * dim;
    for (let i = 0; i < countP; i++) {
      passageEmbeddings.push(allFloats.subarray(pOffset + i * dim, pOffset + (i + 1) * dim));
    }

    return { queryEmbeddings, passageEmbeddings };
  }

  if (fs.existsSync(DENSE_JSON_PATH)) {
    console.log(`Loading precomputed dense vectors from legacy "${DENSE_JSON_PATH}"...`);
    const raw = JSON.parse(fs.readFileSync(DENSE_JSON_PATH, 'utf-8'));
    return {
      queryEmbeddings: raw.queryEmbeddings.map((v: number[]) => new Float32Array(v)),
      passageEmbeddings: raw.passageEmbeddings.map((v: number[]) => new Float32Array(v))
    };
  }

  return null;
}

interface TrainingDatasetPair {
  query: string;
  rule: string;
  docId: string;
}

async function loadJsonlDataset(filepath: string): Promise<TrainingDatasetPair[]> {
  if (!fs.existsSync(filepath)) {
    console.error(`\n❌ Error: Cannot find dataset file "${filepath}".`);
    process.exit(1);
  }

  const pairs: TrainingDatasetPair[] = [];
  const rl = readline.createInterface({
    input: fs.createReadStream(filepath),
    crlfDelay: Infinity
  });

  let counter = 0;
  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('//') || trimmed.startsWith('#')) continue;

    try {
      const parsed = JSON.parse(trimmed);
      const query = parsed.query || parsed.q;
      const passage = parsed.passage || parsed.rule || parsed.text || parsed.document;
      const docId = parsed.docId || parsed.doc || `chunk-${Math.floor(counter / 3) + 1}`;

      if (!query || !passage) continue;
      pairs.push({ query: String(query).trim(), rule: String(passage).trim(), docId: String(docId) });
      counter++;
    } catch {}
  }

  return pairs;
}

// ============================================================================
// MODEL STATE & SCRATCHPAD MEMORY
// ============================================================================
interface ModelState {
  W_skip: Float32Array;
  W_gate: Float32Array;
  b_gate: Float32Array;
  W_val: Float32Array;
  b_val: Float32Array;
  W_out: Float32Array;
  b_out: number;
}

function cloneModelState(src: ModelState): ModelState {
  return {
    W_skip: new Float32Array(src.W_skip),
    W_gate: new Float32Array(src.W_gate),
    b_gate: new Float32Array(src.b_gate),
    W_val: new Float32Array(src.W_val),
    b_val: new Float32Array(src.b_val),
    W_out: new Float32Array(src.W_out),
    b_out: src.b_out
  };
}

function copyModelState(src: ModelState, dst: ModelState): void {
  dst.W_skip.set(src.W_skip);
  dst.W_gate.set(src.W_gate);
  dst.b_gate.set(src.b_gate);
  dst.W_val.set(src.W_val);
  dst.b_val.set(src.b_val);
  dst.W_out.set(src.W_out);
  dst.b_out = src.b_out;
}

function createInitialModel(rng: PRNG): ModelState {
  const W_skip = new Float32Array(NUM_FEATURES);
  const skipStd = 1.0 / Math.sqrt(NUM_FEATURES);
  for (let i = 0; i < NUM_FEATURES; i++) {
    W_skip[i] = rng.normal(0.0, skipStd);
  }

  const W_gate = new Float32Array(NUM_FEATURES * HIDDEN_NEURONS);
  const W_val  = new Float32Array(NUM_FEATURES * HIDDEN_NEURONS);
  const inStd = 1.0 / Math.sqrt(NUM_FEATURES);
  for (let i = 0; i < W_gate.length; i++) {
    W_gate[i] = rng.normal(0.0, inStd);
    W_val[i]  = rng.normal(0.0, inStd);
  }

  const b_gate = new Float32Array(HIDDEN_NEURONS).fill(0.0);
  const b_val  = new Float32Array(HIDDEN_NEURONS).fill(0.0);

  const W_out = new Float32Array(HIDDEN_NEURONS);
  const outStd = 1.0 / Math.sqrt(HIDDEN_NEURONS);
  for (let j = 0; j < HIDDEN_NEURONS; j++) {
    W_out[j] = rng.normal(0.0, outStd);
  }

  const b_out = -1.2;
  return { W_skip, W_gate, b_gate, W_val, b_val, W_out, b_out };
}

interface ForwardScratchpad {
  z: number;
  gate: Float32Array;
  val: Float32Array;
  sigG: Float32Array;
  h: Float32Array;
}

function createForwardScratchpad(): ForwardScratchpad {
  return {
    z: 0.0,
    gate: new Float32Array(HIDDEN_NEURONS),
    val: new Float32Array(HIDDEN_NEURONS),
    sigG: new Float32Array(HIDDEN_NEURONS),
    h: new Float32Array(HIDDEN_NEURONS)
  };
}

interface AdamWOptimizerState {
  mSkip: Float32Array;  vSkip: Float32Array;
  mGate: Float32Array;  vGate: Float32Array;
  mBGate: Float32Array; vBGate: Float32Array;
  mVal: Float32Array;   vVal: Float32Array;
  mBVal: Float32Array;  vBVal: Float32Array;
  mOut: Float32Array;   vOut: Float32Array;
}

function createAdamWState(): AdamWOptimizerState {
  return {
    mSkip: new Float32Array(NUM_FEATURES),
    vSkip: new Float32Array(NUM_FEATURES),
    mGate: new Float32Array(NUM_FEATURES * HIDDEN_NEURONS),
    vGate: new Float32Array(NUM_FEATURES * HIDDEN_NEURONS),
    mBGate: new Float32Array(HIDDEN_NEURONS),
    vBGate: new Float32Array(HIDDEN_NEURONS),
    mVal: new Float32Array(NUM_FEATURES * HIDDEN_NEURONS),
    vVal: new Float32Array(NUM_FEATURES * HIDDEN_NEURONS),
    mBVal: new Float32Array(HIDDEN_NEURONS),
    vBVal: new Float32Array(HIDDEN_NEURONS),
    mOut: new Float32Array(HIDDEN_NEURONS),
    vOut: new Float32Array(HIDDEN_NEURONS)
  };
}

function resetAdamWState(opt: AdamWOptimizerState): void {
  opt.mSkip.fill(0);  opt.vSkip.fill(0);
  opt.mGate.fill(0);  opt.vGate.fill(0);
  opt.mBGate.fill(0); opt.vBGate.fill(0);
  opt.mVal.fill(0);   opt.vVal.fill(0);
  opt.mBVal.fill(0);  opt.vBVal.fill(0);
  opt.mOut.fill(0);   opt.vOut.fill(0);
}

// ============================================================================
// FORWARD, BACKWARD, AND ADAMW KERNELS (ZERO-GC)
// ============================================================================
function forwardInto(x: Float32Array, model: ModelState, out: ForwardScratchpad): void {
  let z = model.b_out;
  for (let i = 0; i < NUM_FEATURES; i++) {
    z += x[i]! * model.W_skip[i]!;
  }

  const gate = out.gate;
  const val  = out.val;
  const sigG = out.sigG;
  const h    = out.h;

  for (let j = 0; j < HIDDEN_NEURONS; j++) {
    let g = model.b_gate[j]!;
    let v = model.b_val[j]!;

    for (let i = 0; i < NUM_FEATURES; i++) {
      const xi = x[i]!;
      const idx = i * HIDDEN_NEURONS + j;
      g += xi * model.W_gate[idx]!;
      v += xi * model.W_val[idx]!;
    }

    gate[j] = g;
    val[j]  = v;

    const clampedG = Math.max(-20, Math.min(20, g));
    const s = 1.0 / (1.0 + Math.exp(-clampedG));
    sigG[j] = s;
    const hj = g * s * v;
    h[j] = hj;

    z += hj * model.W_out[j]!;
  }

  out.z = z;
}

function computeLogit(x: Float32Array, model: ModelState): number {
  let z = model.b_out;
  for (let i = 0; i < NUM_FEATURES; i++) {
    z += x[i]! * model.W_skip[i]!;
  }

  for (let j = 0; j < HIDDEN_NEURONS; j++) {
    let g = model.b_gate[j]!;
    let v = model.b_val[j]!;

    for (let i = 0; i < NUM_FEATURES; i++) {
      const idx = i * HIDDEN_NEURONS + j;
      g += x[i]! * model.W_gate[idx]!;
      v += x[i]! * model.W_val[idx]!;
    }

    const clampedG = Math.max(-20, Math.min(20, g));
    const sig = 1.0 / (1.0 + Math.exp(-clampedG));
    z += (g * sig * v) * model.W_out[j]!;
  }

  return z;
}

function accumulateGradients(
  x: Float32Array,
  f: ForwardScratchpad,
  model: ModelState,
  deltaIn: number,
  gSkip: Float32Array,
  gGate: Float32Array,
  gBGate: Float32Array,
  gVal: Float32Array,
  gBVal: Float32Array,
  gOut: Float32Array
): void {
  for (let i = 0; i < NUM_FEATURES; i++) {
    gSkip[i] = gSkip[i]! + deltaIn * x[i]!;
  }

  for (let j = 0; j < HIDDEN_NEURONS; j++) {
    gOut[j] = gOut[j]! + deltaIn * f.h[j]!;
    const deltaH = deltaIn * model.W_out[j]!;

    const siluVal = f.gate[j]! * f.sigG[j]!;
    const deltaVal = deltaH * siluVal;
    const dSilu = f.sigG[j]! * (1.0 + f.gate[j]! * (1.0 - f.sigG[j]!));
    const deltaGate = deltaH * f.val[j]! * dSilu;

    gBVal[j]  = gBVal[j]!  + deltaVal;
    gBGate[j] = gBGate[j]! + deltaGate;

    for (let i = 0; i < NUM_FEATURES; i++) {
      const idx = i * HIDDEN_NEURONS + j;
      const xi = x[i]!;
      gVal[idx]  = gVal[idx]!  + deltaVal  * xi;
      gGate[idx] = gGate[idx]! + deltaGate * xi;
    }
  }
}

function stepAdamW(
  param: Float32Array,
  grad: Float32Array,
  m: Float32Array,
  v: Float32Array,
  stepIdx: number,
  lr: number,
  applyDecay: boolean
): void {
  const beta1 = ADAM_BETA1;
  const beta2 = ADAM_BETA2;
  const eps = ADAM_EPS;
  const bc1 = 1.0 - Math.pow(beta1, stepIdx);
  const bc2 = 1.0 - Math.pow(beta2, stepIdx);

  for (let i = 0; i < param.length; i++) {
    const g = grad[i]!;
    const mi = beta1 * m[i]! + (1.0 - beta1) * g;
    const vi = beta2 * v[i]! + (1.0 - beta2) * (g * g);
    m[i] = mi;
    v[i] = vi;

    const mHat = mi / bc1;
    const vHat = vi / bc2;

    if (applyDecay) {
      param[i] = param[i]! * (1.0 - lr * WEIGHT_DECAY);
    }
    param[i] = param[i]! - (lr * mHat) / (Math.sqrt(vHat) + eps);
  }
}

function getScheduledLR(
  epoch: number,
  totalEpochs: number = TOTAL_EPOCHS,
  baseLR: number = BASE_LR,
  minLR: number = MIN_LR,
  warmupEpochs: number = WARMUP_EPOCHS
): number {
  if (epoch <= warmupEpochs) {
    return minLR + (baseLR - minLR) * (epoch / warmupEpochs);
  }
  const progress = (epoch - warmupEpochs) / (totalEpochs - warmupEpochs);
  return minLR + 0.5 * (baseLR - minLR) * (1.0 + Math.cos(Math.PI * progress));
}

const sigmoid = (z: number): number => 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, z))));

interface CalibrationSample {
  z: number;
  label: number;
}

interface CalibrationResult {
  temperature: number;
  calibrationBias: number;
  brierBefore: number;
  brierAfter: number;
}

function fitCalibration(samples: CalibrationSample[]): CalibrationResult {
  if (samples.length === 0) {
    return { temperature: 1.0, calibrationBias: 0.0, brierBefore: 0.0, brierAfter: 0.0 };
  }

  const prob = (z: number, logT: number, bias: number) => sigmoid((z + bias) / Math.exp(logT));
  const brier = (logT: number, bias: number) => {
    let sum = 0.0;
    for (let i = 0; i < samples.length; i++) {
      const diff = prob(samples[i]!.z, logT, bias) - samples[i]!.label;
      sum += diff * diff;
    }
    return sum / samples.length;
  };

  const brierBefore = brier(0.0, 0.0);
  let logT = 0.0;
  let bias = 0.0;
  let mLogT = 0.0, vLogT = 0.0;
  let mBias = 0.0, vBias = 0.0;
  const LR = 0.04;

  for (let iter = 1; iter <= 1200; iter++) {
    const temperature = Math.exp(logT);
    let gBias = 0.0;
    let gLogT = 0.0;

    for (let i = 0; i < samples.length; i++) {
      const s = (samples[i]!.z + bias) / temperature;
      const p = sigmoid(s);
      const d = p - samples[i]!.label;
      gBias += d / temperature;
      gLogT += -d * s;
    }

    gBias /= samples.length;
    gLogT /= samples.length;

    mBias = 0.9 * mBias + 0.1 * gBias;
    vBias = 0.999 * vBias + 0.001 * (gBias * gBias);
    mLogT = 0.9 * mLogT + 0.1 * gLogT;
    vLogT = 0.999 * vLogT + 0.001 * (gLogT * gLogT);

    const mBHat = mBias / (1.0 - Math.pow(0.9, iter));
    const vBHat = vBias / (1.0 - Math.pow(0.999, iter));
    const mLHat = mLogT / (1.0 - Math.pow(0.9, iter));
    const vLHat = vLogT / (1.0 - Math.pow(0.999, iter));

    bias -= (LR * mBHat) / (Math.sqrt(vBHat) + 1e-8);
    logT -= (LR * mLHat) / (Math.sqrt(vLHat) + 1e-8);

    logT = Math.max(-3.0, Math.min(3.0, logT));
    bias = Math.max(-20.0, Math.min(20.0, bias));
  }

  return {
    temperature: Math.exp(logT),
    calibrationBias: bias,
    brierBefore,
    brierAfter: brier(logT, bias)
  };
}

// ============================================================================
// MAIN INDUSTRIAL TRAINING PIPELINE
// ============================================================================
async function run(): Promise<void> {
  console.log(`\n=============================================================`);
  console.log(`🚀 INDUSTRIAL HYBRID SwiGLU RANKER TRAINING (17 Features, Tri-BM25)`);
  console.log(`=============================================================`);

  const rng = createPRNG(1337);
  const dataset = await loadJsonlDataset(DATASET_PATH);
  const N_QUERIES = dataset.length;

  console.log(`Loaded ${N_QUERIES} training queries from "${DATASET_PATH}".`);

  // 1. Index unique documents
  const uniqueDocs: { id: string; text: string; firstSampleIdx: number }[] = [];
  const docIdToIdx = new Map<string, number>();

  for (let i = 0; i < dataset.length; i++) {
    const item = dataset[i]!;
    const id = item.docId;
    if (!docIdToIdx.has(id)) {
      const idx = uniqueDocs.length;
      docIdToIdx.set(id, idx);
      uniqueDocs.push({ id, text: item.rule, firstSampleIdx: i });
    }
  }

  const N_DOCS = uniqueDocs.length;
  console.log(`Indexed ${N_DOCS} unique passages into SearchEngine.`);

  const engine = new SearchEngine();
  for (let d = 0; d < uniqueDocs.length; d++) {
    engine.addDocument(uniqueDocs[d]!.id, uniqueDocs[d]!.text);
  }

  const queryTargetDoc: number[] = dataset.map((item) => docIdToIdx.get(item.docId)!);
  const denseData = loadTrainingEmbeddings();
  const docEmbeddings: (Float32Array | null)[] = uniqueDocs.map((d) => {
    return denseData?.passageEmbeddings?.[d.firstSampleIdx] || null;
  });

  // 2. Extract feature matrix
  console.log(`Extracting 17-feature vectors for all (${N_QUERIES} queries × ${N_DOCS} docs)...`);
  const queryFeatures: Float32Array[][] = [];

  for (let q = 0; q < N_QUERIES; q++) {
    const qVecs: Float32Array[] = [];
    const qEmb = denseData?.queryEmbeddings?.[q] || null;

    for (let d = 0; d < N_DOCS; d++) {
      const dEmb = docEmbeddings[d];
      let denseSim = 0.0;

      if (qEmb && dEmb) {
        let dot = 0.0;
        const len = qEmb.length;
        for (let k = 0; k < len; k++) {
          dot += qEmb[k]! * dEmb[k]!;
        }
        denseSim = dot;
      }

      qVecs.push(engine.computeFeatures(dataset[q]!.query, d, null, denseSim));
    }
    queryFeatures.push(qVecs);
  }

  function getRelevance(qIdx: number, dIdx: number): number {
    const target = queryTargetDoc[qIdx]!;
    if (dIdx === target) return 1.0;
    if (Math.abs(dIdx - target) === 1) return 0.5;
    return 0.0;
  }

  // 3. Construct natural pairwise training comparisons
  interface PairwiseComparison {
    vecA: Float32Array;
    vecB: Float32Array;
    targetProb: number;
    weight: number;
  }

  const pairs: PairwiseComparison[] = [];
  const MAX_RANDOM_NEGATIVES = 20;

  for (let q = 0; q < N_QUERIES; q++) {
    const positives: { dIdx: number; rel: number; vec: Float32Array }[] = [];
    const negatives: { dIdx: number; rel: number; vec: Float32Array }[] = [];

    for (let d = 0; d < N_DOCS; d++) {
      const rel = getRelevance(q, d);
      if (rel > 0) positives.push({ dIdx: d, rel, vec: queryFeatures[q]![d]! });
      else negatives.push({ dIdx: d, rel: 0.0, vec: queryFeatures[q]![d]! });
    }

    let selectedNegatives = negatives;
    if (negatives.length > MAX_RANDOM_NEGATIVES) {
      selectedNegatives = [];
      const pool = [...negatives];
      for (let k = 0; k < MAX_RANDOM_NEGATIVES && pool.length > 0; k++) {
        const randIdx = Math.floor(rng.uniform() * pool.length);
        selectedNegatives.push(pool.splice(randIdx, 1)[0]!);
      }
    }

    for (let p = 0; p < positives.length; p++) {
      const pos = positives[p]!;
      for (let n = 0; n < selectedNegatives.length; n++) {
        const neg = selectedNegatives[n]!;
        const deltaRel = pos.rel - neg.rel;

        const rawTarget = 0.5 + 0.5 * deltaRel;
        const targetProb = rawTarget * (1.0 - 2.0 * LABEL_SMOOTH_ALPHA) + LABEL_SMOOTH_ALPHA;

        pairs.push({
          vecA: pos.vec,
          vecB: neg.vec,
          targetProb,
          weight: deltaRel
        });
      }
    }
  }

  const numPairs = pairs.length;
  console.log(`Constructed ${numPairs.toLocaleString()} label-smoothed pairwise comparisons.\n`);

  // Pre-allocate scratchpads & gradient buffers once (Zero-GC throughout training)
  const scratchA = createForwardScratchpad();
  const scratchB = createForwardScratchpad();

  const gSkip  = new Float32Array(NUM_FEATURES);
  const gGate  = new Float32Array(NUM_FEATURES * HIDDEN_NEURONS);
  const gBGate = new Float32Array(HIDDEN_NEURONS);
  const gVal   = new Float32Array(NUM_FEATURES * HIDDEN_NEURONS);
  const gBVal  = new Float32Array(HIDDEN_NEURONS);
  const gOut   = new Float32Array(HIDDEN_NEURONS);

  // Pre-allocated static buffers for Feature Masking (Zero allocation inside hot loop)
  const savedA = new Float32Array(NUM_RETRIEVAL_FEATURES);
  const savedB = new Float32Array(NUM_RETRIEVAL_FEATURES);
  const droppedMask = new Uint8Array(NUM_RETRIEVAL_FEATURES);

  // Helper: Execute a single epoch step over all pairs
  function runSingleEpoch(
    m: ModelState,
    opt: AdamWOptimizerState,
    stepIdx: number,
    lr: number,
    enableDropout: boolean
  ): number {
    gSkip.fill(0);
    gGate.fill(0);
    gBGate.fill(0);
    gVal.fill(0);
    gBVal.fill(0);
    gOut.fill(0);

    let epochLoss = 0.0;

    for (let p = 0; p < numPairs; p++) {
      const pair = pairs[p]!;

      // 1. Apply 5% Random Feature Masking to Retrieval Features (0..13)
      if (enableDropout) {
        for (let f = 0; f < NUM_RETRIEVAL_FEATURES; f++) {
          if (rng.uniform() < FEATURE_DROPOUT_PROB) {
            droppedMask[f] = 1;
            savedA[f] = pair.vecA[f]!;
            savedB[f] = pair.vecB[f]!;
            pair.vecA[f] = 0.0;
            pair.vecB[f] = 0.0;
          } else {
            droppedMask[f] = 0;
          }
        }
      }

      forwardInto(pair.vecA, m, scratchA);
      forwardInto(pair.vecB, m, scratchB);

      const deltaZ = scratchA.z - scratchB.z;
      const sig = sigmoid(deltaZ);

      const y = pair.targetProb;
      const loss = -(y * Math.log(sig + 1e-12) + (1.0 - y) * Math.log(1.0 - sig + 1e-12)) * pair.weight;
      epochLoss += loss;

      const dScore = ((sig - y) * pair.weight) / numPairs;

      accumulateGradients(pair.vecA, scratchA, m, dScore, gSkip, gGate, gBGate, gVal, gBVal, gOut);
      accumulateGradients(pair.vecB, scratchB, m, -dScore, gSkip, gGate, gBGate, gVal, gBVal, gOut);

      // 2. Restore masked features
      if (enableDropout) {
        for (let f = 0; f < NUM_RETRIEVAL_FEATURES; f++) {
          if (droppedMask[f] === 1) {
            pair.vecA[f] = savedA[f]!;
            pair.vecB[f] = savedB[f]!;
          }
        }
      }
    }

    stepAdamW(m.W_skip, gSkip, opt.mSkip, opt.vSkip, stepIdx, lr, true);
    stepAdamW(m.W_gate, gGate, opt.mGate, opt.vGate, stepIdx, lr, true);
    stepAdamW(m.W_val,  gVal,  opt.mVal,  opt.vVal,  stepIdx, lr, true);
    stepAdamW(m.W_out,  gOut,  opt.mOut,  opt.vOut,  stepIdx, lr, true);

    stepAdamW(m.b_gate, gBGate, opt.mBGate, opt.vBGate, stepIdx, lr, false);
    stepAdamW(m.b_val,  gBVal,  opt.mBVal,  opt.vBVal,  stepIdx, lr, false);

    return epochLoss / numPairs;
  }

  // =========================================================================
  // STAGE 1: BASIN-HOPPING (PRINT EVERY 1/10 OF TOTAL HOPS)
  // =========================================================================
  console.log(`=============================================================`);
  console.log(`🔍 STAGE 1: BASIN-HOPPING SEARCH FOR BEST INITIAL BASIN (${BH_HOPS} hops)`);
  console.log(`   (Feature Dropout: DISABLED for deterministic basin finding)`);
  console.log(`=============================================================`);

  const currentModel = createInitialModel(rng);
  const bestInitialBasin = cloneModelState(currentModel);
  const trialModel = cloneModelState(currentModel);

  const optBH = createAdamWState();
  const bhStart = performance.now();

  let currentLoss = 0.0;
  for (let e = 1; e <= BH_MINI_EPOCHS; e++) {
    currentLoss = runSingleEpoch(currentModel, optBH, e, BASE_LR, false);
  }
  let bestLoss = currentLoss;
  copyModelState(currentModel, bestInitialBasin);

  console.log(`Seed Basin Loss: ${currentLoss.toFixed(4)}`);
  let acceptedHops = 0;

  // Print every 1/10 of total hops (e.g. 100 / 10 = 10 hops)
  const bhLogInterval = Math.max(1, Math.floor(BH_HOPS / 10));

  for (let hop = 1; hop <= BH_HOPS; hop++) {
    copyModelState(currentModel, trialModel);

    for (let i = 0; i < NUM_FEATURES; i++) {
      trialModel.W_skip[i] += rng.normal(0.0, BH_PERTURB_SIGMA);
    }
    for (let i = 0; i < trialModel.W_gate.length; i++) {
      trialModel.W_gate[i] += rng.normal(0.0, BH_PERTURB_SIGMA);
      trialModel.W_val[i]  += rng.normal(0.0, BH_PERTURB_SIGMA);
    }
    for (let j = 0; j < HIDDEN_NEURONS; j++) {
      trialModel.W_out[j] += rng.normal(0.0, BH_PERTURB_SIGMA);
    }

    resetAdamWState(optBH);
    let trialLoss = 0.0;
    for (let e = 1; e <= BH_MINI_EPOCHS; e++) {
      trialLoss = runSingleEpoch(trialModel, optBH, e, BASE_LR, false);
    }

    const deltaE = trialLoss - currentLoss;

    // Always draw exactly one uniform per hop. Drawing only when deltaE >= 0
    // makes the RNG stream depend on a floating-point comparison, so a tiny
    // difference in the embeddings can shift every subsequent random number.
    const metroProb = deltaE < 0 ? 1.0 : Math.exp(-deltaE / BH_METROPOLIS_TEMP);
    const accept = rng.uniform() < metroProb;

    if (accept) {
      copyModelState(trialModel, currentModel);
      currentLoss = trialLoss;
      acceptedHops++;

      if (trialLoss < bestLoss) {
        bestLoss = trialLoss;
        copyModelState(trialModel, bestInitialBasin);
      }
    }

    // Print every 1/10 of total hops or last hop
    if (hop === 1 || hop % bhLogInterval === 0 || hop === BH_HOPS) {
      process.stdout.write(
        `  Hop [${String(hop).padStart(3, ' ')}/${BH_HOPS}] - ` +
        `Current: ${currentLoss.toFixed(4)} | ` +
        `Best Basin Loss: 🌟 ${bestLoss.toFixed(4)} | ` +
        `Accepted: ${((acceptedHops / hop) * 100).toFixed(0)}%\n`
      );
    }
  }

  const bhSec = ((performance.now() - bhStart) / 1000).toFixed(1);
  console.log(`\n✅ Optimal starting basin located in ${bhSec}s! (Loss: 🌟 ${bestLoss.toFixed(4)})\n`);

  // =========================================================================
  // STAGE 2: MASTER 1,000-EPOCH TRAINING (PRINT EVERY 1/25 OF TOTAL EPOCHS)
  // =========================================================================
  console.log(`=============================================================`);
  console.log(`🏋️  STAGE 2: MASTER ${TOTAL_EPOCHS}-EPOCH ADAMW TRAINING (WARMUP + COSINE)`);
  console.log(`   (Feature Dropout: ENABLED at ${(FEATURE_DROPOUT_PROB * 100).toFixed(0)}% across features 0..${NUM_RETRIEVAL_FEATURES - 1})`);
  console.log(`=============================================================`);

  const masterModel = cloneModelState(bestInitialBasin);
  const masterOpt = createAdamWState();
  const trainStart = performance.now();

  // Print every 1/25 of total epochs (e.g. 1000 / 25 = 40 epochs)
  const epochLogInterval = Math.max(1, Math.floor(TOTAL_EPOCHS / 25));

  for (let epoch = 1; epoch <= TOTAL_EPOCHS; epoch++) {
    const currentLR = getScheduledLR(epoch, TOTAL_EPOCHS, BASE_LR, MIN_LR, WARMUP_EPOCHS);
    const loss = runSingleEpoch(masterModel, masterOpt, epoch, currentLR, true);

    const isMilestone =
      epoch === 1 ||
      epoch === WARMUP_EPOCHS ||
      epoch % epochLogInterval === 0 ||
      epoch === TOTAL_EPOCHS;

    if (isMilestone) {
      const epStr = `Epoch ${String(epoch).padStart(4, ' ')}/${TOTAL_EPOCHS}`;
      const lrStr = `[LR: ${currentLR.toFixed(5)}]`;
      const lossStr = `Loss: ${loss.toFixed(4)}`;

      let phase = '';
      if (epoch < WARMUP_EPOCHS) {
        phase = `(Warmup: ${((epoch / WARMUP_EPOCHS) * 100).toFixed(0)}%)`;
      } else if (epoch === WARMUP_EPOCHS) {
        phase = `(✨ Warmup Complete)`;
      } else if (epoch === TOTAL_EPOCHS) {
        phase = `(🏆 Master Training Finished)`;
      }

      console.log(`  ${epStr} ${lrStr.padEnd(13)} - ${lossStr} ${phase}`);
    }
  }

  const totalMin = ((performance.now() - trainStart) / 60000).toFixed(2);
  console.log(`\n✅ ${TOTAL_EPOCHS} Epochs completed in ${totalMin} minutes.`);

  // =========================================================================
  // STAGE 3: PROBABILITY CALIBRATION (Guo et al., 2017)
  // =========================================================================
  console.log(`\n=============================================================`);
  console.log(`🎛️  STAGE 3: POST-TRAINING TEMPERATURE CALIBRATION`);
  console.log(`=============================================================`);

  const calibPos: CalibrationSample[] = [];
  const calibNegPool: CalibrationSample[] = [];

  for (let q = 0; q < N_QUERIES; q++) {
    for (let d = 0; d < N_DOCS; d++) {
      const rel = getRelevance(q, d);
      const logit = computeLogit(queryFeatures[q]![d]!, masterModel);
      if (rel > 0.0) calibPos.push({ z: logit, label: rel });
      else calibNegPool.push({ z: logit, label: 0.0 });
    }
  }

  const calibRng = createPRNG(4242);
  const calibNeg: CalibrationSample[] = [];
  const negPoolCopy = calibNegPool.slice();
  const negCount = Math.min(calibPos.length, negPoolCopy.length);
  for (let i = 0; i < negCount; i++) {
    const idx = Math.floor(calibRng.uniform() * negPoolCopy.length);
    calibNeg.push(negPoolCopy.splice(idx, 1)[0]!);
  }

  const calibSamples = [...calibPos, ...calibNeg];
  const calibration = fitCalibration(calibSamples);

  console.log(`  Temperature (T): ${calibration.temperature.toFixed(4)} | Calibration Bias: ${calibration.calibrationBias.toFixed(4)}`);
  console.log(`  Brier Score:     ${calibration.brierBefore.toFixed(4)} -> ${calibration.brierAfter.toFixed(4)}`);

  // =========================================================================
  // STAGE 4: VALIDATION RECALL EVALUATION
  // =========================================================================
  console.log(`\n=============================================================`);
  console.log(`📊 STAGE 4: VALIDATION RANKING EVALUATION`);
  console.log(`=============================================================`);

  let strictRank1 = 0;
  let relevantRank1 = 0;

  for (let q = 0; q < N_QUERIES; q++) {
    const target = queryTargetDoc[q]!;
    const scores: { dIdx: number; score: number; rel: number }[] = [];

    for (let d = 0; d < N_DOCS; d++) {
      const logit = computeLogit(queryFeatures[q]![d]!, masterModel);
      const calibratedProb = sigmoid((logit + calibration.calibrationBias) / calibration.temperature);
      scores.push({ dIdx: d, score: calibratedProb, rel: getRelevance(q, d) });
    }
    scores.sort((a, b) => b.score - a.score);

    if (scores[0]!.dIdx === target) {
      strictRank1++;
      relevantRank1++;
    } else if (scores[0]!.rel >= 0.5) {
      relevantRank1++;
    }
  }

  const strictPct = ((strictRank1 / N_QUERIES) * 100).toFixed(1);
  const relevantPct = ((relevantRank1 / N_QUERIES) * 100).toFixed(1);

  console.log(`Strict Exact-Chunk Rank #1: ${strictRank1}/${N_QUERIES} (${strictPct}%)`);
  console.log(`Relevant Section Recall #1: ${relevantRank1}/${N_QUERIES} (${relevantPct}%) (Exact Hit OR ±1 Sister Chunk)`);

  if (relevantRank1 / N_QUERIES < 0.60) {
    console.error('❌ Validation check failed: relevant recall below 60% threshold. Aborting save.');
    process.exit(1);
  }

  // Write binary weights (976 bytes)
  writeModelWeights({
    W_skip: masterModel.W_skip,
    W_gate: masterModel.W_gate,
    b_gate: masterModel.b_gate,
    W_val: masterModel.W_val,
    b_val: masterModel.b_val,
    W_out: masterModel.W_out,
    b_out: masterModel.b_out,
    temperature: calibration.temperature,
    calibrationBias: calibration.calibrationBias
  });

  console.log(`\n=============================================================`);
  console.log(`💾 Saved model weights to weights.bin (976 bytes)`);
  console.log(`🎉 Training pipeline finished successfully!`);
  console.log(`=============================================================\n`);
}

void run();
