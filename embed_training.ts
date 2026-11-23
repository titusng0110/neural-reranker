import fs from 'node:fs';
import readline from 'node:readline';

// Automatically detect dataset file or accept CLI argument
const DATASET_PATH =
  process.argv[2] ||
  (fs.existsSync('./training_data.jsonl')
    ? './training_data.jsonl'
    : './training_dataset.jsonl');

const OUTPUT_BIN = process.argv[3] || './training_embeddings.bin';
const LLAMA_SERVER_URL = 'http://127.0.0.1:9000/v1/embeddings';

// Safe batch sizes: queries are short (16/batch), passages are 2000 chars (4/batch)
const QUERY_BATCH_SIZE = 16;
const PASSAGE_BATCH_SIZE = 4;

function l2Normalize(vec: number[]): number[] {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i]! * vec[i]!;
  const norm = Math.sqrt(sum) || 1e-12;
  for (let i = 0; i < vec.length; i++) vec[i] = vec[i]! / norm;
  return vec;
}

interface LlamaEmbeddingResponse {
  data: { index: number; embedding: number[] }[];
}

async function run(): Promise<void> {
  if (!fs.existsSync(DATASET_PATH)) {
    console.error(`\n❌ Error: Cannot find dataset file at "${DATASET_PATH}".`);
    console.error('Please make sure your training jsonl file exists.\n');
    process.exit(1);
  }

  console.log(`\n=============================================================`);
  console.log(`🧠 EMBEDDING TRAINING PAIRS TO BINARY (.BIN)`);
  console.log(`=============================================================`);
  console.log(`Source Dataset: "${DATASET_PATH}"`);
  console.log(`Target Binary:  "${OUTPUT_BIN}"\n`);

  const queries: string[] = [];
  const passages: string[] = [];

  const rl = readline.createInterface({
    input: fs.createReadStream(DATASET_PATH),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('//') || trimmed.startsWith('#')) continue;
    try {
      const parsed = JSON.parse(trimmed);
      const q = parsed.query || parsed.q;
      const p = parsed.passage || parsed.rule || parsed.text || parsed.document;
      if (q && p) {
        queries.push(String(q).trim());
        passages.push(String(p).trim());
      }
    } catch {}
  }

  const totalPairs = queries.length;
  console.log(`Loaded ${totalPairs.toLocaleString()} training pairs.\n`);

  if (totalPairs === 0) {
    console.error('❌ Error: No valid pairs found in dataset.');
    process.exit(1);
  }

  let dim = 768;

  // 1. Embed Queries
  console.log(`Embedding ${totalPairs} queries (batch size ${QUERY_BATCH_SIZE})...`);
  const queryEmbeddings: number[][] = [];

  for (let i = 0; i < totalPairs; i += QUERY_BATCH_SIZE) {
    const batch = queries.slice(i, i + QUERY_BATCH_SIZE).map((q) => `Query: ${q}`);
    process.stdout.write(`\rQuery Progress: [${Math.min(totalPairs, i + QUERY_BATCH_SIZE)}/${totalPairs}] `);

    const res = await fetch(LLAMA_SERVER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: batch }),
    });

    if (!res.ok) {
      throw new Error(`llama-server (port 9000) returned HTTP ${res.status}: ${await res.text()}`);
    }

    const data = (await res.json()) as LlamaEmbeddingResponse;
    if (!data || !Array.isArray(data.data)) {
      throw new Error(`Invalid response from llama-server: ${JSON.stringify(data)}`);
    }

    const sorted = data.data.sort((a, b) => a.index - b.index);
    for (const item of sorted) {
      const vec = l2Normalize(item.embedding);
      dim = vec.length;
      queryEmbeddings.push(vec);
    }
  }

  // 2. Embed Passages
  console.log(`\n\nEmbedding ${totalPairs} passages (safe batch size ${PASSAGE_BATCH_SIZE})...`);
  const passageEmbeddings: number[][] = [];

  for (let i = 0; i < totalPairs; i += PASSAGE_BATCH_SIZE) {
    const batch = passages.slice(i, i + PASSAGE_BATCH_SIZE).map((p) => `Document: ${p}`);
    process.stdout.write(`\rPassage Progress: [${Math.min(totalPairs, i + PASSAGE_BATCH_SIZE)}/${totalPairs}] `);

    const res = await fetch(LLAMA_SERVER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: batch }),
    });

    if (!res.ok) {
      throw new Error(`llama-server (port 9000) returned HTTP ${res.status}: ${await res.text()}`);
    }

    const data = (await res.json()) as LlamaEmbeddingResponse;
    if (!data || !Array.isArray(data.data)) {
      throw new Error(`Invalid response from llama-server: ${JSON.stringify(data)}`);
    }

    const sorted = data.data.sort((a, b) => a.index - b.index);
    for (const item of sorted) {
      passageEmbeddings.push(l2Normalize(item.embedding));
    }
  }

  // 3. Binary Packing:
  // [0..3]: uint32LE countQueries
  // [4..7]: uint32LE countPassages
  // [8..11]: uint32LE dimension
  // [12..]: Float32LE data
  const countQ = queryEmbeddings.length;
  const countP = passageEmbeddings.length;
  const totalFloats = (countQ + countP) * dim;
  const outBuf = Buffer.alloc(12 + totalFloats * 4);

  outBuf.writeUInt32LE(countQ, 0);
  outBuf.writeUInt32LE(countP, 4);
  outBuf.writeUInt32LE(dim, 8);

  let offset = 12;
  for (let i = 0; i < countQ; i++) {
    const vec = queryEmbeddings[i]!;
    for (let d = 0; d < dim; d++) {
      outBuf.writeFloatLE(vec[d]!, offset);
      offset += 4;
    }
  }

  for (let i = 0; i < countP; i++) {
    const vec = passageEmbeddings[i]!;
    for (let d = 0; d < dim; d++) {
      outBuf.writeFloatLE(vec[d]!, offset);
      offset += 4;
    }
  }

  fs.writeFileSync(OUTPUT_BIN, outBuf);

  console.log(`\n\n💾 Saved ${countQ} queries & ${countP} passages to "${OUTPUT_BIN}" (${(outBuf.length / 1024).toFixed(1)} KB)!`);
  console.log(`Ready for instant zero-copy training.\n`);
}

void run();
