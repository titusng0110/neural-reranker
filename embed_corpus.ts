import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { EMBEDDING_DIM, fetchEmbeddings } from './cosine_similarity.ts';

// ============================================================================
// CONFIGURATION
// ============================================================================
const BATCH_SIZE = 16;
const TARGET_CHUNK_SIZE = 2000;

const rawInputArg = process.argv[2];
if (!rawInputArg) {
  console.log('\n❌ Error: Please specify an input text file to embed.');
  console.log('Usage:');
  console.log('  npx tsx embed_corpus.ts <path/to/corpus.txt> [output_embeddings.bin]\n');
  console.log('Examples:');
  console.log('  npx tsx embed_corpus.ts ./cap221.txt');
  console.log('  npx tsx embed_corpus.ts ./cap221.txt ./my_vectors.bin\n');
  process.exit(1);
}

const INPUT_FILE: string = rawInputArg;
if (!fs.existsSync(INPUT_FILE)) {
  console.error(`\n❌ Error: Cannot find input file "${INPUT_FILE}".`);
  process.exit(1);
}

const defaultBinName = `${path.basename(INPUT_FILE, path.extname(INPUT_FILE))}_embeddings.bin`;
const OUTPUT_BIN: string = process.argv[3] || defaultBinName;

// ============================================================================
// 1. SMART BOUNDARY CHUNKER (~2,000 CHARACTERS)
// ============================================================================
export function chunkText(text: string, targetSize: number = TARGET_CHUNK_SIZE): string[] {
  const chunks: string[] = [];
  let i = 0;
  const len = text.length;

  while (i < len) {
    let end = Math.min(i + targetSize, len);

    if (end < len) {
      const nextSpace = text.indexOf(' ', end);
      const nextNewline = text.indexOf('\n', end);
      let snap = -1;

      if (nextSpace !== -1 && nextNewline !== -1) {
        snap = Math.min(nextSpace, nextNewline);
      } else {
        snap = nextSpace !== -1 ? nextSpace : nextNewline;
      }

      if (snap !== -1 && snap - end < 200) {
        end = snap;
      }
    }

    const chunk = text.slice(i, end).trim();
    if (chunk.length > 150) {
      chunks.push(chunk);
    }
    i = end;
  }

  return chunks;
}

// ============================================================================
// 2. MAIN RUNNER
// ============================================================================
async function run(): Promise<void> {
  console.log(`\n=============================================================`);
  console.log(`🧠 UNIVERSAL CORPUS EMBEDDER (${EMBEDDING_DIM}-DIM ON PORT 9000)`);
  console.log(`=============================================================`);
  console.log(`Input Corpus: "${INPUT_FILE}"`);
  console.log(`Output File:  "${OUTPUT_BIN}"`);

  const rawText: string = fs.readFileSync(INPUT_FILE, 'utf-8');
  const chunks = chunkText(rawText, TARGET_CHUNK_SIZE);
  const totalChunks = chunks.length;
  console.log(`Extracted ${totalChunks.toLocaleString()} chunks (~${TARGET_CHUNK_SIZE} chars each).\n`);

  const start = performance.now();
  const allVectors: Float32Array[] = [];
  const dim = EMBEDDING_DIM;

  for (let i = 0; i < totalChunks; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);
    const progress = Math.min(totalChunks, i + BATCH_SIZE);
    process.stdout.write(`\rEncoding chunks [${i + 1}-${progress}/${totalChunks}]... `);

    const vectors = await fetchEmbeddings(batch, 'document');
    allVectors.push(...vectors);
  }

  const elapsedSec = ((performance.now() - start) / 1000).toFixed(1);
  console.log(`\n✅ Embedded all ${totalChunks} chunks in ${elapsedSec}s!`);

  // Write compact binary: [uint32 count, uint32 dim, ...Float32LE values]
  const totalFloats = totalChunks * dim;
  const outBuf = Buffer.alloc(8 + totalFloats * 4);

  outBuf.writeUInt32LE(totalChunks, 0);
  outBuf.writeUInt32LE(dim, 4);

  let offset = 8;
  for (let i = 0; i < totalChunks; i++) {
    const vec = allVectors[i]!;
    for (let d = 0; d < dim; d++) {
      outBuf.writeFloatLE(vec[d]!, offset);
      offset += 4;
    }
  }

  fs.writeFileSync(OUTPUT_BIN, outBuf);
  console.log(`💾 Saved ${totalChunks} vectors to "${OUTPUT_BIN}" (${(outBuf.length / (1024 * 1024)).toFixed(2)} MB).\n`);
}

void run();
