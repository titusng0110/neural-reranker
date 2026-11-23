import fs from 'node:fs';
import readline from 'node:readline';

const INPUT_FILE = './o200k_base.tiktoken';
const OUTPUT_BIN = './bpe_vocab.bin';

const ENGLISH_MAX_RANK = 11000;
const SYMBOL_MAX_RANK = 50000;
const MULTILINGUAL_REGEX = /^[\p{L}\p{M}\p{N}\p{P}\p{S}]+$/u;
const PURE_ENGLISH_REGEX = /^[a-z]+$/;
const HAS_NUM_OR_SYMBOL = /[0-9\p{P}\p{S}]/u;

async function build(): Promise<void> {
  if (!fs.existsSync(INPUT_FILE)) {
    console.error(`Error: Could not find "${INPUT_FILE}".`);
    process.exit(1);
  }

  console.log(`Processing ${INPUT_FILE} (English < ${ENGLISH_MAX_RANK}, Symbols < ${SYMBOL_MAX_RANK})...`);
  const tokenRanks = new Map<string, number>();

  const fileStream = fs.createReadStream(INPUT_FILE);
  const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

  for await (const line of rl) {
    const spaceIdx = line.indexOf(' ');
    if (spaceIdx === -1) continue;

    const b64 = line.slice(0, spaceIdx);
    const rank = parseInt(line.slice(spaceIdx + 1), 10);

    try {
      const raw = Buffer.from(b64, 'base64').toString('utf-8');
      const clean = raw.trim().normalize('NFC').toLowerCase();

      if (clean.length > 0 && Buffer.byteLength(clean) < 256 && MULTILINGUAL_REGEX.test(clean)) {
        if (PURE_ENGLISH_REGEX.test(clean) && rank >= ENGLISH_MAX_RANK) continue;
        if (HAS_NUM_OR_SYMBOL.test(clean) && rank >= SYMBOL_MAX_RANK) continue;

        if (!tokenRanks.has(clean) || rank < (tokenRanks.get(clean) as number)) {
          tokenRanks.set(clean, rank);
        }
      }
    } catch {}
  }

  const items = Array.from(tokenRanks.entries()).map(([token, rank]) => ({
    buf: Buffer.from(token, 'utf-8'),
    origRank: rank,
  }));

  items.sort((a, b) => a.origRank - b.origRank);

  const count = items.length;
  let totalDataBytes = 4;
  for (const item of items) {
    totalDataBytes += 4 + item.buf.length;
  }

  const outBuf = Buffer.alloc(totalDataBytes);
  outBuf.writeUInt32LE(count, 0);

  let offset = 4;
  for (let denseRank = 0; denseRank < count; denseRank++) {
    const item = items[denseRank]!;
    const len = item.buf.length;

    outBuf.writeUIntBE(denseRank, offset, 3);
    outBuf[offset + 3] = len;
    offset += 4;

    item.buf.copy(outBuf, offset);
    offset += len;
  }

  fs.writeFileSync(OUTPUT_BIN, outBuf);

  console.log(`\n--- Optimization Summary ---`);
  console.log(`Tokens kept:         ${count.toLocaleString()}`);
  console.log(`Compact Binary Size: ${(totalDataBytes / 1024).toFixed(1)} KB`);
  console.log(`Saved to:            ${OUTPUT_BIN}`);
}

build();
