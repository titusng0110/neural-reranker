import fs from 'node:fs';
import readline from 'node:readline';

const GOOGLE_PATH = './google-10000-english.txt';
const MIXED_PATH = './top_english_words_mixed_10000.txt';
const OXFORD_PATH = './oxford.txt';
const TIKTOKEN_PATH = './o200k_base.tiktoken';
const OUTPUT_BIN = './simple_vocab.bin';

const CODE_JARGON_BLACKLIST = new Set<string>([
  'async', 'await', 'const', 'void', 'null', 'var', 'let', 'def',
  'lambda', 'eval', 'regex', 'json', 'html', 'http', 'https', 'xml',
  'api', 'npm', 'url', 'uri', 'boolean', 'byte', 'char', 'int', 'sizeof'
]);

function loadWordList(filepath: string): Set<string> | null {
  if (!fs.existsSync(filepath)) {
    console.warn(`Warning: Could not find "${filepath}". Skipping this filter.`);
    return null;
  }
  const lines = fs.readFileSync(filepath, 'utf-8').split('\n');
  const words = new Set<string>();
  for (const line of lines) {
    const clean = line.trim().toLowerCase();
    if (clean.length >= 3 && /^[a-z]+$/.test(clean) && /[aeiou]/i.test(clean)) {
      words.add(clean);
    }
  }
  return words;
}

async function buildTripleFilteredVocab(): Promise<void> {
  console.log('=============================================================');
  console.log('🧼 BUILDING PRISTINE 1,250 ELEMENTARY VOCABULARY (.BIN)');
  console.log('=============================================================');

  console.log('1. Loading frequency word lists...');
  const googleWords = loadWordList(GOOGLE_PATH);
  const mixedWords = loadWordList(MIXED_PATH);

  if (googleWords) console.log(`   - Loaded ${googleWords.size.toLocaleString()} words from Google 10,000`);
  if (mixedWords) console.log(`   - Loaded ${mixedWords.size.toLocaleString()} words from Mixed 10,000`);

  console.log('\n2. Extracting clean headwords from Oxford dictionary...');
  const oxfordHeadwords = new Set<string>();

  const oxfordRl = readline.createInterface({
    input: fs.createReadStream(OXFORD_PATH),
    crlfDelay: Infinity,
  });

  for await (const line of oxfordRl) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const match = trimmed.match(/^([A-Z][a-z]{2,})/);
    if (!match) continue;

    const word = match[1]!.toLowerCase();

    if (
      word.length >= 4 &&
      /^[a-z]+$/.test(word) &&
      /[aeiou]/i.test(word) &&
      !CODE_JARGON_BLACKLIST.has(word)
    ) {
      oxfordHeadwords.add(word);
    }
  }
  console.log(`   - Extracted ${oxfordHeadwords.size.toLocaleString()} Oxford headwords`);

  console.log('\n3. Intersecting Oxford headwords with top-10k lists...');
  const verifiedCandidates = new Set<string>();

  for (const word of oxfordHeadwords) {
    const inGoogle = googleWords ? googleWords.has(word) : true;
    const inMixed = mixedWords ? mixedWords.has(word) : true;

    if (inGoogle || inMixed) {
      verifiedCandidates.add(word);
    }
  }
  console.log(`   - Verified ${verifiedCandidates.size.toLocaleString()} mutually confirmed human words.`);

  console.log(`\n4. Ranking candidates via ${TIKTOKEN_PATH}...`);
  const candidates = new Map<string, number>();

  const tiktokenRl = readline.createInterface({
    input: fs.createReadStream(TIKTOKEN_PATH),
    crlfDelay: Infinity,
  });

  for await (const line of tiktokenRl) {
    const spaceIdx = line.indexOf(' ');
    if (spaceIdx === -1) continue;

    const b64 = line.slice(0, spaceIdx);
    const rank = parseInt(line.slice(spaceIdx + 1), 10);

    try {
      const raw = Buffer.from(b64, 'base64').toString('utf-8');
      const clean = raw.trim().toLowerCase();

      if (verifiedCandidates.has(clean)) {
        if (!candidates.has(clean) || rank < (candidates.get(clean) as number)) {
          candidates.set(clean, rank);
        }
      }
    } catch {}
  }

  const sortedWords = Array.from(candidates.entries())
    .sort((a, b) => a[1] - b[1])
    .slice(0, 1250)
    .map(([word]) => word);

  // =========================================================================
  // BINARY PACKING: [uint32LE wordCount] + N * [uint8 length + utf8 bytes]
  // =========================================================================
  const wordBuffers = sortedWords.map((w) => Buffer.from(w, 'utf-8'));
  let totalDataBytes = 4; // uint32 wordCount
  for (const wb of wordBuffers) {
    totalDataBytes += 1 + wb.length; // 1 byte length prefix + utf8 data
  }

  const outBuf = Buffer.alloc(totalDataBytes);
  outBuf.writeUInt32LE(wordBuffers.length, 0);

  let offset = 4;
  for (const wb of wordBuffers) {
    outBuf.writeUInt8(wb.length, offset);
    offset += 1;
    wb.copy(outBuf, offset);
    offset += wb.length;
  }

  fs.writeFileSync(OUTPUT_BIN, outBuf);

  console.log(`\n=============================================================`);
  console.log(`🎉 SUCCESS! Saved ${sortedWords.length} words to binary "${OUTPUT_BIN}" (${(totalDataBytes / 1024).toFixed(2)} KB)`);
  console.log(`=============================================================`);
}

buildTripleFilteredVocab();
