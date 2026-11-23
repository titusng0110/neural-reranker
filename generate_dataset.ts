import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { performance } from 'node:perf_hooks';

// ============================================================================
// CONFIGURATION & TYPE DEFINITIONS
// ============================================================================
const PC_IP = '192.168.50.172';
const PORT = 9080;
const LLM_ENDPOINT = `http://${PC_IP}:${PORT}/v1/chat/completions`;

// Target character size per chunk
const TARGET_CHUNK_SIZE = 2000;

interface LlamaResponse {
  choices: {
    message: {
      content: string;
    };
  }[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// CLI Arguments: tsx generate_dataset.ts <input_file.txt> [output_file.jsonl]
const rawInputArg = process.argv[2];
const OUTPUT_FILE: string = process.argv[3] || './training_data.jsonl';

if (!rawInputArg) {
  console.log('\n❌ Error: Please specify an input text file.');
  console.log('Usage:');
  console.log('  npx tsx generate_dataset.ts <path/to/document.txt> [output_dataset.jsonl]\n');
  process.exit(1);
}

const INPUT_FILE: string = rawInputArg;

if (!fs.existsSync(INPUT_FILE)) {
  console.error(`\n❌ Error: Cannot find input file <q>"${INPUT_FILE}"</q>.`);
  process.exit(1);
}

const FILE_BASENAME: string = path.basename(INPUT_FILE, path.extname(INPUT_FILE))
  .toLowerCase()
  .replace(/[^a-z0-9_-]/g, '_');

// ============================================================================
// 1. SMART BOUNDARY CHUNKER (~2,000 CHARACTERS)
// ============================================================================
function chunkText(text: string, targetSize: number = TARGET_CHUNK_SIZE): string[] {
  const chunks: string[] = [];
  let i = 0;
  const len = text.length;

  while (i < len) {
    let end = Math.min(i + targetSize, len);

    if (end < len) {
      const lookahead = text.slice(end, Math.min(end + 150, len));
      const match = lookahead.match(/(\.\s+|\n\n|\n|[!?]\s+)/);

      if (match && match.index !== undefined) {
        end += match.index + match[0].length;
      } else {
        const nextSpace = text.indexOf(' ', end);
        if (nextSpace !== -1 && nextSpace - end < 100) {
          end = nextSpace + 1;
        }
      }
    }

    const chunk = text.slice(i, end).trim();
    if (chunk.length > 120) {
      chunks.push(chunk);
    }
    i = end;
  }

  return chunks;
}

// ============================================================================
// 2. CHECK ALREADY PROCESSED CHUNKS (RESUME CAPABILITY)
// ============================================================================
async function getProcessedDocIds(filepath: string): Promise<Set<string>> {
  const processed = new Set<string>();
  if (!fs.existsSync(filepath)) return processed;

  const rl = readline.createInterface({
    input: fs.createReadStream(filepath),
    crlfDelay: Infinity
  });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed.docId) processed.add(parsed.docId);
    } catch {}
  }

  return processed;
}

// ============================================================================
// 3. ROBUST JSON EXTRACTOR (IMMUNE TO THINKING TOKENS)
// ============================================================================
function extractJsonFromResponse(raw: string): any | null {
  // Strategy 1: Look for markdown ```json ... ``` codeblocks (grab the LAST one, which is the final output)
  const codeBlocks = [...raw.matchAll(/```(?:json)?\s*([\s\S]*?)\s*```/gi)];
  if (codeBlocks.length > 0) {
    const lastBlock = codeBlocks[codeBlocks.length - 1]![1]!.trim();
    try {
      return JSON.parse(lastBlock);
    } catch {}
  }

  // Strategy 2: Strip thinking tokens if present (<channel|> or </think>)
  let clean = raw;
  if (clean.includes('<channel|>')) {
    clean = clean.split('<channel|>').pop()!;
  }
  if (clean.includes('</think>')) {
    clean = clean.split('</think>').pop()!;
  }

  // Strategy 3: Find the outermost braces { ... } after thinking
  const firstBrace = clean.indexOf('{');
  const lastBrace = clean.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    const candidate = clean.slice(firstBrace, lastBrace + 1);
    try {
      return JSON.parse(candidate);
    } catch {}
  }

  return null;
}

// ============================================================================
// 4. MAIN RUNNER
// ============================================================================
async function run(): Promise<void> {
  console.log('=============================================================');
  console.log(`🏭 UNIVERSAL DATASET GENERATOR (via RTX 3080: ${PC_IP}:${PORT})`);
  console.log('=============================================================');
  console.log(`Input Source:  "${INPUT_FILE}"`);
  console.log(`Output Target: "${OUTPUT_FILE}"`);

  const rawText: string = fs.readFileSync(INPUT_FILE, 'utf-8');
  const chunks = chunkText(rawText, TARGET_CHUNK_SIZE);
  console.log(`Total Text:    ${(rawText.length / 1024).toFixed(1)} KB (${rawText.length.toLocaleString()} chars)`);
  console.log(`Generated:     ${chunks.length.toLocaleString()} chunks (~${TARGET_CHUNK_SIZE} chars each)\n`);

  const processedDocIds = await getProcessedDocIds(OUTPUT_FILE);
  const alreadyDoneForThisFile = chunks.filter((_, idx) =>
    processedDocIds.has(`${FILE_BASENAME}-chunk-${idx + 1}`)
  ).length;

  if (alreadyDoneForThisFile > 0) {
    console.log(`⏩ Resuming: ${alreadyDoneForThisFile}/${chunks.length} chunks already processed for this document.`);
  }

  const systemPrompt = `You are an expert AI retrieval benchmark generator.
Given any source text chunk, think carefully and generate exactly 3 distinct search queries that this specific chunk best answers:

1. "layman": High-vibiness natural conversational query. Everyday simple words, zero specialized jargon, phrased like a curious beginner asking Google.
2. "expert": High-BM25 specialist query. Dense formal terminology, industry-standard jargon, technical mechanisms, and professional keywords.
3. "reference": Structural, numeric, or exact entity query. Focuses on specific names, product models, citations, formula names, or precise key phrases.

Output your final result strictly wrapped in a \`\`\`json markdown block like this:
\`\`\`json
{
  "layman": "...",
  "expert": "...",
  "reference": "..."
}
\`\`\`
Do not write anything after the closing \`\`\`.`;

  let newlyGenerated = 0;
  const startTime = performance.now();

  for (let idx = 0; idx < chunks.length; idx++) {
    const docId = `${FILE_BASENAME}-chunk-${idx + 1}`;
    const chunkContent = chunks[idx]!;

    if (processedDocIds.has(docId)) {
      continue;
    }

    const preview = chunkContent.replace(/\s+/g, ' ').slice(0, 50);
    const progress = `[${idx + 1}/${chunks.length}]`;
    process.stdout.write(`${progress.padEnd(11)} ${docId.slice(0, 22).padEnd(23)} "${preview}..." `);

    const userPrompt = `Passage:\n"""\n${chunkContent}\n"""`;

    try {
      const qStart = performance.now();
      const res = await fetch(LLM_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
          ],
          temperature: 0.2,
          max_tokens: 6144 // Uncapped to allow full deep reasoning + full JSON output
        }),
        signal: AbortSignal.timeout(90000)
      });

      if (!res.ok) {
        console.log(`❌ HTTP ${res.status}`);
        continue;
      }

      const data = (await res.json()) as LlamaResponse;
      const rawContent: string = data.choices[0]?.message?.content || '';

      const parsed = extractJsonFromResponse(rawContent);

      if (!parsed) {
        console.log(`⚠️ Invalid JSON (Response preview: "${rawContent.slice(-80).replace(/\s+/g, ' ')}")`);
        continue;
      }

      const layman = parsed.layman || parsed.conversational;
      const expert = parsed.expert || parsed.barrister || parsed.technical;
      const reference = parsed.reference || parsed.citation || parsed.entity;

      if (layman && expert && reference) {
        const pairs = [
          { query: String(layman).trim(), passage: chunkContent, docId, persona: 'layman' },
          { query: String(expert).trim(), passage: chunkContent, docId, persona: 'expert' },
          { query: String(reference).trim(), passage: chunkContent, docId, persona: 'reference' }
        ];

        const jsonlBlock = pairs.map((p) => JSON.stringify(p)).join('\n') + '\n';
        fs.appendFileSync(OUTPUT_FILE, jsonlBlock, 'utf-8');

        newlyGenerated += 3;
        processedDocIds.add(docId);

        const sec = ((performance.now() - qStart) / 1000).toFixed(1);
        const tokens = data.usage?.completion_tokens ?? 0;
        const speed = (tokens / (Number(sec) || 1)).toFixed(0);
        console.log(`✅ (${sec}s | ${speed} t/s | ${tokens} tok)`);
      } else {
        console.log(`⚠️ Missing keys in JSON`);
      }
    } catch (err: any) {
      console.log(`❌ Error: ${err.message}`);
    }
  }

  const totalMin = ((performance.now() - startTime) / 60000).toFixed(1);
  console.log('\n=============================================================');
  console.log(`🎉 COMPLETED! Generated ${newlyGenerated} pairs in ${totalMin} mins.`);
  console.log(`All data successfully appended to: <q>"${OUTPUT_FILE}"</q>`);
  console.log('=============================================================\n');
}

void run();
