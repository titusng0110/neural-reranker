import { performance } from 'node:perf_hooks';

// Windows PC LAN endpoint
const PC_IP = '192.168.50.172';
const PORT = 9080;
const LLM_ENDPOINT = `http://${PC_IP}:${PORT}/v1/chat/completions`;
const HEALTH_ENDPOINT = `http://${PC_IP}:${PORT}/health`;

// Sample statutory passage (Cap. 622, Section 465)
const SAMPLE_LEGAL_CHUNK = `
465. Duty to exercise reasonable care, skill and diligence
(1) A director of a company must exercise reasonable care, skill and diligence.
(2) Reasonable care, skill and diligence mean the care, skill and diligence that would be exercised by a reasonably diligent person with—
(a) the general knowledge, skill and experience that may reasonably be expected of a person carrying out the functions carried out by the director in relation to the company; and
(b) the general knowledge, skill and experience that the director has.
`.trim();

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

async function testConnection(): Promise<void> {
  console.log('=============================================================');
  console.log(`📡 TESTING LAN LLM SERVER: ${PC_IP}:${PORT}`);
  console.log('=============================================================');

  // 1. Health check
  process.stdout.write(`1. Pinging llama-server on PC... `);
  try {
    const healthRes = await fetch(HEALTH_ENDPOINT, { signal: AbortSignal.timeout(3000) });
    if (healthRes.ok) {
      console.log('✅ Connected!');
    } else {
      console.log(`⚠️ Server replied with status: ${healthRes.status}`);
    }
  } catch {
    console.log('❌ Connection failed!');
    console.error(`\nCould not reach ${HEALTH_ENDPOINT}. Check your PC IP and Windows Firewall.`);
    process.exit(1);
  }

  // 2. Query generation test
  console.log('\n2. Testing 3-Persona Synthetic Query Generation on Gemma 12B...');
  console.log('-------------------------------------------------------------');
  console.log('Input Legal Chunk:');
  console.log(`"${SAMPLE_LEGAL_CHUNK.slice(0, 140)}..."\n`);

  const systemPrompt = `You are an expert AI data synthesizer for a statutory search engine.
Given a legal text chunk, generate exactly 3 distinct search queries that this chunk answers:
1. "layman": High vibiness query using conversational language, everyday words, no legal jargon.
2. "barrister": High BM25 query using formal legal terminology and doctrine keywords.
3. "citation": Statutory citation query focusing on section numbers, sub-clauses, or exact legal titles.

Return ONLY a valid JSON object with the keys "layman", "barrister", and "citation". Do not wrap in markdown or add commentary.`;

  const userPrompt = `Legal Chunk:\n"""\n${SAMPLE_LEGAL_CHUNK}\n"""`;

  const requestBody = {
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ],
    temperature: 0.1,
    max_tokens: 800
  };

  const start = performance.now();

  try {
    const res = await fetch(LLM_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(30000)
    });

    if (!res.ok) {
      throw new Error(`llama-server returned HTTP ${res.status}: ${await res.text()}`);
    }

    const elapsed = performance.now() - start;
    const data = (await res.json()) as LlamaResponse;
    const completionTokens = data.usage?.completion_tokens ?? 0;
    const tokensPerSec = (completionTokens / (elapsed / 1000)).toFixed(1);

    console.log('🎉 Generation Finished!\n');
    console.log(`⏱️  Latency:      ${elapsed.toFixed(1)} ms`);
    console.log(`⚡ Output Speed: ${tokensPerSec} tokens/sec`);
    console.log(`📊 Tokens:       ${completionTokens} tokens generated`);

    let rawContent = data.choices[0]?.message?.content || '';

    console.log('\n-------------------------------------------------------------');
    console.log('RAW LLM OUTPUT FROM GEMMA:');
    console.log('-------------------------------------------------------------');
    console.log(rawContent);
    console.log('-------------------------------------------------------------');

    // Strip Gemma thinking channel if present: <|channel>thought ... <channel|>
    if (rawContent.includes('<channel|>')) {
      const parts = rawContent.split('<channel|>');
      rawContent = parts[parts.length - 1]!.trim();
    }

    // Extract the JSON object or array
    const jsonMatch = rawContent.match(/\{[\s\S]*\}|\[[\s\S]*\]/);

    if (jsonMatch) {
      try {
        const parsed = JSON.parse(jsonMatch[0]);
        console.log('\n✅ SUCCESSFULLY PARSED JSON:');
        console.dir(parsed, { depth: null });

        if (parsed.layman && parsed.barrister && parsed.citation) {
          console.log('\n-------------------------------------------------------------');
          console.log('RETRIEVAL QUERIES:');
          console.log('-------------------------------------------------------------');
          console.log(`🗣️  [Layman / High Vibe]:\n   "${parsed.layman}"\n`);
          console.log(`⚖️  [Barrister / High BM25]:\n   "${parsed.barrister}"\n`);
          console.log(`📜 [Citation / Symbol Shingles]:\n   "${parsed.citation}"`);
        }
      } catch (err: any) {
        console.log('\n⚠️ Found JSON regex match, but JSON.parse failed:', err.message);
      }
    } else {
      console.log('\n⚠️ No JSON substring pattern found in the output.');
    }

    console.log('\n=============================================================\n');

  } catch (err: any) {
    console.error('❌ Request failed:', err.message);
  }
}

void testConnection();
