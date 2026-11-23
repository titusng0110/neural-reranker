import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { SearchEngine } from './search.ts';

const LISTINGRULES_FILE = './listingrules.txt';
const DENSE_BIN = './listingrules_embeddings.bin';
const TARGET_CHUNK_SIZE = 2000;

interface TestQuery {
  title: string;
  query: string;
}

interface SearchResult {
  id: string;
  text: string;
  score: number;
  probability: number;
  denseSim: number;
  features: number[];
}

// Exact 2000-character chunker matching embed_corpus.ts
function chunkText(text: string, targetSize: number = TARGET_CHUNK_SIZE): string[] {
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
// HKEX LISTING RULES MULTI-PERSONA BENCHMARK SUITE (22 QUERIES)
// ============================================================================
const TEST_QUERIES: readonly TestQuery[] = [
  // --------------------------------------------------------------------------
  // CATEGORY A: LAYMAN / HIGH-VIBINESS (Tests Vibiness Prior & Jina Dense)
  // Conversational questions, no chapter/rule numbers, zero legal jargon.
  // --------------------------------------------------------------------------
  {
    title: 'A1. Connected Deal with Boss (Chapter 14A)',
    query: 'can a company boss or major shareholder secretly buy an asset from the business without telling the public'
  },
  {
    title: 'A2. Pre-Revenue Biotech Listing (Chapter 18A)',
    query: 'can a biotech medical company that has never made any profit or revenue list its shares on the stock exchange'
  },
  {
    title: 'A3. Dead Shell Company Delisting (Rule 13.24)',
    query: 'what happens to a listed company if its business completely dries up and it becomes an empty shell with no real operations'
  },
  {
    title: 'A4. Director Insider Trading Blackout (Model Code / App C3)',
    query: 'how many days before annual financial results are released is a director completely forbidden from buying or selling company shares'
  },
  {
    title: 'A5. Public Float Requirement (Rule 8.08)',
    query: 'how much percentage of a company shares must be owned by ordinary public retail investors instead of the founders'
  },
  {
    title: 'A6. Independent Board Members & Accounting Qualification (Rule 3.10)',
    query: 'how many independent outside directors does a listed board need to have and does one of them have to be an accountant'
  },

  // --------------------------------------------------------------------------
  // CATEGORY B: PRACTITIONER / FINANCIAL DOCTRINE (Tests BM25 & BPE IDF)
  // Formal terms of art, investment banking concepts, zero rule numbers.
  // --------------------------------------------------------------------------
  {
    title: 'B1. Reverse Takeover / Backdoor Listing (Rule 14.06B)',
    query: 'acquisition of assets by listed issuer constituting very substantial acquisition involving change in control or within 36 months of change in control'
  },
  {
    title: 'B2. The Five Percentage Size Tests (Rule 14.07)',
    query: 'calculation of the five percentage ratios assets ratio profits ratio revenue ratio consideration ratio and equity capital ratio for classification of notifiable transactions'
  },
  {
    title: 'B3. Continuing Connected Transactions & Annual Caps (Rule 14A.51 / 14A.53)',
    query: 'written agreement for continuing connected transactions setting annual monetary caps and requirement for independent non-executive directors annual review'
  },
  {
    title: 'B4. General Mandate Placing Limit & Discount Restriction (Rule 13.36)',
    query: 'general mandate to allot and issue shares not exceeding twenty percent of total issued shares and placing price discount restriction of twenty percent to benchmarked price'
  },
  {
    title: 'B5. Basic Listing Profit Requirement (Rule 8.05(1))',
    query: 'financial eligibility criteria under profit test requiring aggregate profit attributable to shareholders of not less than eighty million Hong Kong dollars over three consecutive financial years'
  },
  {
    title: 'B6. Weighted Voting Rights Beneficiary Restrictions (Chapter 8A)',
    query: 'weighted voting rights structure where beneficiaries must be executive directors with voting power restricted to maximum ten to one ratio and non-transferable'
  },

  // --------------------------------------------------------------------------
  // CATEGORY C: REFERENCE & EXACT CITATION (Tests Symbol Shingles & Citations)
  // Precise statutory codes, chapter numbers, sub-clauses, and rules.
  // --------------------------------------------------------------------------
  {
    title: 'C1. Connected Transaction Announcement & Approval under Rule 14A.35 / 14A.36',
    query: 'Rule 14A.35 announcement and Rule 14A.36 independent shareholders approval requirements for connected transactions'
  },
  {
    title: 'C2. Share Buy-back Restrictions under Rule 10.06(1) & (2)',
    query: 'Rule 10.06 share repurchases on exchange and blackout period prohibiting buy-backs within 30 days immediately preceding results announcement'
  },
  {
    title: 'C3. Pre-Revenue Biotech Criteria under Chapter 18A Rule 18A.03',
    query: 'Chapter 18A Rule 18A.03 biotech company eligibility Phase I clinical trials on Core Product and minimum market capitalization of HK$1.5 billion'
  },
  {
    title: 'C4. Minimum 25% Public Float under Rule 8.08(1)(a)',
    query: 'Rule 8.08(1)(a) minimum public float requirement at least 25% of the issuers total number of issued shares held by the public'
  },
  {
    title: 'C5. Sufficient Operations Requirement under Rule 13.24',
    query: 'Rule 13.24 issuer shall carry out a business with a sufficient level of operations and assets of sufficient value to support operations to warrant continued listing'
  },
  {
    title: 'C6. Minimum INED Composition under Rule 3.10 and Rule 3.10A',
    query: 'Rule 3.10 and Rule 3.10A board composition requiring at least three independent non-executive directors representing at least one-third of the board'
  },

  // --------------------------------------------------------------------------
  // CATEGORY D: AMBIGUOUS & HARD ADVERSARIAL QUERIES
  // Distinguishing subtle statutory boundaries and percentage thresholds.
  // --------------------------------------------------------------------------
  {
    title: 'D1. Discloseable (5%) vs Major (25%) Transaction Classification',
    query: 'percentage ratio threshold distinguishing discloseable transaction 5% from major transaction 25% requiring circular and shareholders approval'
  },
  {
    title: 'D2. De Minimis Connected Transaction Exemption (5% vs 0.1%)',
    query: 'de minimis exemption under Rule 14A.76 fully exempt if all percentage ratios are less than 0.1% or less than 1% or less than 5% with total consideration under HK$3 million'
  },
  {
    title: 'D3. Scheme Mandate Limit (Chapter 17 vs Chapter 13)',
    query: 'Chapter 17 share option and award schemes scheme mandate limit not exceeding 10% of total issued shares of the issuer'
  },
  {
    title: 'D4. Specialist Technology Companies Chapter 18C Market Cap Thresholds',
    query: 'Chapter 18C Specialist Technology Companies minimum market capitalization of HK$6 billion for Commercial Companies and HK$10 billion for Pre-Commercial Companies'
  }
];

// ============================================================================
// MAIN RUNNER
// ============================================================================
async function run(): Promise<void> {
  if (!fs.existsSync(LISTINGRULES_FILE)) {
    console.error(`\n❌ Error: Cannot find "${LISTINGRULES_FILE}" in current directory.`);
    console.log(`Please place "listingrules.txt" in this folder first.\n`);
    process.exit(1);
  }

  console.log('\n=============================================================');
  console.log('📈 HKEX MAIN BOARD LISTING RULES: ZERO-SHOT SEARCH BENCHMARK');
  console.log('=============================================================');

  console.log(`Reading ${LISTINGRULES_FILE}...`);
  const rawText: string = fs.readFileSync(LISTINGRULES_FILE, 'utf-8');
  console.log(
    `Total text size: ${(rawText.length / 1024).toFixed(1)} KB (${rawText.length.toLocaleString()} characters)`
  );

  console.log(`Chunking into ~${TARGET_CHUNK_SIZE} character windows...`);
  const chunks = chunkText(rawText, TARGET_CHUNK_SIZE);
  console.log(`Generated ${chunks.length.toLocaleString()} text chunks.\n`);

  console.log(`Indexing chunks into SearchEngine...`);
  const indexStart = performance.now();
  const engine = new SearchEngine();

  if (fs.existsSync(DENSE_BIN)) {
    console.log(`Loading dense embeddings from ${DENSE_BIN}...`);
    engine.loadDenseEmbeddings(DENSE_BIN);
    console.log(`Dense vectors loaded!`);
  } else {
    console.warn(`⚠️ Warning: "${DENSE_BIN}" not found. Run:`);
    console.warn(`  npx tsx embed_corpus.ts ./listingrules.txt ./listingrules_embeddings.bin`);
    console.warn(`to enable hybrid vector reranking.`);
  }

  for (let i = 0; i < chunks.length; i++) {
    engine.addDocument(`listingrules-chunk-${i + 1}`, chunks[i]!);
  }
  const indexTime = (performance.now() - indexStart).toFixed(1);
  console.log(`Indexed all ${chunks.length.toLocaleString()} chunks in ${indexTime} ms! Ready to search.\n`);

  console.log('=============================================================');
  console.log(`🔍 EXECUTING ${TEST_QUERIES.length} BENCHMARK QUERIES (TOP 3 CHUNKS)`);
  console.log('=============================================================');

  for (const item of TEST_QUERIES) {
    console.log(`\n-------------------------------------------------------------`);
    console.log(`📋 [${item.title}]`);
    console.log(`🔎 QUERY: "${item.query}"`);

    const qStart = performance.now();
    const results: SearchResult[] = await engine.search(item.query, 3);
    const qTime = (performance.now() - qStart).toFixed(2);

    console.log(`⏱️  Search latency: ${qTime} ms`);

    if (results.length === 0) {
      console.log('   No matching chunks found.');
      continue;
    }

    for (let rank = 0; rank < results.length; rank++) {
      const r = results[rank]!;
      const preview = r.text.replace(/\s+/g, ' ').slice(0, 400);

      console.log(
        `\n   #${rank + 1} [Score: ${r.score.toFixed(3)} | Prob: ${(r.probability * 100).toFixed(1)}%] ${r.id}`
      );
      console.log(`      "${preview}..."`);
    }
  }

  console.log('\n=============================================================');
  console.log('🏁 Search completed across HKEX Main Board Listing Rules.');
  console.log('=============================================================\n');
}

void run();
