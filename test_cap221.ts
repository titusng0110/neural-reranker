import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { SearchEngine } from './search.ts';

const CAP221_FILE = './cap221.txt';
const DENSE_BIN = './cap221_embeddings.bin';
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
// 10 REAL-WORLD CRIMINAL PROCEDURE QUERIES (CAP. 221)
// ============================================================================
// ============================================================================
// DIVERSE MULTI-PERSONA TEST SUITE ACROSS CAP. 221 (24 BENCHMARK QUERIES)
// ============================================================================
const TEST_QUERIES: readonly TestQuery[] = [
  // --------------------------------------------------------------------------
  // CATEGORY A: LAYMAN / HIGH-VIBINESS (Tests Vibiness Prior & Jina Embedding)
  // Zero section numbers, conversational grammar, everyday English words.
  // --------------------------------------------------------------------------
  {
    title: "A1. Citizen's Arrest (Section 101)",
    query: "can a regular ordinary person arrest someone who is committing a serious crime"
  },
  {
    title: "A2. Young Offenders Imprisonment Ban (Section 109A)",
    query: "can an 18 year old teenager be sent to prison or jail if convicted"
  },
  {
    title: "A3. Child Witness Video Evidence (Section 79B / 79C)",
    query: "can a child give evidence in court through a video recording instead of standing in the witness box"
  },
  {
    title: "A4. Skipping Bail / Absconding (Section 9L)",
    query: "what happens if a defendant fails to show up to court after getting released on bail"
  },
  {
    title: "A5. Suspended Prison Sentence (Section 109B)",
    query: "how does a suspended prison sentence work and do you go to jail if you commit another crime"
  },
  {
    title: "A6. Compensation to Crime Victims (Section 73)",
    query: "can the criminal judge order the convicted person to pay money back to the victim for injury and loss"
  },
  {
    title: "A7. New Evidence on Appeal (Section 83V)",
    query: "what can you do if you find fresh witness evidence after being convicted at trial"
  },
  {
    title: "A8. Unclaimed Police Property (Section 102)",
    query: "what happens to stolen items that the police seized if nobody claims them after six months"
  },

  // --------------------------------------------------------------------------
  // CATEGORY B: BARRISTER / LEGAL DOCTRINE (Tests BM25 & BPE IDF Salience)
  // Dense formal terminology, procedural terms of art, zero section numbers.
  // --------------------------------------------------------------------------
  {
    title: "B1. Review of Sentence as Manifestly Inadequate (Section 81A)",
    query: "application by secretary for justice for review of sentence on grounds of being manifestly inadequate or excessive"
  },
  {
    title: "B2. Reference on Point of Law Following Acquittal (Section 81D)",
    query: "reference of point of law to court of appeal following acquittal on indictment without affecting verdict"
  },
  {
    title: "B3. Formal Admissions of Fact (Section 65C)",
    query: "formal admission of facts in criminal proceedings dispensing with proof signed by counsel or solicitor"
  },
  {
    title: "B4. Unsafe or Unsatisfactory Verdict (Section 83)",
    query: "quashing conviction on indictment where verdict of jury is unsafe or unsatisfactory or material irregularity during trial"
  },
  {
    title: "B5. Double Jeopardy / Autrefois Pleas (Section 64)",
    query: "plea in bar of autrefois acquit and autrefois convict determination of previous conviction or acquittal"
  },
  {
    title: "B6. Severance / Separate Trials of Indictment (Section 23)",
    query: "prejudice to accused by misjoinder of counts and application for separate trial of distinct offences"
  },
  {
    title: "B7. Hospital Orders on Finding of Insanity (Section 76)",
    query: "special verdict of not guilty by reason of insanity and order admitting defendant to correctional mental hospital"
  },
  {
    title: "B8. Retrial Powers of Appellate Court (Section 83E)",
    query: "court of appeal ordering new trial following appeal against conviction where interests of justice require retrial"
  },

  // --------------------------------------------------------------------------
  // CATEGORY C: REFERENCE & EXACT CITATION (Tests Symbol Shingles & Citations)
  // Exact statutory codes, subsections, brackets, and numbers.
  // --------------------------------------------------------------------------
  {
    title: "C1. Bail Refusal Grounds under Section 9G",
    query: "section 9G grounds for refusal of bail risk of absconding or committing further offences"
  },
  {
    title: "C2. Notice of Alibi under Section 65D",
    query: "section 65D notice of alibi particulars of witness and place"
  },
  {
    title: "C3. Restrictions on Imprisoning Youths under Section 109A",
    query: "section 109A restriction on imprisonment of persons under 21 years of age"
  },
  {
    title: "C4. Disposal of Stolen / Seized Goods under Section 102",
    query: "section 102 disposal of property in possession of police connected with offences"
  },
  {
    title: "C5. Written Witness Statements under Section 65B",
    query: "section 65B written statement of witness admissible by consent"
  },
  {
    title: "C6. Committal by Transfer under Section 80B / Part III",
    query: "part III notice of transfer of indictable offence to court of first instance"
  },

  // --------------------------------------------------------------------------
  // CATEGORY D: AMBIGUOUS & HARD ADVERSARIAL QUERIES
  // Distinguishing subtle statutory differences across different sections.
  // --------------------------------------------------------------------------
  {
    title: "D1. Witness Arrest Warrant (Section 34 vs 101)",
    query: "warrant for arrest of witness failing to attend court after being served with witness summons"
  },
  {
    title: "D2. Costs Against Private Prosecutors (Section 73A / 74)",
    query: "award of defense costs against private prosecutor where indictment dismissed or defendant acquitted"
  }
];


// ============================================================================
// MAIN RUNNER
// ============================================================================
async function run(): Promise<void> {
  if (!fs.existsSync(CAP221_FILE)) {
    console.error(`Error: Cannot find "${CAP221_FILE}" in current directory.`);
    console.log(`Please place "cap221.txt" in this folder first.`);
    process.exit(1);
  }

  console.log('\n=============================================================');
  console.log('⚖️  CAP. 221 CRIMINAL PROCEDURE: ZERO-SHOT SEARCH BENCHMARK');
  console.log('=============================================================');

  console.log(`Reading ${CAP221_FILE}...`);
  const rawText: string = fs.readFileSync(CAP221_FILE, 'utf-8');
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
    console.warn(`⚠️ Warning: "${DENSE_BIN}" not found. Run "npx tsx embed_corpus.ts ./cap221.txt" first.`);
  }

  for (let i = 0; i < chunks.length; i++) {
    engine.addDocument(`cap221-chunk-${i + 1}`, chunks[i]!);
  }
  const indexTime = (performance.now() - indexStart).toFixed(1);
  console.log(`Indexed all ${chunks.length.toLocaleString()} chunks in ${indexTime} ms! Ready to search.\n`);

  console.log('=============================================================');
  console.log('🔍 EXECUTING 10 CRIMINAL PROCEDURE QUERIES (TOP 3 CHUNKS)');
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
      const preview = r.text.replace(/\s+/g, ' ').slice(0, 500);

      console.log(
        `\n   #${rank + 1} [Score: ${r.score.toFixed(3)} | Prob: ${(r.probability * 100).toFixed(1)}%] ${r.id}`
      );
      console.log(`      "${preview}..."`);
    }
  }

  console.log('\n=============================================================');
  console.log('🏁 Search completed across Cap. 221 Criminal Procedure Ordinance.');
  console.log('=============================================================\n');
}

void run();
