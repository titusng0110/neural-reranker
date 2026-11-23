import fs from 'node:fs';
import { performance } from 'node:perf_hooks';

// Import both tokenizers
import {
  tokenize as tokenizeBPE,
  tokenizeWord as tokenizeWordBPE
} from './bpe_tokenizer.js';
import {
  tokenize as tokenizeNGram,
  tokenizeWord as tokenizeWordNGram
} from './ngram_tokenizer.js';

interface DemoSentence {
  category: string;
  text: string;
}

interface TokenizerBenchmarkStats {
  name: string;
  passes: number;
  wordsProcessed: number;
  elapsed: number;
  wordsPerSec: number;
  nsPerWord: string;
}

// ============================================================================
// 1. THE 20 COMPREHENSIVE MULTILINGUAL & INDUSTRY TEST CASES
// ============================================================================
const DEMO_SENTENCES: readonly DemoSentence[] = [
  // --- 10 ENGLISH INDUSTRIES (~30 words each) ---
  {
    category: '1. Law & Jurisprudence',
    text: 'The defendant indemnified the plaintiff against statutory civil liability, filing an interlocutory injunction asserting that the contentious contractual obligations were rendered unenforceable due to material breach and unconscionable misrepresentation.'
  },
  {
    category: '2. Clinical Medicine & Pharmacology',
    text: 'Pharmacokinetic evaluations demonstrated that intravenous administration of the therapeutic monoclonal antibody induced significant immunosuppression without precipitating acute neurodegenerative toxicity or gastrointestinal contraindications in randomized clinical trials.'
  },
  {
    category: '3. Finance & Quantitative Trading',
    text: 'Algorithmic market makers capitalized on decentralized liquidity pools, hedging collateralized debt obligations against macroeconomic inflationary volatility using automated high-frequency arbitrage algorithms across international commodity derivative exchanges.'
  },
  {
    category: '4. Distributed Systems & Cloud Infrastructure',
    text: 'Asynchronous microservices orchestrated via Kubernetes maintained fault-tolerant distributed consensus through Raft replication, scaling horizontally to mitigate telemetric ingestion latency across multi-region serverless container clusters.'
  },
  {
    category: '5. Aerospace & Orbital Mechanics',
    text: 'The telemetric guidance computer calculated orbital re-entry trajectories, modulating supersonic aerodynamic control surfaces and cryogenic thruster vectoring to counteract atmospheric thermal dissipation during ballistic deceleration.'
  },
  {
    category: '6. Civil & Structural Engineering',
    text: 'Prestressed geotechnical reinforcement anchors stabilized the seismic foundation, dissipating hydraulic cantilever shear stress through elastomeric dampers to prevent catastrophic structural resonance during high-magnitude tectonic disturbances.'
  },
  {
    category: '7. E-Commerce & Transnational Logistics',
    text: 'Omnichannel fulfillment centers optimized predictive inventory replenishment, utilizing automated freight consignment tracking and localized supply chain logistics to accelerate doorstep fulfillment while reducing cross-border customs overhead.'
  },
  {
    category: '8. Real-Time Graphics & Game Engines',
    text: 'The rasterization pipeline utilized compute shaders for anisotropic ray-traced reflections, blending skeletal animation transforms with volumetric atmospheric scattering to render photorealistic environments at ultra-low frame latency.'
  },
  {
    category: '9. Molecular Genetics & Biotechnology',
    text: 'CRISPR-mediated transcriptional regulation suppressed oncogenic ribonucleic mutations, altering histone methylation patterns to inhibit polymorphic genomic replication across selectively targeted malignant cellular phenotypes in vitro.'
  },
  {
    category: '10. Philosophy & Epistemology',
    text: 'Phenomenological hermeneutics interrogates the metaphysical presuppositions of existential ontology, demonstrating that subjective epistemological determinism cannot be reconciled with radical deontological ethics without encountering recursive dialectical contradictions.'
  },

  // --- 10 GLOBAL LANGUAGE FAMILIES (~30 words each) ---
  {
    category: '11. Spanish (Romance)',
    text: 'El desarrollo de la inteligencia artificial transforma profundamente las economías modernas, generando debates éticos sobre la automatización laboral, la protección de datos personales y la soberanía tecnológica internacional.'
  },
  {
    category: '12. German (Germanic - tests massive compound nouns!)',
    text: 'Die Bundesverfassungsgerichtsentscheidung über die Geschwindigkeitsbegrenzungsinfrastruktur auf Bundesautobahnen erforderte umfangreiche wissenschaftliche Gutachten über Umweltverträglichkeitsprüfungen und wirtschaftliche Gesamtentwicklungsprognosen für den Personenverkehr.'
  },
  {
    category: '13. French (Romance - accents & elisions)',
    text: 'Les chercheurs de renommée internationale étudient attentivement les conséquences socio-économiques du réchauffement climatique mondial, soulignant la nécessité impérative d’une transition énergétique équitable pour préserver les écosystèmes fragiles.'
  },
  {
    category: '14. Russian (Slavic - Cyrillic morphology)',
    text: 'Современные телекоммуникационные системы обеспечивают мгновенную передачу конфиденциальной информации через защищённые каналы связи, стимулируя стремительное развитие глобальной цифровой экономики и международного научно-технического сотрудничества.'
  },
  {
    category: '15. Chinese / Mandarin (Sino-Tibetan - no spaces)',
    text: '人工智能深度学习算法正在迅速革新医疗诊断技术，科研人员利用海量临床数据训练高精度神经网络模型，显著提高了早期罕见疾病的筛查准确率与治愈率。'
  },
  {
    category: '16. Japanese (Japonic - Kanji + Hiragana + Katakana)',
    text: '次世代の量子コンピューティング技術は暗号通信や分子シミュレーションの分野に劇的な進展をもたらし、世界中の研究機関が持続可能な情報社会の実現に向けて開発を加速しています。'
  },
  {
    category: '17. Arabic (Semitic - right-to-left diacritics)',
    text: 'تعتمد التنمية المستدامة في منطقة الشرق الأوسط على الابتكار التكنولوجي والاستثمار في مصادر الطاقة المتجددة، مما يعزز الاستقرار الاقتصادي والتعاون الإقليمي لمكافحة التصحر والتغير المناخي.'
  },
  {
    category: '18. Hindi (Indo-Aryan - Devanagari script & matras)',
    text: 'आधुनिक सूचना प्रौद्योगिकी और डिजिटल क्रांति ने समाज के प्रत्येक वर्ग को सशक्त बनाया है, जिससे सरकारी सेवाओं की पारदर्शिता बढ़ी है और ग्रामीण अर्थव्यवस्था में आर्थिक विकास हुआ है।'
  },
  {
    category: '19. Korean (Koreanic - Hangul syllabic blocks)',
    text: '최첨단 반도체 제조 공정과 인공지능 프로세서 설계 기술은 글로벌 정보통신 산업의 핵심 경쟁력으로 부상하며 새로운 디지털 패러다임 변화를 주도하고 있습니다.'
  },
  {
    category: '20. Turkish (Turkic - Agglutinative suffixes)',
    text: 'Uluslararası bilimsel araştırmalar çerçevesinde geliştirilen yenilikçi nanoteknolojik malzemeler, enerji depolama sistemlerinin verimliliğini artırarak çevre dostu elektrikli araçların yaygınlaşmasına büyük katkılar sağlamaktadır.'
  }
];

// ============================================================================
// 2. SIDE-BY-SIDE LINGUISTIC COMPARISON
// ============================================================================
function runLinguisticComparison(): void {
  console.log('\n================================================================');
  console.log('🔬 SIDE-BY-SIDE LINGUISTIC DEMO: BPE vs. N-GRAM');
  console.log('================================================================');

  for (const item of DEMO_SENTENCES) {
    console.log(`\n----------------------------------------------------------------`);
    console.log(`[${item.category}]`);
    console.log(`Input: "${item.text}"`);

    const bpeTokens = tokenizeBPE(item.text);
    const ngramTokens = tokenizeNGram(item.text);

    console.log(`\n  BPE (${bpeTokens.length} tokens):`);
    console.log(`  `, bpeTokens);

    console.log(`\n  N-GRAM (${ngramTokens.length} tokens):`);
    console.log(`  `, ngramTokens);
  }
}

// ============================================================================
// 3. MACBOOK DICTIONARY SPEED BENCHMARK (/usr/share/dict/words)
// ============================================================================
function runMacDictionaryBenchmark(): void {
  const DICT_PATH = '/usr/share/dict/words';
  if (!fs.existsSync(DICT_PATH)) {
    console.log(`\nNote: Dictionary ${DICT_PATH} not found. Skipping dictionary speed test.`);
    return;
  }

  console.log('\n================================================================');
  console.log('⚡ MACBOOK DICTIONARY SPEED BENCHMARK (/usr/share/dict/words)');
  console.log('================================================================');

  // Read raw words into memory
  const rawData = fs.readFileSync(DICT_PATH, 'utf-8');
  const words = rawData.split('\n').filter((w) => w.length > 0);
  const totalWords = words.length;

  console.log(`Loaded ${totalWords.toLocaleString()} dictionary words into memory.\n`);

  // --- Helper to benchmark a tokenizer ---
  function measureTokenizer(
    name: string,
    tokenizeFn: (word: string) => string[]
  ): TokenizerBenchmarkStats {
    console.log(`[${name}] Warming up JIT...`);
    for (let i = 0; i < 20000; i++) {
      tokenizeFn(words[i]!);
    }

    console.log(`[${name}] Running benchmark loop for ~1.5s...`);
    let wordsProcessed = 0;
    let passes = 0;
    const start = performance.now();
    let elapsed = 0;

    // Loop for at least 1,500ms
    while (elapsed < 1500) {
      for (let i = 0; i < totalWords; i++) {
        tokenizeFn(words[i]!);
      }
      wordsProcessed += totalWords;
      passes++;
      elapsed = performance.now() - start;
    }

    const elapsedSeconds = elapsed / 1000;
    const wordsPerSec = Math.round(wordsProcessed / elapsedSeconds);
    const nsPerWord = ((elapsed / wordsProcessed) * 1e6).toFixed(1);

    return {
      name,
      passes,
      wordsProcessed,
      elapsed,
      wordsPerSec,
      nsPerWord
    };
  }

  // Run both
  const bpeStats = measureTokenizer('BPE Tokenizer', tokenizeWordBPE);
  const ngramStats = measureTokenizer('N-Gram Tokenizer', tokenizeWordNGram);

  // --- Print Summary Table ---
  console.log('\n================================================================');
  console.log('📊 HEAD-TO-HEAD BENCHMARK RESULTS');
  console.log('================================================================');

  const printStats = (stats: TokenizerBenchmarkStats): void => {
    console.log(`\n${stats.name}:`);
    console.log(`  Completed Passes:   ${stats.passes} full passes`);
    console.log(`  Total Words Tested: ${stats.wordsProcessed.toLocaleString()}`);
    console.log(`  Elapsed Time:       ${stats.elapsed.toFixed(1)} ms`);
    console.log(`  Throughput:         ⚡ ${stats.wordsPerSec.toLocaleString()} words/second`);
    console.log(`  Latency Per Word:   ${stats.nsPerWord} ns/word`);
  };

  printStats(bpeStats);
  printStats(ngramStats);

  const speedDiff = (ngramStats.wordsPerSec / bpeStats.wordsPerSec).toFixed(2);
  const fasterName =
    ngramStats.wordsPerSec > bpeStats.wordsPerSec ? ngramStats.name : bpeStats.name;

  console.log(`\n🏆 Performance Verdict:`);
  console.log(`  ${fasterName} is ${speedDiff}x faster on pure dictionary throughput.`);
  console.log('================================================================\n');
}

// ============================================================================
// MAIN RUNNER
// ============================================================================
runLinguisticComparison();
runMacDictionaryBenchmark();
