# 📐 Feature Specification & Architectural Reference

This document provides a mathematical and architectural specification of all **18 normalized features** ($x_0$ to $x_{17}$) extracted by `search.ts` and evaluated by the 1028-byte SwiGLU neural reranker (`scoring.ts`).

Every feature is strictly bounded within the interval **$[0.0, 1.0]$**, ensuring numerical stability through the linear Skip Highway and non-linear SwiGLU gating activations.

---

## Summary Matrix

| Index | Feature Name | Category | Type | Range | Primary Purpose |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **`0`** | **BPE Lexical Ratio** | Sparse Lexical | Continuous | $[0, 1]$ | Rare-subword IDF mass coverage |
| **`1`** | **N-Gram Fuzzy Ratio** | Sub-character | Continuous | $[0, 1]$ | Typo & morphological stem tolerance |
| **`2`** | **Query Coverage** | Lexical Breadth | Continuous | $[0, 1]$ | Proportion of unique query terms matched |
| **`3`** | **Proximity Span** | Spatial Geometry | Continuous | $[0, 1]$ | Minimum covering window compactness |
| **`4`** | **Clause Integrity** | Syntactic Syntax | Continuous | $(0, 1]$ | Sentence terminal penalty in covering window |
| **`5`** | **Early Position Bias** | Structural Layout| Continuous | $(0, 1]$ | Position of first salient (above-median IDF) hit |
| **`6`** | **Order Monotonicity** | Sequential Flow | Boolean | $\{0, 1\}$ | Enforces left-to-right query phrase order |
| **`7`** | **Dense Cosine Similarity**| Deep Semantic | Continuous | $[0, 1]$ | Jina v5 embedding vector similarity |
| **`8`** | **Full BM25 Saturation** | Classic IR | Continuous | $[0, 1]$ | Length-normalized Okapi BM25 score |
| **`9`** | **Salience Gate (Max IDF)**| Critical Keyword | Boolean | $\{0, 1\}$ | Hard gate: does the rarest keyword hit? |
| **`10`** | **Information Density** | Noise Filter | Continuous | $[0, 1]$ | Proportion of document composed of hits |
| **`11`** | **Exact Phrase Match** | Precision Anchor| Continuous | $[0, 1]$ | Salient 2/3-gram collocation matches |
| **`12`** | **Query Vibiness Prior** | Demographic Router| Continuous | $[0, 1]$ | Detects conversational vs. legal queries |
| **`13`** | **Query Numeric Ratio** | Syntax Intent | Continuous | $[0, 1]$ | Detects statutory/code-seeking intent |
| **`14`** | **Document Prose Quality**| Cleanliness Filter| Continuous | $(0, 1]$ | Penalizes noisy tables/bracket dumps |
| **`15`** | **Symbol/Number Coverage**| Citation Match | Continuous | $[0, 1]$ | Fraction of section numbers/letters matched |
| **`16`** | **Symbol Salience Gate** | Primary Citation| Boolean | $\{0, 1\}$ | Hard gate: does the main section code hit? |
| **`17`** | **Exact Citation Hierarchy**| Shingle Match | Boolean | $\{0, 1\}$ | Compound match (e.g. `109A(1)(a)`) |

---

## Detailed Specifications

### 0. BPE Lexical Ratio
* **Formula:**
  $$x_0 = \frac{\sum_{t \in Q_{\text{BPE}} \cap D} \text{IDF}(t)}{\sum_{t \in Q_{\text{BPE}}} \text{IDF}(t)}$$
* **What it measures:** Fraction of the query's total **BPE subword information mass** present in the passage.
* **Why it matters:** Weighting by Robertson-Spärck Jones IDF guarantees that rare subwords drive the ratio toward $1.0$, while function words contribute negligibly.

### 1. N-Gram Fuzzy Ratio
* **Formula:**
  $$x_1 = \frac{\sum_{g \in Q_{\text{NGram}} \cap D} \text{IDF}(g)}{\sum_{g \in Q_{\text{NGram}}} \text{IDF}(g)}$$
* **What it measures:** Character-level sub-token matching using character trigrams (Latin) and bigrams (CJK).
* **Why it matters:** Provides resilience against spelling errors, OCR noise, and morphological variations (e.g., *"wound up"* vs. *"winding up"*).

### 2. Query Coverage
* **Formula:**
  $$x_2 = \frac{|\text{Matched Unique Query Tokens}|}{|Q_{\text{unique}}|}$$
* **What it measures:** Breadth of query satisfaction normalized by the number of unique query terms.
* **Why it matters:** Prevents a document that repeats a single rare keyword 50 times from dominating a multi-concept query.

### 3. Proximity Span
* **Formula:**
  $$x_3 = \begin{cases} \min\left(1.0, \frac{K}{\text{Span}_{\min}}\right) & \text{if } K \ge 2 \text{ and } \text{Span}_{\min} < \infty \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Compactness of the **Minimum Covering Window** containing all $K$ matched distinct query terms.
* **Why it matters:** Computed via a two-pointer sliding window over sorted positions. Stray word repetitions at the beginning or end of a document are discarded, isolating the densest semantic cluster.

### 4. Clause Integrity
* **Formula:**
  $$x_4 = \begin{cases} \frac{1.0}{1.0 + \text{SentenceTerminals}(L^*, R^*)} & \text{if } \text{Span}_{\min} < \infty \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Syntactic cohesion across the minimum covering window $[L^*, R^*]$. Counts sentence terminators (`.`, `!`, `?`, `\n`).
* **Why it matters:** Query terms appearing within the same sentence describe a single coherent legal rule. Terms scattered across multiple subsections describe unrelated exceptions.

### 5. Early Position Bias
* **Formula:**
  $$x_5 = \begin{cases} \frac{1.0}{1.0 + 0.05 \times \text{pos}_{\text{salient}}} & \text{if } \text{pos}_{\text{salient}} \ge 0 \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Hyperbolic decay based on the first occurrence of a **salient content word** ($\text{IDF} \ge \text{Median IDF}$, excluding stopwords).
* **Why it matters:** Statutory definitions and section headings appear at position 0. Stopwords like *"a"* or *"the"* are barred from triggering this reward.

### 6. Order Monotonicity
* **Formula:**
  $$x_6 = \begin{cases} 1.0 & \text{if query terms appear in left-to-right monotonic order} \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Sequential phrase directionality.
* **Why it matters:** Distinguishes *"victim killed defendant"* from *"defendant killed victim"*.

### 7. Dense Cosine Similarity
* **Formula:**
  $$x_7 = \max\left(0.0, \min\left(1.0, \frac{\mathbf{v}_Q \cdot \mathbf{v}_D}{\|\mathbf{v}_Q\|_2 \|\mathbf{v}_D\|_2}\right)\right)$$
* **What it measures:** 768-dimensional latent semantic vector similarity (Jina v5).
* **Why it matters:** Overcomes vocabulary mismatch, bridging conversational layman queries to formal legal jargon.

### 8. Full BM25 Saturation
* **Formula:**
  $$x_8 = \min\left(1.0, \frac{\text{BM25}(Q, D)}{\text{MaxBM25}(Q)}\right)$$
* **What it measures:** Length-normalized Okapi BM25 score divided by the theoretical maximum score for that query.
* **Why it matters:** Maps the unbounded range of BM25 into $[0, 1]$.

### 9. Salience Gate (Max IDF)
* **Formula:**
  $$x_9 = \begin{cases} 1.0 & \text{if } \arg\max_{t} \text{IDF}(t) \in D \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Hard boolean gate checking if the single rarest keyword exists in the document.
* **Why it matters:** Rejects documents matching secondary keywords while missing the primary concept.

### 10. Information Density
* **Formula:**
  $$x_{10} = \min\left(1.0, \frac{\sum_{t \in Q} \text{TF}(t, D)}{|D_{\text{words}}|}\right)$$
* **What it measures:** Concentration of query keywords relative to passage length.
* **Why it matters:** Penalizes long boilerplate clauses that contain only an incidental mention of the query terms.

### 11. Exact Phrase Match
* **Formula:**
  $$x_{11} = \begin{cases} 1.0 & \text{if whole query is an exact substring} \\ \min\left(1.0, \frac{\text{Matched Collocations}}{\min(3, |\text{Salient Phrases}|)}\right) & \text{otherwise} \end{cases}$$
* **What it measures:** Exact match of the full query or constituent 2-gram / 3-gram collocations that contain content words.
* **Why it matters:** Rewards contiguous phrase presence (e.g. *"reduction of share capital"*, *"autrefois acquit"*).

### 12. Query Vibiness Prior
* **Formula:**
  $$x_{12} = \min\left(1.0, \frac{\sum_{w \in Q} \mathbb{I}(w \in \text{WH\_WORDS}) \times 1.0 + \mathbb{I}(w \in \text{SIMPLE\_VOCAB}) \times 0.8}{|Q_{\text{words}}|}\right)$$
* **What it measures:** The conversational nature of the query.
* **Why it matters:** Routes neural attention. Activates SwiGLU Neuron #3 (amplifying Dense Cosine) when conversational phrasing is detected.

### 13. Query Numeric Ratio
* **Formula:**
  $$x_{13} = \frac{|\text{Query Symbols / Numbers}|}{|\text{Query Words}| + |\text{Query Symbols / Numbers}|}$$
* **What it measures:** Ratio of numeric/symbolic tokens to total tokens.
* **Why it matters:** Signals whether the query is seeking a concept ($0.0$) or a section code ($0.5+$).

### 14. Document Prose Quality Prior
* **Formula:**
  $$x_{14} = \frac{1.0}{1.0 + 6.0 \times \left(\frac{\text{Punctuation Count}}{\max(1, \text{Letter Count})}\right)}$$
* **What it measures:** Syntactic readability and prose fluency.
* **Why it matters:** Down-ranks signature blocks, tables of contents, and noisy tables of numbers.

### 15. Symbol/Number Coverage
* **Formula:**
  $$x_{15} = \begin{cases} \min\left(1.0, \frac{\sum_{s \in Q_{\text{symbols}}} \mathbb{I}(s \in D_{\text{symbols}})}{|Q_{\text{symbols}}|}\right) & \text{if } |Q_{\text{symbols}}| > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Fraction of section numbers, decimal sub-rules, and parentheticals matched.
* **Why it matters:** Prevents subsection mismatches (e.g. Section 102(1) vs Section 102(4)).

### 16. Symbol Salience Gate
* **Formula:**
  $$x_{16} = \begin{cases} 1.0 & \text{if } |Q_{\text{symbols}}| > 0 \text{ and } Q_{\text{symbols}}[0] \in D_{\text{symbols}} \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Whether the **first / primary section code** appears in the document.
* **Why it matters:** Ensures the primary governing statute number is present.

### 17. Exact Citation Hierarchy Match
* **Formula:**
  $$x_{17} = \begin{cases} 1.0 & \text{if } |Q_{\text{symbols}}| \ge 2 \text{ and } \bigoplus Q_{\text{symbols}} \in D_{\text{symbol\_shingles}} \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Verbatim match of the concatenated compound citation shingle (e.g., `109` + `a` $\implies$ `"109a"`).
* **Why it matters:** Disambiguates complex hierarchical citations like Rule 14A.35 from Rule 14A.36.
