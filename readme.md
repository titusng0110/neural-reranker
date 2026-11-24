# 📐 Feature Specification & Architectural Reference

This document provides the definitive mathematical and architectural specification of all **17 normalized features** ($x_0$ to $x_{16}$) extracted by `search.ts` and evaluated by the **976-byte SwiGLU neural reranker** (`scoring.ts`).

Every feature is strictly bounded within the interval **$[0.0, 1.0]$**, ensuring numerical stability through the linear Skip Highway and non-linear SwiGLU gating activations ($\text{SiLU}(\text{Gate}) \odot \text{Value}$).

---

## Pre-Tokenization Pipeline & Whitespace Normalization

Prior to tokenization, both query strings and document passages pass through a deterministic text preprocessor:

$$\text{text} \longrightarrow \text{Regex}\Big(\texttt{/[^\textbackslash S\textbackslash r\textbackslash n]\{3,\}/g} \to \texttt{"  "}\Big) \longrightarrow \text{Regex}\Big(\texttt{/(?:\textbackslash r?\textbackslash n)\{3,\}/g} \to \texttt{"\textbackslash n\textbackslash n"}\Big)$$

1. **Horizontal Space Collapse**: 3 or more contiguous non-newline spaces collapse into exactly 2 spaces.
2. **Vertical Space Collapse**: 3 or more contiguous newline sequences collapse into exactly 2 newlines (`\n\n`).
3. **Casing & Normalization**: Strings are case-folded (`toLowerCase()`) and Unicode-normalized (`NFC`).

---

## The Tri-Tokenizer Architecture

The engine runs three specialized, concurrent tokenization pipelines over the normalized text:

```
                                  ┌──► Tokenizer 1: Universal BPE (Full Text)
                                  │    Pruned OpenAI o200k_base (68,174 tokens)
Normalized Text ─────────────────┼──► Tokenizer 2: Symbol & Alphanumeric BPE
                                  │    Prunes pure language; preserves codes (e.g., "109A", "apple1", "(1)")
                                  └──► Tokenizer 3: Multilingual Language N-Gram
                                       Prunes symbols/codes; character N-grams (CJK: 2-gram, Others: 3-gram)
```

1. **Tokenizer 1 (Universal BPE)**: Breaks the entire passage and query into subwords via zero-allocation linked-list BPE.
2. **Tokenizer 2 (Symbol & Alphanumerics BPE)**: Prunes a token *if and only if* it consists entirely of pure natural language letters (`/^[\p{L}\p{M}]+$/u`). Preserves any token with digits, section numbers, parentheticals, punctuation, or mixed alphanumerics (e.g., `apple1`, `109A`, `(1)`, `14A.35`, `$500`).
3. **Tokenizer 3 (Multilingual Language N-Gram)**: Prunes pure numbers and symbols, isolating natural language words across all world scripts. Generates script-adaptive character n-grams (**2-gram** for CJK, **3-gram** for Latin, Cyrillic, Arabic, etc.).

---

## Summary Matrix

| Index | Feature Name | Tokenizer Domain | Type | Range | Primary Purpose |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **`0`** | **Full BPE BM25** | Tokenizer 1 (Universal) | Continuous | $[0, 1]$ | Length-normalized Okapi BM25 across all subwords |
| **`1`** | **Symbol & Alphanumeric BM25** | Tokenizer 2 (Symbols/Codes) | Continuous | $[0, 1]$ | Isolated BM25 for statutory codes and mixed tokens |
| **`2`** | **Language N-Gram BM25** | Tokenizer 3 (Language) | Continuous | $[0, 1]$ | Morphological, typo, and CJK sub-word BM25 |
| **`3`** | **Tokenizer 1 Query Coverage** | Tokenizer 1 (Universal) | Continuous | $[0, 1]$ | Breadth ratio of matched unique BPE query terms |
| **`4`** | **Tokenizer 2 Query Coverage** | Tokenizer 2 (Symbols/Codes) | Continuous | $[0, 1]$ | Breadth ratio of matched statutory/symbol terms |
| **`5`** | **Tokenizer 3 Query Coverage** | Tokenizer 3 (Language) | Continuous | $[0, 1]$ | Breadth ratio of matched language character n-grams |
| **`6`** | **Tokenizer 1 Salience Gate** | Tokenizer 1 (Universal) | Boolean | $\{0, 1\}$ | Hard gate: does the rarest universal keyword match? |
| **`7`** | **Tokenizer 2 Salience Gate** | Tokenizer 2 (Symbols/Codes) | Boolean | $\{0, 1\}$ | Hard gate: does the rarest section/symbol code match? |
| **`8`** | **Tokenizer 1 Exact Phrase Match** | Tokenizer 1 (Universal) | Continuous | $[0, 1]$ | Verbatim query substring or salient phrase collocation |
| **`9`** | **Tokenizer 2 Exact Citation Match** | Tokenizer 2 (Symbols/Codes) | Boolean | $\{0, 1\}$ | Verbatim compound citation match (e.g., `109a(1)(a)`) |
| **`10`** | **Clause Integrity** | Tokenizer 1 (Positional) | Continuous | $[0, 1]$ | Syntactic sentence boundary cohesion within hit span |
| **`11`** | **Early Position Bias** | Tokenizer 1 (Positional) | Continuous | $[0, 1]$ | Hyperbolic reward for salient hit near chunk start |
| **`12`** | **Lexical Information Density** | Tokenizer 1 (Frequency) | Continuous | $[0, 1]$ | Ratio of query hits to total passage word length |
| **`13`** | **Dense Cosine Similarity** | Latent Semantic Vector | Continuous | $[0, 1]$ | Jina v5 768-dim $L_2$-normalized vector cosine similarity |
| **`14`** | **Query Vibiness Prior** | Query-side Prior | Continuous | $[0, 1]$ | Detects conversational / beginner intent vs. formal |
| **`15`** | **Query Numeric Ratio** | Query-side Prior | Continuous | $[0, 1]$ | Ratio of numbers/symbols to total query tokens |
| **`16`** | **Document Prose Quality Prior** | Document-side Prior | Continuous | $(0, 1]$ | Penalizes noisy tables, bracket dumps, and index junk |

---

## Detailed Specifications

### 0. Full BPE BM25 (Tokenizer 1)
* **Formula:**
  $$x_0 = \min\left(1.0, \frac{\text{BM25}_{T1}(Q, D)}{\text{MaxQueryBM25}_{T1}(Q)}\right)$$
  Where:
  $$\text{BM25}_{T1}(Q, D) = \sum_{t \in Q_1 \cap D_1} \text{IDF}_1(t) \cdot \frac{\text{TF}_1(t, D) \cdot (k_1 + 1)}{\text{TF}_1(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D_1|}{\text{avgdl}_1}\right)}$$
  $$\text{MaxQueryBM25}_{T1}(Q) = \sum_{t \in Q_1} \text{IDF}_1(t) \cdot (k_1 + 1)$$
  *(Hyperparameters: $k_1 = 1.2$, $b = 0.75$, Robertson–Spärck Jones IDF).*
* **What it measures:** General Okapi BM25 saturation over universal BPE subwords.
* **Why it matters:** Incorporates non-linear term frequency saturation and document length normalization across the entire vocabulary.

---

### 1. Symbol & Alphanumeric BM25 (Tokenizer 2)
* **Formula:**
  $$x_1 = \begin{cases} \min\left(1.0, \frac{\text{BM25}_{T2}(Q, D)}{\text{MaxQueryBM25}_{T2}(Q)}\right) & \text{if } |Q_2| > 0 \text{ and } \text{MaxQueryBM25}_{T2} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Okapi BM25 score computed strictly over tokens that contain numbers, statutory codes, punctuation, or mixed alphanumerics (e.g., `apple1`, `109A`, `§12`, `(1)`, `14A.35`). Pure dictionary language words are barred from this calculation.
* **Why it matters:** Prevents generic text from drowning out precision section matches. If a user asks for *"Rule 14A.35"*, this feature fires at near $1.0$ only on documents containing the actual statutory code.

---

### 2. Language N-Gram BM25 (Tokenizer 3)
* **Formula:**
  $$x_2 = \begin{cases} \min\left(1.0, \frac{\text{BM25}_{T3}(Q, D)}{\text{MaxQueryBM25}_{T3}(Q)}\right) & \text{if } |Q_3| > 0 \text{ and } \text{MaxQueryBM25}_{T3} > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Okapi BM25 score over language character n-grams (bigrams for CJK, trigrams for Latin and other scripts), ignoring numbers and standalone symbols.
* **Why it matters:** Acts as an automatic safety net for morphological stems (*"indemnified"* vs *"indemnification"*), spelling typos, and unsegmented Asian languages (Chinese, Japanese) where whitespace tokenization fails.

---

### 3. Tokenizer 1 Query Coverage
* **Formula:**
  $$x_3 = \begin{cases} \frac{|Q_1 \cap D_1|}{|Q_1|} & \text{if } |Q_1| > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Unweighted proportion of distinct universal BPE query subwords present in the document.
* **Why it matters:** Ensures multi-concept queries (e.g. *"injunction material breach misrepresentation"*) match across all required concepts rather than over-saturating on one repeated word.

---

### 4. Tokenizer 2 Query Coverage
* **Formula:**
  $$x_4 = \begin{cases} \frac{|Q_2 \cap D_2|}{|Q_2|} & \text{if } |Q_2| > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Unweighted proportion of distinct symbol, statutory, and alphanumeric query terms present in the document.
* **Why it matters:** Pinpoints multi-clause citations (e.g., verifying both `14A` and `35` are present for a Rule 14A.35 query).

---

### 5. Tokenizer 3 Query Coverage
* **Formula:**
  $$x_5 = \begin{cases} \frac{|Q_3 \cap D_3|}{|Q_3|} & \text{if } |Q_3| > 0 \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** Unweighted proportion of distinct language character n-grams matched.
* **Why it matters:** Quantifies overall morphological coverage across all linguistic terms in the query.

---

### 6. Tokenizer 1 Salience Gate
* **Formula:**
  $$x_6 = \begin{cases} 1.0 & \text{if } \arg\max_{t \in Q_1} \text{IDF}_1(t) \in D_1 \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** A hard boolean gate checking if the single most informative/rarest BPE term in the query exists in the document.
* **Why it matters:** Immediately discards passages that match secondary common terms (e.g., *"court"*, *"order"*) but completely omit the core concept (e.g., *"autrefois"*).

---

### 7. Tokenizer 2 Salience Gate
* **Formula:**
  $$x_7 = \begin{cases} 1.0 & \text{if } |Q_2| > 0 \text{ and } \arg\max_{t \in Q_2} \text{IDF}_2(t) \in D_2 \\ 0.0 & \text{otherwise} \end{cases}$$
* **What it measures:** A hard boolean gate checking if the rarest symbol/alphanumeric code in the query exists in the document.
* **Why it matters:** In statutory reference queries (e.g., *"section 109A restriction on imprisonment"*), this ensures the document contains the anchor section number `109A`.

---

### 8. Tokenizer 1 Exact Phrase Match
* **Formula:**
  $$x_8 = \begin{cases} 1.0 & \text{if cleaned query string is a verbatim substring of } D \\ \min\left(1.0, \frac{\text{Matched Salient Phrases}}{\min(3, |\text{Salient Phrases}|)}\right) & \text{if salient collocations exist} \\ 0.0 & \text{otherwise} \end{cases}$$
  *(Salient phrases are 2-gram and 3-gram word windows containing at least one non-stopword).*
* **What it measures:** Verbatim contiguity of the entire query or multi-word content phrases.
* **Why it matters:** Distinguishes contiguous terms of art (e.g., *"winding up"*, *"reverse takeover"*) from documents where the words happen to appear in separate paragraphs.

---

### 9. Tokenizer 2 Exact Citation Match
* **Formula:**
  $$x_9 = \begin{cases} 1.0 & \text{if } |Q_{\text{symbols}}| \ge 2 \text{ and } \bigoplus Q_{\text{symbols}} \in D_{\text{symbol\_shingles}} \\ 0.0 & \text{otherwise} \end{cases}$$
  *(Where $\bigoplus Q_{\text{symbols}}$ is the concatenated string of symbols, e.g., `109a` + `(1)` + `(a)` $\implies$ `"109a(1)(a)"`).*
* **What it measures:** Verbatim match of hierarchical statutory codes and parentheticals.
* **Why it matters:** Disambiguates specific sub-rules (e.g., Rule 8.08(1)(a) vs. Rule 8.08(1)(b)).

---

### 10. Clause Integrity
* **Formula:**
  $$x_{10} = \begin{cases} \frac{1.0}{1.0 + \text{SentenceTerminals}(L^*, R^*)} & \text{if } K \ge 2 \text{ and } [L^*, R^*] \text{ is valid} \\ 0.0 & \text{otherwise} \end{cases}$$
  Where $[L^*, R^*]$ is the minimal covering window of Tokenizer 1 word positions containing all $K$ matched distinct terms, and $\text{SentenceTerminals}$ counts sentence terminators (`.`, `!`, `?`, `\n`) between $L^*$ and $R^*$.
* **What it measures:** Syntactic cohesion of the query hits. Evaluates whether query terms co-occur within an unbroken sentence boundary rather than being fragmented across different subsections.
* **Why it matters:** A passage where all query terms occur in a single sentence describes one cohesive rule; a passage where hits cross 5 periods describes unrelated provisions.

---

### 11. Early Position Bias (Tokenizer 1)
* **Formula:**
  $$x_{11} = \begin{cases} \frac{1.0}{1.0 + 0.05 \times \text{pos}_{\text{salient}}} & \text{if } \text{pos}_{\text{salient}} \ge 0 \\ 0.0 & \text{otherwise} \end{cases}$$
  Where $\text{pos}_{\text{salient}}$ is the zero-based word index of the earliest Tokenizer 1 content hit satisfying:
  $$t \notin \text{FUNCTION\_WORDS} \quad \land \quad \text{IDF}_1(t) \ge \text{Median}(\text{IDF}_1(Q))$$
* **What it measures:** Hyperbolic decay based on how close to the start of the passage the core topic appears.
* **Why it matters:** Major statutory rules, section titles, and definition headwords appear in the opening line of a section chunk.

---

### 12. Lexical Information Density
* **Formula:**
  $$x_{12} = \min\left(1.0, \frac{\sum_{t \in Q_1 \cap D_1} \text{TF}_1(t, D)}{|D_{\text{words}}|}\right)$$
* **What it measures:** Ratio of total query term occurrences to total document length.
* **Why it matters:** Rejects bloated, verbose boilerplate sections where query hits are merely incidental passing mentions.

---

### 13. Dense Cosine Similarity
* **Formula:**
  $$x_{13} = \max\left(0.0, \min\left(1.0, \frac{\mathbf{q}_{\text{emb}} \cdot \mathbf{d}_{\text{emb}}}{\|\mathbf{q}_{\text{emb}}\|_2 \|\mathbf{d}_{\text{emb}}\|_2}\right)\right)$$
* **What it measures:** 768-dimensional latent semantic vector similarity generated by the Jina v5 embedding model.
* **Why it matters:** Overcomes vocabulary mismatch. Bridges conversational layman slang (e.g., *"boss secretly buying an asset"*) to formal statutory drafting (e.g., *"connected transaction with director"*).

---

### 14. Query Vibiness Prior
* **Formula:**
  $$x_{14} = \min\left(1.0, \frac{\sum_{w \in Q_{\text{words}}} \Big(\mathbb{I}(w \in \text{WH\_WORDS}) \times 1.0 + \mathbb{I}(w \in \text{SIMPLE\_VOCAB}) \times 0.3\Big)}{|Q_{\text{words}}|}\right)$$
  *(WH_WORDS = question words & conversational pronouns; SIMPLE_VOCAB = 1,250 triple-filtered elementary English words).*
* **What it measures:** The conversational nature of the query ($1.0$ = casual beginner query, $0.0$ = formal legal terminology).
* **Why it matters:** Dynamically routes SwiGLU hidden neurons to elevate Dense Cosine Similarity ($x_{13}$) and suppress BM25 over-reliance on natural-language queries.

---

### 15. Query Numeric Ratio
* **Formula:**
  $$x_{15} = \frac{|Q_{\text{symbols}}|}{|Q_{\text{words}}| + |Q_{\text{symbols}}|}$$
* **What it measures:** Proportion of the query composed of numbers, decimal codes, and parentheticals.
* **Why it matters:** Acts as an automatic architectural switch: when $x_{15} > 0.3$, the model suppresses general language ambiguity and routes attention directly into the Symbol BM25 ($x_1$) and Citation Match ($x_9$) lanes.

---

### 16. Document Prose Quality Prior
* **Formula:**
  $$x_{16} = \frac{1.0}{1.0 + 6.0 \times \left(\frac{\text{Punctuation Count}}{\max(1, \text{Letter Count})}\right)}$$
* **What it measures:** The syntactic fluency and prose quality of the document text.
* **Why it matters:** Suppresses noisy tables of numbers, signature blocks, repetitive formatting, and table-of-contents fragments without discarding valid short sections.

---

## Binary Weight Packing Specification (`weights.bin`: 976 Bytes)

The neural reranker weights are serialized in zero-copy little-endian IEEE 754 format:

$$\begin{aligned}
\text{Total Parameters} &= W_{\text{skip}}(17) + W_{\text{gate}}(17 \times 6) + b_{\text{gate}}(6) + W_{\text{val}}(17 \times 6) + b_{\text{val}}(6) + W_{\text{out}}(6) + b_{\text{out}}(1) + T(1) + \text{Bias}(1) \\
&= 17 + 102 + 6 + 102 + 6 + 6 + 1 + 1 + 1 = \mathbf{242 \text{ floats}} \\
\text{Total File Size} &= 8 \text{ bytes header } (\text{uint32 } \text{NUM\_FEATURES} + \text{uint32 } \text{HIDDEN\_NEURONS}) + (242 \times 4 \text{ bytes}) = \mathbf{976 \text{ bytes}}
\end{aligned}$$