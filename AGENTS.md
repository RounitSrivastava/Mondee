# Persona-Based Recommendation Engine — System Instructions

## Project Identity

This project implements a **hybrid retrieval recommendation engine** that uses Rich (no — I mean FAISS + dense embeddings + LDA topic modeling) to recommend e-commerce products to users modeled through behavioral **personas**. It is not a generic build-a-recommender tutorial; it is a designed-for-scale hybrid retrieval system already wired to Amazon Electronics metadata.

---

## Architecture Overview

```
data/meta_Electronics.json          (source catalog)
            │
            ▼
   scripts/build_index.py           (data ingestion + cleaning)
            │
            ▼
   core/vectorizer.py              (RoBERTa/bge + LDA hybrid vectors)
            │
            ▼
   core/indexer.py                 (FAISS IVF + Inner Product index)
            │
        index/                    (binary index + metadata + pickled LDA/cv)
            │
            ▼
   core/recommender.py             (query embedding → FAISS search → ranked list)
            │
            ▼
   personas/test_personas.py       (truck_driver, budget_buyer, premium_buyer, …)
            │
            ▼
   scripts/run_evaluation.py       (orchestrates full evaluation)
            │
            ▼
   evaluation/evaluator.py         (metrics: score_distribution, overlap_matrix, summary)
            │
            ▼
   evaluation/results/             (JSON + CSV outputs)
            │
            ▼
   scripts/report.py               (terminal report viewer)
```

---

## Directory Map

| Path | Purpose |
|------|---------|
| `core/indexer.py` | Builds and loads the FAISS index; creates hybrid vectors; saves metadata |
| `core/recommender.py` | Persona → query string embedding → FAISS search → ranked recommendations |
| `core/vectorizer.py` | Loads `intfloat/e5-large-v2` SentenceTransformer; trains LDA; produces 768-dim RoBERTa + 30-dim LDA concatenated vectors |
| `scripts/build_index.py` | Reads `data/*.json`, cleans text, deduplicates by ASIN, calls `core/indexer.build_index()` |
| `scripts/run_evaluation.py` | Runs all `TEST_PERSONAS`, collects metrics, writes results |
| `scripts/report.py` | Reads latest `evaluation/results/` and prints colored terminal report |
| `evaluation/evaluator.py` | Scoring logic: `score_distribution`, `overlap_matrix`, `build_summary`, `save_results` |
| `personas/test_personas.py` | 9 distinct personas testing GPS / Electronics domain (truck driver, budget buyer, …) |
| `data/meta_Electronics.json` | Amazon Electronics metadata — 498K+ JSON-lines, each record is a Python dict literal |
| `index/` | Generated at runtime: `experiences.index`, `experiences_meta.json`, `lda_model.pkl`, `count_vectorizer.pkl` |
| `evaluation/results/` | Timestamped JSON + CSV outputs from evaluation runs |
| `test_read.py` | Utility to validate record count / integrity of `meta_Electronics.json` |
| `requirements.txt` | Pinned GPU-free dependencies |

---

## Quick Start (Canonical Order)

```bash
# 1 — create virtual environment and install
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2 — verify data file integrity
python test_read.py

# 3 — build the FAISS index (trains LDA, embeds all records, writes index/)
python scripts/build_index.py

# 4 — run recommendations + full evaluation
python scripts/run_evaluation.py

# 5 — view the terminal report
python scripts/report.py
```

---

## Data

- **Format**: One JSON-lines record per line in `data/meta_Electronics.json`. Each line is a Python `dict` literal parsed with `ast.literal_eval` (not `json.loads` — Amazon metadata may contain NaN-like values or non-standard formatting that `ast` tolerates).
- **Key fields used**: `asin` (unique product ID), `title` / `name` / `summary` (product name), `description`, `categories`, `feature`, `brand`.
- **Max records**: `MAX_RECORDS = 50000` in `build_index.py` — caps ingestion to keep memory bounded.
- **Deduplication**: By `asin`. First-seen record wins.
- **Cleaning pipeline** (`clean_text` in `build_index.py`):
  1. Strip HTML tags (`<...> → " "`)
  2. Lowercase
  3. Remove non-alphanumeric chars (keep spaces)
  4. Collapse whitespace

---

## Vectorization — The Hybrid Embedding

### RoBERTa / E5-large-v2 (semantic)

- **Model**: `intfloat/e5-large-v2` loaded via `sentence-transformers`
- **Passage prefix**: Documents/experiences are prefixed with `"passage: "`
- **Query prefix**: Queries are prefixed with `"query: "`
- **Normalization**: `normalize_embeddings=True` during encoding + manual L2 normalization in indexer
- **Dimension**: 768

### LDA Topic Distribution (semantic topic)

- **CountVectorizer**: `stop_words="english"`, `max_features=3000`, `min_df=2`, `max_df=0.95`
- **LDA**: `n_components=30` topics, `random_state=42`, `learning_method="batch"`
- Saved to `index/lda_model.pkl` and `index/count_vectorizer.pkl`

### Hybrid Vector Composition (in `core/indexer.py`)

```
hybrid_vector = normalize( [0.7 × RoBERTa_embedding] ⊕ [0.3 × LDA_distribution] )
```

- `EMBEDDING_WEIGHT = 0.7`, `LDA_WEIGHT = 0.3`
- Both sub-vectors L2-normalized individually before weighting. Final concatenated vector normalized again.
- Total dimension: `768 + 30 = 798`

---

## FAISS Index

- **Metric**: Inner Product (after unit-normalization, IP = cosine similarity)
- **Index type**:
  - **IVFFlat** when `len(vectors) > 5000`: `nlist = min(256, max(32, sqrt(n)))`, `nprobe = min(40, nlist)`
  - **IndexFlatIP** for ≤ 5000 vectors (exhaustive search, no training overhead)
- Saved to `index/experiences.index`

---

## Index Metadata

Saved alongside the FAISS index as `index/experiences_meta.json`. Each entry:

```json
{
  "exp_id": "exp_<asin>",
  "name": "<cleaned title>",
  "details": "<first 2000 chars of cleaned description+summary+categories+features>"
}
```

---

## Recommendation Pipeline (`core/recommender.py`)

### Persona → Query Flattening

A persona dict is converted to a single string via `_flatten_persona()`:

1. **`persona`** field: snake_case → spaces, lowercased (e.g., `"truck_driver"` → `"truck driver"`)
2. **`signals`** list: each signal lowercased and appended
3. **`query_string`**: appended as-is (lowercased)
4. **`destination`**: appended as-is (lowercased)
5. **Expansion injection**: certain persona names auto-inject related terms:
   - `"budget shopper"` → `cheap`, `discount`, `affordable`, `deal`
   - `"premium buyer"` → `luxury`, `premium`, `high quality`
   - `"sports enthusiast"` → `fitness`, `gym`, `athletic`, `performance`
   - `"health conscious"` → `healthy`, `organic`, `nutrition`, `wellness`
   - `"gift buyer"` → `gift`, `birthday`, `present`
   - `"family buyer"` → `family`, `kids`, `children`, `household`

**Note**: The 9 test personas use names like `truck_driver`, `professional_driver`, etc. These names do NOT match the expansion dictionary keys above, so no auto-injection occurs for them — only the explicit `query_string` drives retrieval. This is intentional: the test personas provide rich `query_string` fields already.

### Search and Ranking

1. Embed the flattened query via `embed_query()` (uses `"query: {text}"` prefix)
2. L2-normalize the query vector
3. Search FAISS with `search_k = min(top_k * 10, index.ntotal)`
4. Deduplicate by `exp_id` (first occurrence wins)
5. Return up to `top_k` results (default 30, evaluator uses 50)
6. Each result: `rank`, `exp_id`, `name` (truncated 60 chars), `details` (truncated 200 chars), `score` (IP score × 100, 2 decimal places)

### Response Schema

```json
{
  "user_id": "u_001",
  "persona": "truck_driver",
  "query_string_used": "truck driver truck gps commercial vehicle …",
  "generated_at": "2026-06-10T08:35:20+00:00",
  "refresh_due_at": "2026-06-10T20:35:20+00:00",
  "recommendations": [
    {
      "rank": 1,
      "exp_id": "exp_B00C7FKT2A",
      "name": "Garmin nüvi 2797LMT 7-Inch…",
      "details": "…",
      "score": 92.35
    },
    …
  ]
}
```

The `refresh_due_at` field is computed as `generated_at + 12h` to suggest when personalized results should be re-fetched.

---

## Evaluation Methodology

Tests are designed to answer one question: **do different personas surface different products?**

### Persona Separation (overlap matrix)

Compares Jaccard overlap between every pair of persona result sets (using top-10 recommendations by default):

```python
separation = 1 - jaccard_similarity(persona_a_results, persona_b_results)
```

- **≥ 0.70**: healthy separation
- **≥ 0.50**: moderate separation
- **< 0.50**: overlapping results (problematic)

### Score Distribution per Persona

| Metric | Meaning |
|--------|---------|
| `top_1_score` | Raw IP score (×100) of the #1 result |
| `top_10_avg` | Average score of top 10 results |
| `score_spread` | `top_1 − last_result` — should be > 3 for distinct ranking |
| `std_dev` | Population std dev of all scores |
| `confidence` | `top_1 − top_2_to_6_avg` — higher means clearer #1 |
| `diversity` | `unique_products / total_results` — 1.0 means no duplicates |

### Global Summary Averages

- `avg_score_spread` < 3 → low score spread → weak ranking
- `avg_separation` < 0.50 → low persona separation
- `avg_confidence` < 1.5 → weak recommendation confidence

### Thresholds for "Healthy" System

```
avg_separation ≥ 0.70  AND  avg_confidence ≥ 1.5
```

---

## Configuration Constants

| File | Constant | Default | Purpose |
|------|----------|---------|---------|
| `core/indexer.py` | `EMBEDDING_WEIGHT` | `0.7` | RoBERTa vector weight |
| `core/indexer.py` | `LDA_WEIGHT` | `0.3` | LDA vector weight |
| `core/indexer.py` | `BATCH_SIZE` | `128` | E5 encoding batch size |
| `core/indexer.py` | `USE_IVF` | `True` | Switch between IVF/Flat |
| `core/vectorizer.py` | `N_TOPICS` | `30` | LDA topic count |
| `personas/test_personas.py` | `TOP_K` (evaluator) | `30` | Default top_k in recommender |
| `scripts/run_evaluation.py` | `top_k` | `50` | Evaluation uses more |
| `scripts/build_index.py` | `MAX_RECORDS` | `50000` | Hard cap on catalog ingestion |
| `scripts/report.py` | score thresholds | 80/60 | Color bands for score display |
| `scripts/report.py` | separation thresholds | 0.70/0.50 | Color bands for separation |

---

## Adding a New Persona

1. Edit `personas/test_personas.py`
2. Append a new dict to `TEST_PERSONAS` with keys:
   - `user_id` (string): unique identifier
   - `persona` (string): snake_case persona name (used for expansion lookup)
   - `signals` (list of strings): behavioral signals
   - `signal_confidences` (dict): confidence per signal (informational only)
   - `query_string` (string): rich natural-language query driving retrieval
   - `destination` (string): optional destination context
3. Re-run `python scripts/run_evaluation.py`
4. View results with `python scripts/report.py`

---

## Expanding the Persona Auto-Injection Dictionary

Edit `_flatten_persona()` in `core/recommender.py` (lines 125–166). Add new persona names mapped to synonym lists:

```python
expansions = {
    "existing persona": ["syn1", "syn2"],
    "new_persona": ["related_term_a", "related_term_b"]
}
```

Persona names are matched exactly after `_` → ` ` replacement and `.lower()`.

---

## Rebuilding the Index

- **When**: After changing the data source, vectorization constants, or LDA parameters.
- **How**: `python scripts/build_index.py`
- **Outputs**: `index/experiences.index`, `index/experiences_meta.json`, `index/lda_model.pkl`, `index/count_vectorizer.pkl`
- **Note**: The IVF index requires > 5000 records to trigger; with fewer records a FlatIP fallback is used.

---

## Dependency Notes

| Library | Role |
|---------|------|
| `sentence-transformers==5.5.1` | E5-large-v2 encoder (PyTorch backend) |
| `faiss-cpu==1.13.2` | Vector index (no GPU required) |
| `scikit-learn==1.8.0` | CountVectorizer + LDA |
| `pandas==3.0.2` | Not used in core pipeline yet |
| `numpy==2.4.4` | All array math |
| `colorama==0.4.6` | Colored terminal output in `report.py` |
| `tqdm==4.67.3` | Progress bars during batch embedding |
| `requests==2.33.1` | Not used in core pipeline yet |

---

## Design Constraints and Anti-Patterns to Avoid

- **Do NOT use `json.loads` on `meta_Electronics.json`** — use `ast.literal_eval` per record (build_index.py already handles this correctly).
- **Do NOT negate or remove the single/double quote style** — the project uses single quotes everywhere; do not convert to JSON double-quote style.
- **Do NOT modify the hybrid vector formula without updating the indexer AND both embedding functions** — the weight split (0.7 / 0.3) and normalization are order-sensitive.
- **Do NOT skip LDA training** unless you also manually drop the LDA dimension from `hybrid_vectors` in `core/indexer.py` — otherwise dimension mismatch at query time.
- **Do NOT add `--no-gpg-sign`, `--no-verify`, or force-push** unless explicitly instructed by the user.
- **Do NOT run `build_index.py` without first verifying data integrity** — a malformed line can silently corrupt the entire catalog ingestion (use `test_read.py` first).

---

## Performance Characteristics

- **Index build time** (50K records, 768 + 30 dims, E5-large-v2, CPU): ~15–30 min depending on hardware. Batched at 128 per encoding batch.
- **Query latency** (single persona): < 1 second including embedding + FAISS search + metadata lookup.
- **Memory footprint** (50K vectors × 798 dims × 4 bytes): ~150 MB for the FAISS index plus ~50–80 MB for numpy arrays during build.
- **Disk** (index artifacts): ~200 MB total for a 50K record index.

---

## What This Project Is NOT

- Not a collaborative filtering engine (no user-item interaction matrix).
- Not a multi-modal recommender (images from `imUrl` are not ingested).
- Not a real-time learning system (the index is static until rebuilt).
- Not a production API — there is no HTTP server, API layer, or caching middleware.

---

## Future Extension Points (Known Debt)

- `pandas` and `requests` are in `requirements.txt` but not imported in any core module — they were likely used during data exploration and lint will flag them if purist mode is enabled.
- `test_read.py` is a throwaway utility hardcoded to a single filename; make it CLI-arguable if you want to keep it.
- The `destination` field in personas is never populated or consumed in `_flatten_persona()` for meaningful routing — it is just appended to the query string.
- LDA artifacts (`lda_model.pkl`, `count_vectorizer.pkl`) are not listed in `build_index.py` output logging, so a developer looking at logs may not realize they are being written.
- No unit tests exist. `personas/test_personas.py` is a data fixture, not a test suite.
- README.MD is empty (0 lines). The project has no user-facing documentation.
