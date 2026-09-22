# NLP-Based Academic Information Retrieval System

A non-generative retrieval application for the supplied KUET CSE curriculum. Academic answers use deterministic templates and the PDF's structured records. There are no generative model APIs, RAG pipelines, LangChain, or LlamaIndex dependencies.

## Quick start on this computer

Open PowerShell in this folder:

```powershell
& '..\Scripts\python.exe' -m streamlit run app.py
```

Or double-click `start_app.bat`. The application opens at http://localhost:8501. Once the assets and indexes have been prepared, all retrieval runs locally.

## Reproduce from scratch

Python 3.11–3.13 is recommended. The existing workspace uses Python 3.13.

```powershell
python -m venv .venv
& '.\.venv\Scripts\Activate.ps1'
python -m pip install -r requirements.txt
python -m scripts.build_corpus --pdf 'data/raw/CourseContentsCSE_14_08_2022.pdf'
python -m scripts.prepare_assets --word2vec slim
python -m scripts.build_all_indexes --models a b
python -m scripts.train_model_c
python -m scripts.build_all_indexes --models d
python -m scripts.calibrate_thresholds
python -m scripts.evaluate_all
python -m pytest -q
python -m streamlit run app.py
```

Prepare assets before building indexes: the optional WordNet download changes lexical normalization, so rebuilding indexes is necessary when switching from offline fallback rules to WordNet.

For full Google News vectors use `python -m scripts.prepare_assets --word2vec full`. This downloads about 1.6 GB and requires several GB of RAM. The practical default is the published [Google News SLIM subset](https://github.com/eyaler/word2vec-slim), with 299,567 pretrained 300-dimensional vectors. It retains a broad external vocabulary, including words absent from the curriculum. This is explicitly recorded in `artifacts/vector_source.json`; it must be identified as a subset in any report.

For an existing trusted file:

```powershell
python -m scripts.prepare_assets --word2vec local --path 'D:\models\GoogleNews-vectors-negative300.bin.gz'
```

Changing vectors requires rebuilding B, retraining C, and recalibrating. Pickled Gensim/joblib files must come from a trusted source. Downloads are explicit setup operations; selecting a model in the UI never downloads missing weights.

### Run without Sentence-BERT

```powershell
python -m scripts.prepare_assets --word2vec slim --skip-sbert
python -m scripts.build_all_indexes --models a b
python -m scripts.train_model_c
python -m scripts.calibrate_thresholds --models a b c
python -m scripts.evaluate_all --models a b c
$env:DISABLE_MODEL_D = '1'
python -m streamlit run app.py
```

Model A also works independently with only PyMuPDF, NumPy, scikit-learn, NLTK, joblib, pandas, and Streamlit installed.

## Architecture

```text
PDF → course records + metadata + provenance
Query → code normalization → aliases / conservative typos → rule-based intent
      ├─ exact code/title or year/term → structured lookup
      └─ topic → selected retrieval model → model-specific rejection threshold
Results → verbatim curriculum excerpts + deterministic templates → Streamlit
```

| Model | Representation | Ranking |
|---|---|---|
| A | TF-IDF unigrams/bigrams, sublinear TF, noun lemmatization | Cosine |
| B | TF-IDF-weighted pretrained Word2Vec average | Cosine |
| C | Dynamic Word2Vec sequences, shared bidirectional LSTM, paired classifier | Sigmoid relevance |
| D | `sentence-transformers/all-MiniLM-L6-v2` | Normalized embedding cosine |

All four models expose `retrieve(query: str, top_k: int = 3)`. Each returns `model`, `intent`, `status`, `answer`, `results`, and `processing_details`. They index the same ordered corpus. Raw score magnitudes are not comparable across model families. Cosine scores and Model C probabilities are not guarantees of correctness.

### Corpus fidelity

The 19-page PDF contains 108 term-specific course records and 107 unique course codes. A missing colon in CSE 4245 is handled. CSE 4000 occurs in both fourth-year terms, with different credits: both records remain, with unique `record_id` fields. Year/term queries return the complete matching list, including optional offerings explicitly labeled by group; the system does not claim a student takes all optional courses.

Each record includes raw text, physical source page(s), credits, contact hours, term, year, type, optional group, parent laboratory course, description, and search text. Titles appear twice in search text as requested. Generic laboratory records remain available for exact/laboratory queries but are excluded from general topic results. Descriptions continue across page boundaries and stop before summary tables.

The PDF itself has contradictory values for EEE 2118, CSE 4101, CSE 4102, and HUM 4207. Detailed-syllabus values are stored as the primary values; summary-table values and conflicts are retained and shown. For example, HUM 4207 has 3 credits in the syllabus and 2 in the summary. The app flags this discrepancy. It never silently repairs the source. See `data/processed/corpus_statistics.json`.

### Model C

Pretrained vectors are looked up dynamically, so unseen curriculum words can still have embeddings. OOV positions are explicitly identified and removed before sequence packing; relative order of covered words is preserved. All-OOV queries are rejected. There are no random retrieval vectors.

The encoder uses a 300-dimensional input, 128 hidden units per direction, one shared BiLSTM, and final forward/backward hidden states. The classifier consumes `[q, d, abs(q-d), q*d]`, followed by Linear → ReLU → Dropout(0.3) → Linear. Training uses weighted `BCEWithLogitsLoss`, Adam, gradient clipping, seed 42, and early stopping on validation loss. Course representations are cached after training. Documents are limited to 256 tokens; this limitation is recorded in configuration.

Negative sampling uses three high-ranked incorrect TF-IDF courses and two other random courses. All declared positives are excluded from negatives. Unanswerable training queries receive only negative pairs. The training pair CSV and query IDs are saved for inspection.

### Evaluation and leakage controls

`relevance_queries.csv` contains 68 training and 24 validation queries. `evaluation_queries.csv` contains 48 separately authored held-out questions across eight categories. `group_id` is an additional column for paraphrase-group checks. Training reads only the relevance file; calibration uses only validation rows. Exact cross-split duplicates, shared declared groups, and nonexistent gold course codes fail validation.

This is a small development benchmark authored while implementing the system, not an independently collected test set. Semantic group independence and judgments still merit instructor review; code cannot prove that two differently worded questions are not paraphrases. Do not tune model architecture, thresholds, or query rules to improve the published test score. Create a new independently authored test set before making broader research claims.

Some topics occur in several real courses. For example, public-key encryption is present in E-Commerce as well as Computer Security, and cryptographic fault attacks appear in Fault Tolerant System. Multiple gold codes are allowed, separated with `|`; ranking Computer Security first is not universally the only correct judgment.

Thresholds maximize validation balanced acceptance/rejection accuracy, with ties favoring rejection. Calibration saves every validation top score, corpus hash, and model-manifest hash. Rebuilt artifacts invalidate old thresholds.

Outputs in `artifacts/evaluation/`:

- `comparison.csv`: Hit@1, Hit@3, MRR, Recall@3, OOD accuracy, false acceptance/rejection rates.
- `by_category.csv`: the same metrics by query category.
- `per_query.csv`: predictions and outcomes for inspection.
- `run_manifest.json`: dataset audit, evaluation fingerprint, and unavailable models.

Two modes are reported. **retrieval_only** measures the selected model's ranking before rejection or metadata routing, with full-ranking MRR. **end_to_end** measures final answers after deterministic routing and rejection, with MRR over returned results. This prevents exact-code lookups from being presented as learned semantic ability. Positive-only ranking metrics exclude unanswerable queries; rejection metrics use the appropriate positive/negative denominators. Recall@3 measures the fraction of all gold courses returned in the first three positions. A year/term query with 21 gold courses therefore has a maximum Recall@3 of 3/21 even when the full UI listing is correct. Missing metrics remain empty; unavailable models never receive invented scores.

### Automated Visualizations and Performance Plots

Running `start_app.bat` or `python -m scripts.generate_plots` automatically generates 300 DPI evaluation figures and organizes them into model-wise folders inside `plotting/`:

- `plotting/model_a/`, `model_b/`, `model_c/`, `model_d/`:
  - `confusion_matrix_retrieval_only.png` & `confusion_matrix_end_to_end.png`: 2×2 confusion matrices (True Positive, False Positive, True Negative, False Negative) evaluating query acceptance vs out-of-domain rejection.
  - `classification_metrics.png`: Acceptance classification Accuracy, Precision, Recall, F1-Score, and Specificity.
  - `retrieval_metrics.png`: Hit@1, Hit@3, MRR, and Recall@3 ranking comparisons.
  - `category_performance.png`: Breakdown across query categories (exact, semantic, metadata, typo, partial, ambiguous).
  - `score_distribution.png`: Top similarity/relevance scores for answerable vs unanswerable queries with the calibrated decision threshold marked.
  - `training_loss_curve.png`: Model C BiLSTM training & validation loss convergence (when trained).
  - `metrics_summary.json`: Detailed machine-readable JSON metrics summary.
- `plotting/comparison/`:
  - `model_comparison_retrieval_only.png` & `model_comparison_end_to_end.png`: 4-model side-by-side performance comparisons.
  - `error_rates_far_frr.png`: False Acceptance Rate (FAR) vs False Rejection Rate (FRR) trade-offs.
  - `overall_radar_chart.png`: Multi-dimensional radar profile summarizing each model's strengths.
- All generated plots are also viewable and refreshable directly inside the Streamlit web application under **"📊 Evaluation Results & Performance Plots"**.

### Limitations

- A small supervised dataset can overfit. Model C is an experiment; it is not guaranteed to beat TF-IDF or pretrained SBERT.
- Broad out-of-domain detection combines explicit rules and calibrated score rejection. It cannot guarantee rejection of every conceivable unsupported topic.
- Exact titles/codes use deterministic metadata routing, clearly labeled in the interface. Scores for those lookups are `null`, not fabricated confidence values.
- SBERT has a finite token limit; its manifest records truncated documents. Both sequence models may miss topics near the end of long descriptions.
- A general query may match several legitimate courses. Results and source excerpts are shown instead of asserting one uniquely correct course.
- Missing prerequisites, instructor names, schedules, fees, and unsupported academic facts are never invented.

## Useful commands

```powershell
python -m scripts.build_all_indexes --models a
python -m scripts.calibrate_thresholds --models a
python -m scripts.evaluate_all --models a
python -m scripts.generate_plots
python -c "from src.registry import create_model; print(create_model('model_a').retrieve('Which course covers RSA and ElGamal?'))"
```

## Technical references

- [Gensim pretrained vector loading](https://radimrehurek.com/gensim/downloader.html)
- [Sentence Transformers embedding interface](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.BCEWithLogitsLoss.html)
