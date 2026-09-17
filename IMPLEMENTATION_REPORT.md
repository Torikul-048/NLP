# Implementation and verification report

Implemented on 15 September 2026; final restart and verification completed on 16 September 2026 using the supplied KUET CSE PDF.

## Delivered

- 108 term-specific course records (107 unique course codes), with source pages, optional groups, laboratory links, raw text, and source-conflict annotations.
- All four requested models, real pretrained assets, saved indexes, trained BiLSTM weights, cached course embeddings, and separate calibrated thresholds.
- Deterministic query parsing and answer formatting; no generative answering or external inference APIs.
- Streamlit UI with immediate model switching, processing details, source excerpts, and evaluation tables.
- Reproducible setup, corpus-building, training, calibration, evaluation, and test commands.

## Measured held-out results

48 total queries: 36 answerable and 12 unanswerable. These results use the supplied manually authored development benchmark.

### Retrieval ranking before metadata routing and rejection

| Model | Hit@1 | Hit@3 | MRR | Recall@3 |
|---|---:|---:|---:|---:|
| A: TF-IDF | 66.7% | 91.7% | 0.794 | 81.4% |
| B: pretrained Word2Vec | 61.1% | 75.0% | 0.707 | 65.9% |
| C: Word2Vec + BiLSTM | 5.6% | 19.4% | 0.183 | 16.7% |
| D: Sentence-BERT | 77.8% | 86.1% | 0.818 | 78.4% |

### Complete application, including metadata lookup and rejection

| Model | Hit@1 | Hit@3 | False acceptance | False rejection |
|---|---:|---:|---:|---:|
| A | 75.0% | 77.8% | 0.0% | 13.9% |
| B | 80.6% | 83.3% | 8.3% | 8.3% |
| C | 33.3% | 36.1% | 16.7% | 13.9% |
| D | 83.3% | 88.9% | 0.0% | 5.6% |

All four applications rejected the five explicitly out-of-domain test queries. This does not establish universal rejection accuracy. Per-category and per-query details are in `artifacts/evaluation/`.

**Model C is implemented and trained, but its retrieval quality is inadequate on this test set.** Training used 68 queries, 398 sampled pairs, and 24 validation queries. Early stopping retained epoch 12 (validation loss 0.5127), after stopping at epoch 17. The falling training loss and weak held-out ranking are consistent with overfitting and limited supervision. The model is retained as an honest experimental comparison, not presented as a successful improvement. A future study should expand and independently review relevance judgments, choose any changes using validation data, and evaluate once on a new untouched test set.

Sentence-BERT achieved the highest top-1 ranking and MRR here; TF-IDF achieved the highest raw Hit@3. These are small-sample observations, not general claims about the model families. SBERT also rejects some correct queries: the demo RSA/ElGamal query has a score just below its validation-selected threshold. Thresholds have not been lowered to accommodate test examples.

## Verification evidence

- `python -m pytest -q`: **28 passed**, including real four-model UI switching, disabling Model D, source extraction edge cases, metadata corrections, OOV handling, padding invariance, gradient learning, and leakage guards.
- `python -m pip check`: no broken requirements.
- All four evaluation runs completed; `unavailable_models` is empty in the evaluation manifest.
- Browser: local page renders, query submission returns sourced results, switching to Sentence-BERT preserves the query and displays its independently scored rejection, and no browser page errors were reported.
- Local server health endpoint returned `ok`.
- Disabled Streamlit's development file watcher after it attempted to import unrelated optional transformer vision modules; model loading and inference do not require torchvision.

## Source and reproducibility caveats

- B/C use the published **Google News SLIM** subset (299,567 pretrained 300-dimensional vectors), not all three million Google News entries. Full-model support remains available through setup. The asset SHA-256 and source URL are saved.
- Sentence-BERT's 256-token limit truncates 23 course descriptions; C also uses a 256-token maximum. These are documented experimental limits.
- Four courses have conflicting PDF metadata. The app preserves both detailed-syllabus and summary values and exposes the differences.
- The benchmark was authored during implementation. A separately collected, instructor-reviewed test set is needed for stronger academic conclusions.
- Direct dependency versions are recorded in `requirements-tested.txt`; setup ranges remain in `requirements.txt`.

## Open the project

Run `start_app.bat`, or from this folder run:

```powershell
& '..\Scripts\python.exe' -m streamlit run app.py
```

The local demo URL is http://127.0.0.1:8501. Restart the server after code changes because automatic source-file watching is disabled.
