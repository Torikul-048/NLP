"""Local Streamlit demo. Answers are formatted from curriculum records only."""
import os

# Assets must be prepared explicitly. UI interactions must not start downloads.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ACADEMIC_IR_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import streamlit as st

from src.registry import MODEL_LABELS, create_model
from src.utils import ROOT, read_json

st.set_page_config(page_title="KUET Academic IR", page_icon="📚", layout="wide")


def file_version(path):
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_data
def course_data(version):
    return read_json(ROOT / "data/processed/courses.json")


@st.cache_data
def array_data(model_id, filename, version):
    return np.load(ROOT / "artifacts" / model_id / filename)


@st.cache_data
def comparison_data(version):
    return pd.read_csv(ROOT / "artifacts/evaluation/comparison.csv")


@st.cache_resource
def cached_model(model_id, corpus_version, manifest_version, threshold_version, vector_version):
    courses = course_data(corpus_version)
    model = create_model(model_id, courses=courses)
    filenames = {"model_b": "document_vectors.npy", "model_c": "document_embeddings.npy", "model_d": "document_embeddings.npy"}
    if model_id in filenames:
        filename = filenames[model_id]
        array = array_data(model_id, filename, file_version(ROOT / "artifacts" / model_id / filename))
        if model_id == "model_c":
            import torch
            model.document_embeddings = torch.from_numpy(array)
        else:
            model.matrix = array
    return model


st.caption("KUET · COMPUTER SCIENCE AND ENGINEERING · NLP LAB PROJECT")
st.title("NLP-Based Academic Information Retrieval System")
st.write("Explore your curriculum. Compare four retrieval models. Check every answer against its source.")

corpus_path = ROOT / "data/processed/courses.json"
if not corpus_path.exists():
    st.error("The curriculum has not been prepared yet.")
    st.code("python -m scripts.build_corpus")
    st.stop()

courses = course_data(file_version(corpus_path))
st.caption(f"{len(courses)} course records · 19 source pages · Deterministic answers from the supplied curriculum")
st.divider()
left, right = st.columns([2, 1], gap="large")
with left:
    query = st.text_input("Natural Language Query", value="Which course covers RSA and ElGamal?",
                          placeholder="Ask about a topic, course, credits, lab, year or term", key="query")
with right:
    selected = st.selectbox("Model", list(MODEL_LABELS.values()), key="model_selection")
model_id = next(key for key, value in MODEL_LABELS.items() if value == selected)
clicked = st.button("Search", type="primary", key="search_button")
if clicked:
    st.session_state["searched"] = True

if st.session_state.get("searched"):
    directory = ROOT / "artifacts" / model_id
    disabled = model_id == "model_d" and os.environ.get("DISABLE_MODEL_D") == "1"
    if disabled or not (directory / "manifest.json").exists():
        st.warning("This model is disabled." if disabled else "This model has not been prepared yet.")
        st.code("python -m scripts.train_model_c" if model_id == "model_c" else f"python -m scripts.build_all_indexes --models {model_id[-1]}")
    else:
        try:
            with st.spinner(f"Searching with {selected.split(' — ')[0]}…"):
                model = cached_model(model_id, file_version(corpus_path), file_version(directory / "manifest.json"),
                                     file_version(ROOT / "artifacts/thresholds.json"), file_version(ROOT / "artifacts/vector_source.json"))
                response = model.retrieve(query)
                st.session_state["last_response"] = response
            a, b, c = st.columns([1, 2, 1])
            a.caption("DETECTED QUERY TYPE")
            a.write(response["intent"].replace("_", " ").title())
            b.caption("SELECTED MODEL")
            b.write(selected)
            c.caption("STATUS")
            c.write(response["status"].replace("_", " ").title())
            st.subheader("Final Answer")
            if response["status"] == "ok":
                st.success(response["answer"].replace("\n", "\n\n"))
            else:
                st.info(response["answer"])
            details = response["processing_details"]
            if details.get("route") == "structured_metadata":
                st.caption("Confidence/Relevance Score: not applicable · Exact curriculum metadata lookup")
            elif "top_score" in details:
                label = "Relevance probability" if model_id == "model_c" else "Cosine similarity"
                st.caption(f"Confidence/Relevance Score: {details['top_score']:.4f} · {label} · Rejection threshold: {model.threshold:.4f}")
            if not model.calibrated:
                st.warning("This model uses an uncalibrated provisional threshold. Run scripts.calibrate_thresholds before reporting results.")
            results = response["results"]
            if results:
                title = "Matching Curriculum Courses" if len(results) > 3 else "Top 3 Retrieved Courses"
                st.subheader(title)
                st.dataframe(pd.DataFrame([{
                    "Rank": i, "Course Code": r["course_code"], "Course Title": r["course_title"],
                    "Score": r["score"], "Credits": r["credits"], "Year": r["year"], "Term": r["term"],
                    "Optional Group": r.get("optional_group") or "Compulsory",
                } for i, r in enumerate(results, 1)]), hide_index=True, width="stretch",
                    column_config={"Score": st.column_config.NumberColumn(format="%.4f")})
                for r in results:
                    with st.expander(f"{r['course_code']} · {r['course_title']} · Source page {r['source_page']}"):
                        st.write(r["matched_excerpt"] or "No topic description is supplied.")
                        st.caption(f"Contact hours: {r['contact_hours']} · Type: {r['course_type']} · Pages: {', '.join(map(str, r['source_pages']))}")
                        if r.get("optional_group"):
                            st.caption(f"Optional offering: {r['optional_group']}. Inclusion does not mean every student takes this course.")
                        if r.get("parent_course"):
                            st.caption(f"Laboratory based on {r['parent_course']}")
                        for conflict in r.get("source_conflicts", []):
                            st.warning(f"PDF source conflict ({conflict['field']}): syllabus says {conflict['syllabus']}; summary on page {conflict['summary_page']} says {conflict['summary']}.")
            with st.expander("Show Processing Details"):
                st.json(details)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            st.error(f"The selected model is unavailable: {exc}")
            st.caption("Prepare the assets and rebuild this model using the README commands, then search again.")
else:
    st.info("Enter an academic query and select Search. After searching, switch the model to compare the same query immediately.")

st.divider()
with st.expander("📊 Evaluation Results & Performance Plots", expanded=False):
    tab_table, tab_plots = st.tabs(["📋 Metrics Table", "📈 Visual Plots & Confusion Matrices"])
    with tab_table:
        path = ROOT / "artifacts/evaluation/comparison.csv"
        if path.exists():
            df = comparison_data(file_version(path))
            st.caption("Held-out development benchmark. Ranking metrics exclude unanswerable questions. The two modes separate model ranking from the complete application.")
            st.dataframe(df, hide_index=True, width="stretch")
        else:
            st.caption("Run python -m scripts.evaluate_all after training and threshold calibration.")

    with tab_plots:
        plotting_dir = ROOT / "plotting"
        # Auto-generate if missing
        if not plotting_dir.exists() or not any(plotting_dir.glob("*/*.png")):
            try:
                from src.plotting import generate_all_plots
                generate_all_plots()
            except Exception:
                pass

        header_col, btn_col = st.columns([3, 1])
        with header_col:
            selected_plot_view = st.selectbox(
                "Select Model or Overview",
                [
                    "Cross-Model Comparison",
                    "Model A — TF-IDF (Sparse)",
                    "Model B — Word2Vec (Dense Centroid)",
                    "Model C — BiLSTM (Neural Matcher)",
                    "Model D — Sentence-BERT (MiniLM)",
                ],
                key="ui_plot_view_selection",
            )
        with btn_col:
            st.write("")
            st.write("")
            if st.button("🔄 Regenerate Plots", key="ui_regen_plots_btn"):
                try:
                    from src.plotting import generate_all_plots
                    res = generate_all_plots()
                    st.success(f"Generated {res['generated_files_count']} figures into {res['output_directory']}!")
                except Exception as exc:
                    st.error(f"Plot regeneration error: {exc}")

        model_key_map = {
            "Model A — TF-IDF (Sparse)": "model_a",
            "Model B — Word2Vec (Dense Centroid)": "model_b",
            "Model C — BiLSTM (Neural Matcher)": "model_c",
            "Model D — Sentence-BERT (MiniLM)": "model_d",
        }

        if selected_plot_view == "Cross-Model Comparison":
            comp_dir = plotting_dir / "comparison"
            st.subheader("Cross-Model Comparative Benchmark")
            c1, c2 = st.columns(2)
            p_ro = comp_dir / "model_comparison_retrieval_only.png"
            p_e2e = comp_dir / "model_comparison_end_to_end.png"
            if p_ro.exists():
                c1.image(str(p_ro), caption="Retrieval Only Mode (Raw Vector Ranking)", use_container_width=True)
            if p_e2e.exists():
                c2.image(str(p_e2e), caption="End-to-End Mode (Final Application Answers)", use_container_width=True)

            c3, c4 = st.columns(2)
            p_err = comp_dir / "error_rates_far_frr.png"
            p_rad = comp_dir / "overall_radar_chart.png"
            if p_err.exists():
                c3.image(str(p_err), caption="False Acceptance Rate (FAR) vs False Rejection Rate (FRR)", use_container_width=True)
            if p_rad.exists():
                c4.image(str(p_rad), caption="Multi-Metric Capability Radar Profile", use_container_width=True)

        elif selected_plot_view in model_key_map:
            m_id = model_key_map[selected_plot_view]
            m_dir = plotting_dir / m_id
            st.subheader(f"{selected_plot_view} Performance Visualizations")

            # Row 1: Confusion Matrices
            st.markdown("#### 1. Confusion Matrices (Query Acceptance & Rejection)")
            c1, c2 = st.columns(2)
            cm_ro = m_dir / "confusion_matrix_retrieval_only.png"
            cm_e2e = m_dir / "confusion_matrix_end_to_end.png"
            if cm_ro.exists():
                c1.image(str(cm_ro), caption="Confusion Matrix — Retrieval Only Mode", use_container_width=True)
            if cm_e2e.exists():
                c2.image(str(cm_e2e), caption="Confusion Matrix — End-to-End Mode", use_container_width=True)

            # Row 2: Metrics
            st.markdown("#### 2. Classification & Retrieval Ranking Metrics")
            c3, c4 = st.columns(2)
            m_clf = m_dir / "classification_metrics.png"
            m_ret = m_dir / "retrieval_metrics.png"
            if m_clf.exists():
                c3.image(str(m_clf), caption="Acceptance Classification Metrics (Acc, Prec, Rec, F1)", use_container_width=True)
            if m_ret.exists():
                c4.image(str(m_ret), caption="Retrieval Ranking Performance (Hit@1, Hit@3, MRR, Recall@3)", use_container_width=True)

            # Row 3: Category & Score Distribution
            st.markdown("#### 3. Category Performance & Score Distributions")
            c5, c6 = st.columns(2)
            m_cat = m_dir / "category_performance.png"
            m_dist = m_dir / "score_distribution.png"
            if m_cat.exists():
                c5.image(str(m_cat), caption="Category-Specific Performance (End-to-End)", use_container_width=True)
            if m_dist.exists():
                c6.image(str(m_dist), caption="Top Score Distribution & Calibrated Decision Boundary", use_container_width=True)

            # Model C training curve
            loss_curve = m_dir / "training_loss_curve.png"
            if loss_curve.exists():
                st.markdown("#### 4. Neural Network Training Dynamics")
                st.image(str(loss_curve), caption="BiLSTM Training & Validation Loss Convergence", use_container_width=True)

with st.expander("Curriculum source and model notes"):
    st.write("KUET CSE undergraduate syllabus, effective from academic session 2021–2022. Answers quote or format the supplied PDF; missing information is reported explicitly.")
    vector_source = ROOT / "artifacts/vector_source.json"
    if vector_source.exists():
        source = read_json(vector_source)
        st.caption(f"Word2Vec: {source['variant']} · {source['vocabulary_size']:,} pretrained words · {source['dimension']} dimensions")
    st.caption("Model C is a small supervised experiment. Similarity scores are not factual certainty. Model D uses an embedding encoder only.")
    source_pdf = ROOT / "data/raw/CourseContentsCSE_14_08_2022.pdf"
    if source_pdf.exists():
        st.download_button("Download source curriculum PDF", source_pdf.read_bytes(), file_name=source_pdf.name, mime="application/pdf")
