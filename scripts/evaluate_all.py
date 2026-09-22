import argparse
import hashlib

from src.datasets import read_queries, validate_datasets
from src.evaluation import evaluate_model, summarize, write_csv
from src.registry import create_model
from src.utils import ROOT, load_courses, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["a", "b", "c", "d"], default=["a", "b", "c", "d"])
    args = parser.parse_args()
    evaluation_path = ROOT / "data/queries/evaluation_queries.csv"
    rows = read_queries(evaluation_path)
    audit = validate_datasets(read_queries(ROOT / "data/queries/relevance_queries.csv"), rows, load_courses())
    all_details, comparison, categories, unavailable = [], [], [], {}
    for letter in args.models:
        try:
            model = create_model(f"model_{letter}")
            if not model.calibrated:
                raise ValueError("Calibrate this model on validation queries first")
            details = evaluate_model(model, rows)
            all_details.extend(details)
            for mode in ("retrieval_only", "end_to_end"):
                group = [r for r in details if r["mode"] == mode]
                comparison.append({"Model": model.model_id, "mode": mode, **summarize(group)})
                for category in sorted({r["category"] for r in group}):
                    categories.append({"Model": model.model_id, "mode": mode, "category": category,
                                       **summarize([r for r in group if r["category"] == category])})
            print(comparison[-2:], flush=True)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            unavailable[f"model_{letter}"] = str(exc)
            print(f"Model {letter} unavailable: {exc}", flush=True)
    output = ROOT / "artifacts/evaluation"
    write_csv(output / "comparison.csv", comparison)
    write_csv(output / "by_category.csv", categories)
    write_csv(output / "per_query.csv", all_details)
    write_json(output / "run_manifest.json", {
        "dataset": audit, "evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        "requested_models": args.models, "unavailable_models": unavailable,
        "retrieval_only": "Ranking before rejection and metadata routing; full-ranking MRR; unique course codes.",
        "end_to_end": "Final returned answers after rejection and deterministic metadata routing; MRR over returned results.",
        "limitations": "Small manually authored development benchmark. Scores are descriptive, not evidence of general superiority. Model selection must not use the test set."})
    try:
        from src.plotting import generate_all_plots
        generate_all_plots()
        print("Generated all evaluation plots in plotting/", flush=True)
    except Exception as exc:
        print(f"Plot generation notice: {exc}", flush=True)
    if unavailable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
