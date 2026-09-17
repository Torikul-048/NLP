import argparse

from src.datasets import read_queries
from src.evaluation import calibrate_threshold
from src.registry import create_model
from src.utils import ROOT, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["a", "b", "c", "d"], default=["a", "b", "c", "d"])
    args = parser.parse_args()
    rows = [r for r in read_queries(ROOT / "data/queries/relevance_queries.csv") if r["split"] == "validation"]
    path = ROOT / "artifacts/thresholds.json"
    thresholds = read_json(path) if path.exists() else {}
    for letter in args.models:
        model = create_model(f"model_{letter}")
        thresholds[model.model_id] = calibrate_threshold(model, rows)
        write_json(path, thresholds)
        print(model.model_id, thresholds[model.model_id]["threshold"], flush=True)


if __name__ == "__main__":
    main()
