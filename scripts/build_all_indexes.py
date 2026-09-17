import argparse

from src.registry import create_model
from src.utils import ROOT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["a", "b", "c", "d"], default=["a", "b", "c", "d"])
    parser.add_argument("--word2vec-path")
    args = parser.parse_args()
    failures = {}
    for letter in args.models:
        model_id = f"model_{letter}"
        try:
            model = create_model(model_id, load=letter == "c", vector_path=args.word2vec_path)
            if letter == "c":
                model.cache_documents()
                model.save()
            else:
                model.build()
            print(f"{model_id}: ready", flush=True)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            failures[model_id] = str(exc)
            print(f"{model_id}: unavailable: {exc}", flush=True)
    write_json(ROOT / "artifacts/build_status.json", {"requested": args.models, "failures": failures})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
