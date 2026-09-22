"""CLI entry point to generate all evaluation plots and model-wise figures.

Usage:
    python -m scripts.generate_plots
    python -m scripts.generate_plots --output-dir plotting --dpi 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.plotting import generate_all_plots
from src.utils import ROOT


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots and confusion matrices.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plotting",
        help="Target folder for plots (default: 'plotting').",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for high resolution export (default: 300).",
    )
    args = parser.parse_args()

    eval_path = ROOT / "artifacts/evaluation/comparison.csv"
    if not eval_path.exists():
        print(f"[Warning] Evaluation results not found at {eval_path}.")
        print("Please run evaluation first: python -m scripts.evaluate_all")
        print("Plot generation will proceed if partial data is available.")

    print(f"Generating evaluation plots into: {args.output_dir}/ ...")
    try:
        result = generate_all_plots(output_dir=ROOT / args.output_dir, dpi=args.dpi)
        print("=======================================================")
        print(" Evaluation Plots Generated Successfully! ")
        print("=======================================================")
        print(f" Target Directory : {result['output_directory']}")
        print(f" Models Plotted   : {', '.join(result['models'])}")
        print(f" Total Figures    : {result['generated_files_count']}")
        print(" Model-wise folders:")
        for model in result["models"]:
            print(f"   - {args.output_dir}/{model}/")
        print(f"   - {args.output_dir}/comparison/")
        print("=======================================================")
    except Exception as exc:
        print(f"[Error] Failed to generate plots: {exc}", file=sys.stderr)
        # Return non-zero exit code if fatal error, but let bat file handle it
        sys.exit(1)


if __name__ == "__main__":
    main()
