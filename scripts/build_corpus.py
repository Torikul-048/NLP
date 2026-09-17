import argparse
import shutil

from src.corpus.corpus_builder import build_corpus
from src.utils import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str)
    args = parser.parse_args()
    target = ROOT / "data/raw/CourseContentsCSE_14_08_2022.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.pdf:
        from pathlib import Path
        source = Path(args.pdf).resolve()
        if source != target.resolve():
            shutil.copy2(source, target)
    elif not target.exists():
        candidates = list(ROOT.parent.glob("CourseContentsCSE*.pdf"))
        if len(candidates) != 1:
            parser.error("Supply the curriculum with --pdf PATH")
        shutil.copy2(candidates[0], target)
    print(build_corpus(target, ROOT / "data/processed"))


if __name__ == "__main__":
    main()
