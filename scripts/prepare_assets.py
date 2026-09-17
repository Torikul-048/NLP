"""Explicit one-time downloads; the Streamlit UI never downloads a missing model."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

from src.utils import ROOT, write_json

SLIM_URL = "https://media.githubusercontent.com/media/eyaler/word2vec-slim/master/GoogleNews-vectors-negative300-SLIM.bin.gz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word2vec", choices=["slim", "full", "local"], default="slim")
    parser.add_argument("--path", help="Existing trusted Word2Vec .bin, .bin.gz or Gensim .kv file")
    parser.add_argument("--skip-sbert", action="store_true")
    parser.add_argument("--skip-wordnet", action="store_true")
    args = parser.parse_args()
    if not args.skip_wordnet:
        import nltk
        target = ROOT / ".cache/nltk"
        target.mkdir(parents=True, exist_ok=True)
        for package in ("wordnet", "omw-1.4"):
            if not nltk.download(package, download_dir=str(target)):
                raise RuntimeError("Could not download " + package)
    if args.word2vec == "local":
        if not args.path:
            parser.error("--word2vec local needs --path")
        path = Path(args.path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        source = "user-supplied pretrained Word2Vec"
    elif args.word2vec == "slim":
        path = ROOT / ".cache/word2vec/GoogleNews-vectors-negative300-SLIM.bin.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            print("Downloading the 300k-word Google News subset (about 264 MiB)...", flush=True)
            urllib.request.urlretrieve(SLIM_URL, partial)
            partial.replace(path)
        source = SLIM_URL
    else:
        from src.models.word_vectors import load_word_vectors
        vectors = load_word_vectors()
        path = None
        source = "gensim:word2vec-google-news-300"
    if path:
        from src.models.word_vectors import load_word_vectors
        vectors = load_word_vectors(str(path))
    try:
        stored_path = path.relative_to(ROOT).as_posix() if path else None
    except ValueError:
        stored_path = str(path)
    write_json(ROOT / "artifacts/vector_source.json", {
        "path": stored_path, "source": source, "vocabulary_size": len(vectors),
        "dimension": vectors.vector_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
        "variant": args.word2vec,
        "note": "SLIM preserves pretrained Google News vectors for roughly 300k English words; it is not the full 3-million-entry model."})
    print(f"Word2Vec ready: {len(vectors)} words, {vectors.vector_size} dimensions", flush=True)
    if not args.skip_sbert:
        from src.models.model_d_sbert import load_encoder
        encoder = load_encoder("sentence-transformers/all-MiniLM-L6-v2")
        print("Sentence-BERT ready:", encoder.get_sentence_embedding_dimension(), flush=True)


if __name__ == "__main__":
    main()
