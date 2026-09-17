import argparse
import copy
import csv
import hashlib
import random

import numpy as np
import torch

from src.datasets import make_pairs, read_queries
from src.models.model_a_tfidf import TfidfRetriever
from src.models.model_c_bilstm import BiLSTMRetriever, RelevanceNetwork
from src.registry import configured_vector_path
from src.utils import ROOT, load_courses, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--word2vec-path")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    courses = load_courses()
    path = ROOT / "data/queries/relevance_queries.csv"
    rows = read_queries(path)
    # No reference to evaluation_queries.csv in this training process.
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    if not train_rows or not val_rows:
        raise ValueError("Both training and validation queries are required")
    if {r['group_id'] for r in train_rows} & {r['group_id'] for r in val_rows}:
        raise ValueError("Training/validation groups overlap")
    lexical = TfidfRetriever(courses).load()
    train_pairs = make_pairs(train_rows, courses, lexical, args.seed)
    val_pairs = make_pairs(val_rows, courses, lexical, args.seed + 1)
    model = BiLSTMRetriever(courses, vector_path=args.word2vec_path or configured_vector_path())
    model.config = dict(input_size=model.vectors.vector_size, hidden_size=args.hidden_size, layers=1,
                        dropout=0.3, max_length=args.max_length, seed=args.seed)
    model.network = RelevanceNetwork(input_size=model.vectors.vector_size, hidden_size=args.hidden_size)
    docs = [model.tensor(c["search_text"]) for c in courses]
    query_tensors = {p["query_id"]: model.tensor(p["query"]) for p in train_pairs + val_pairs}
    optimizer = torch.optim.Adam(model.network.parameters(), lr=1e-3, weight_decay=1e-5)
    positives = sum(p["label"] for p in train_pairs)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_pairs) - positives) / max(positives, 1)))
    history, best_loss, stale, best_state = [], float("inf"), 0, None

    def batch_loss(batch):
        logits = model.network([query_tensors[p["query_id"]] for p in batch], [docs[p["document_index"]] for p in batch])
        return loss_fn(logits, torch.tensor([p["label"] for p in batch], dtype=torch.float32))

    for epoch in range(args.epochs):
        model.network.train()
        random.shuffle(train_pairs)
        losses = []
        for start in range(0, len(train_pairs), args.batch_size):
            optimizer.zero_grad()
            loss = batch_loss(train_pairs[start:start + args.batch_size])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
        model.network.eval()
        with torch.no_grad():
            val_loss = sum(batch_loss(val_pairs[s:s + args.batch_size]).item() * len(val_pairs[s:s + args.batch_size])
                           for s in range(0, len(val_pairs), args.batch_size)) / len(val_pairs)
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_loss=val_loss))
        print(history[-1], flush=True)
        if val_loss < best_loss - 1e-4:
            best_loss, stale, best_state = val_loss, 0, copy.deepcopy(model.network.state_dict())
        else:
            stale += 1
            if stale >= 5:
                break
    model.network.load_state_dict(best_state)
    model.cache_documents()
    model.save()
    write_json(model.directory / "training_history.json", history)
    write_json(model.directory / "training_provenance.json", {
        "seed": args.seed, "training_query_ids": [r["query_id"] for r in train_rows],
        "validation_query_ids": [r["query_id"] for r in val_rows],
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_pairs": len(train_pairs), "validation_pairs": len(val_pairs),
        "best_validation_loss": best_loss, "early_stopping_patience": 5,
        "negative_sampling": "3 TF-IDF hard + 2 random per query; all declared positives excluded"})
    with (model.directory / "training_pairs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(train_pairs[0]))
        writer.writeheader()
        writer.writerows(train_pairs)


if __name__ == "__main__":
    main()
