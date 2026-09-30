"""SciFact (BEIR format): download, load, and the fixed dev split."""

import csv
import json
import random
import urllib.request
import zipfile
from collections import defaultdict

from ladder import ROOT

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_DIR = ROOT / "data" / "scifact"
DEV_SIZE = 100
DEV_SEED = 13


def download():
    if SCIFACT_DIR.exists():
        return
    zip_path = SCIFACT_DIR.with_suffix(".zip")
    zip_path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(SCIFACT_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(SCIFACT_DIR.parent)
    zip_path.unlink()


def load_corpus():
    """doc_id -> "title text", the same concatenation BEIR uses for dense models."""
    corpus = {}
    with open(SCIFACT_DIR / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = f"{doc['title']} {doc['text']}".strip()
    return corpus


def load_queries():
    with open(SCIFACT_DIR / "queries.jsonl") as f:
        return {q["_id"]: q["text"] for q in map(json.loads, f)}


def load_qrels(split):
    """split is "train" or "test". Returns query_id -> {doc_id: relevance}."""
    qrels = defaultdict(dict)
    with open(SCIFACT_DIR / "qrels" / f"{split}.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return dict(qrels)


def dev_qrels():
    """A fixed 100-query sample of the train split, used for every tuning decision."""
    train = load_qrels("train")
    query_ids = random.Random(DEV_SEED).sample(sorted(train), DEV_SIZE)
    return {qid: train[qid] for qid in sorted(query_ids)}
