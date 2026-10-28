"""SciFact (BEIR format): download, load, and the fixed dev split."""

import csv
import json
import random
import tarfile
import urllib.request
import zipfile
from collections import defaultdict

from ladder import ROOT

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_DIR = ROOT / "data" / "scifact"
# The original release carries the SUPPORT/CONTRADICT labels that BEIR drops. BEIR's test split
# is SciFact's labeled dev split (SciFact's own test labels are hidden).
CLAIMS_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
CLAIMS_DIR = ROOT / "data" / "scifact_claims"
CLAIM_FILES = {"train": "claims_train.jsonl", "test": "claims_dev.jsonl"}
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


def download_claims():
    if CLAIMS_DIR.exists():
        return
    tar_path = CLAIMS_DIR.with_suffix(".tar.gz")
    urllib.request.urlretrieve(CLAIMS_URL, tar_path)
    with tarfile.open(tar_path) as tf:
        tf.extractall(CLAIMS_DIR, filter="data")
    tar_path.unlink()


def load_claims(split):
    """split is "train" or "test". Returns query_id -> {"verdict", "evidence": [doc_id]}.

    A claim with no rationale in any cited abstract is NOT_ENOUGH_INFO. No claim mixes
    SUPPORT and CONTRADICT; the unpacking below fails if one ever does.
    """
    claims = {}
    with open(CLAIMS_DIR / "data" / CLAIM_FILES[split]) as f:
        for claim in map(json.loads, f):
            labels = {e["label"] for rationales in claim["evidence"].values() for e in rationales}
            verdict = "NOT_ENOUGH_INFO"
            if labels:
                (verdict,) = labels
            claims[str(claim["id"])] = {"verdict": verdict, "evidence": list(claim["evidence"])}
    return claims


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
