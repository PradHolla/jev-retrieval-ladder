"""Run files, RRF fusion, and reranking the head of a run with a second-stage scorer."""

import json

from ranx import Run, fuse

from ladder import ROOT

RUNS_DIR = ROOT / "runs"
RRF_K = 60
RERANK_DEPTH = 30


def save_run(name, run):
    RUNS_DIR.mkdir(exist_ok=True)
    with open(RUNS_DIR / f"{name}.json", "w") as f:
        json.dump(run, f)


def load_run(name):
    with open(RUNS_DIR / f"{name}.json") as f:
        return json.load(f)


def ranked_doc_ids(doc_scores):
    return sorted(doc_scores, key=doc_scores.get, reverse=True)


def candidates(run):
    return {query_id: ranked_doc_ids(docs)[:RERANK_DEPTH] for query_id, docs in run.items()}


def rrf(*runs):
    fused = fuse([Run.from_dict(run) for run in runs], norm=None, method="rrf", params={"k": RRF_K})
    return fused.to_dict()


def rerank(run, scores):
    """Order each query's top RERANK_DEPTH docs by `scores`; the tail keeps the run's order.

    sorted() is stable, so reranker ties keep the first-stage order. Output scores are
    descending ranks so that evaluation never has to break ties itself.
    """
    reranked = {}
    for query_id, docs in run.items():
        order = ranked_doc_ids(docs)
        head = sorted(order[:RERANK_DEPTH], key=lambda doc_id: -scores[query_id][doc_id])
        final = head + order[RERANK_DEPTH:]
        reranked[query_id] = {doc_id: len(final) - i for i, doc_id in enumerate(final)}
    return reranked
