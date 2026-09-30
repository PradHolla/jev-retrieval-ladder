"""Ranking metrics (via ranx) and the pooled score-threshold sweep."""

import numpy as np
from ranx import Qrels, Run, evaluate

METRICS = ["ndcg@10", "mrr@10", "recall@10", "recall@30", "recall@100"]
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 13


def metrics(qrels, run):
    return evaluate(qrels, {query_id: run[query_id] for query_id in qrels}, METRICS)


def per_query(qrels, run):
    """metric -> {query_id: score}. Read by query id from ranx's stored results, because the
    order of ranx's per-query arrays depends on the input type."""
    ranx_run = Run.from_dict({query_id: run[query_id] for query_id in qrels})
    evaluate(Qrels.from_dict(qrels), ranx_run, METRICS)
    return {m: {query_id: float(s) for query_id, s in ranx_run.scores[m].items()} for m in METRICS}


def resample_indices(n_queries):
    """One shared set of query resamples, so every system and difference is paired."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(0, n_queries, size=(BOOTSTRAP_SAMPLES, n_queries))


def bootstrap_ci(values, resamples):
    means = values[resamples].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def print_table(title, qrels, runs):
    print(f"\n{title} ({len(qrels)} queries)")
    print(f"{'system':<16}" + "".join(f"{m:>12}" for m in METRICS))
    for name, run in runs.items():
        scores = metrics(qrels, run)
        print(f"{name:<16}" + "".join(f"{scores[m]:>12.3f}" for m in METRICS))


def threshold_sweep(qrels, scores):
    """Keep every candidate scoring >= t, for each distinct score t, pooled over queries.

    `scores` maps query_id -> {doc_id: score} over the reranked candidates. Recall counts all
    relevant docs, so its ceiling is the candidate pool's recall.
    """
    values, relevant = [], []
    for query_id in qrels:
        for doc_id, score in scores[query_id].items():
            values.append(score)
            relevant.append(doc_id in qrels[query_id])
    values, relevant = np.array(values), np.array(relevant)
    total_relevant = sum(len(docs) for docs in qrels.values())

    thresholds = np.unique(values)[::-1]
    kept = np.array([(values >= t).sum() for t in thresholds])
    hits = np.array([relevant[values >= t].sum() for t in thresholds])
    precision = hits / kept
    recall = hits / total_relevant
    return {
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "kept_per_query": kept / len(qrels),
    }


def best_f1(sweep):
    i = int(np.nanargmax(sweep["f1"]))
    return {key: float(values[i]) for key, values in sweep.items()}


def at_threshold(sweep, t):
    """The sweep point that keeps exactly the candidates scoring >= t."""
    i = np.where(sweep["threshold"] >= t)[0][-1]
    return {key: float(values[i]) for key, values in sweep.items()} | {"threshold": t}
