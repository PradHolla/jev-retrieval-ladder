"""M3: Jev on the dev split with rubrics A and B. Picks the rubric and both thresholds on dev
and writes them to config/frozen.json before any test-split Jev call."""

import json
from collections import defaultdict
from datetime import date

import numpy as np

from ladder import ROOT, data, evaluate, jev, rank

GPT_RESEARCHER_THRESHOLD = 1.5

corpus = data.load_corpus()
queries = data.load_queries()
dev = data.dev_qrels()
hybrid = {query_id: docs for query_id, docs in rank.load_run("hybrid").items() if query_id in dev}
candidates = rank.candidates(hybrid)
pairs = [(query_id, doc_id) for query_id in dev for doc_id in candidates[query_id]]

runs = {name: rank.load_run(name) for name in ["hybrid", "ce", "hybrid+ce"]}
jev_scores = {}
for rubric in ["A", "B"]:
    requests = [jev.build_request(jev.MODEL, rubric, queries[q], corpus[d]) for q, d in pairs]
    n_missing, estimate = jev.estimate_usd(requests)
    print(f"\nrubric {rubric}: {n_missing} uncached calls, estimated ${estimate:.3f} "
          f"(spent so far ${jev.spent_usd():.4f})")
    entries = jev.call_many(requests)

    scores = defaultdict(dict)
    for (query_id, doc_id), entry in zip(pairs, entries):
        scores[query_id][doc_id] = entry["response"]["answers"]["relevance"]["score"]
    jev_scores[rubric] = scores
    rank.save_run(f"jev_{rubric}_scores_dev", scores)
    runs[f"jev_{rubric}"] = rank.rerank(hybrid, scores)
    runs[f"hybrid+jev_{rubric}"] = rank.rrf(hybrid, runs[f"jev_{rubric}"])

    latency = [entry["latency_ms"] for entry in entries]
    distinct = [len(set(doc_scores.values())) for doc_scores in scores.values()]
    tied_at_top = [sum(s == max(doc_scores.values()) for s in doc_scores.values())
                   for doc_scores in scores.values()]
    print(f"  cost ${sum(e['response']['usage']['cost'] for e in entries):.4f}, "
          f"mean input tokens {np.mean([e['response']['usage']['input_tokens'] for e in entries]):.0f}, "
          f"latency p50 {np.percentile(latency, 50):.0f} ms p95 {np.percentile(latency, 95):.0f} ms")
    print(f"  distinct scores among {rank.RERANK_DEPTH} candidates: mean {np.mean(distinct):.1f}; "
          f"candidates tied at the top score: mean {np.mean(tied_at_top):.2f}")

evaluate.print_table("dev", dev, runs)

ndcg = {rubric: evaluate.metrics(dev, runs[f"jev_{rubric}"])["ndcg@10"] for rubric in ["A", "B"]}
rubric = max(ndcg, key=ndcg.get)
print(f"\nchosen rubric: {rubric} (dev NDCG@10 A={ndcg['A']:.3f}, B={ndcg['B']:.3f})")

jev_sweep = evaluate.threshold_sweep(dev, jev_scores[rubric])
thresholds = {"jev": evaluate.best_f1(jev_sweep),
              "ce": evaluate.best_f1(evaluate.threshold_sweep(dev, rank.load_run("ce_scores")))}
thresholds[f"jev at GPT Researcher's {GPT_RESEARCHER_THRESHOLD}"] = evaluate.at_threshold(
    jev_sweep, GPT_RESEARCHER_THRESHOLD
)
print("\ndev thresholds (pooled over queries)")
for name, point in thresholds.items():
    print(f"  {name:<30} t={point['threshold']:.4f} precision={point['precision']:.3f} "
          f"recall={point['recall']:.3f} f1={point['f1']:.3f} kept/query={point['kept_per_query']:.2f}")

config = {
    "jev_model": jev.MODEL,
    "jev_endpoint": jev.ENDPOINT,
    "rubric": rubric,
    "rerank_depth": rank.RERANK_DEPTH,
    "rrf_k": rank.RRF_K,
    "jev_threshold": thresholds["jev"]["threshold"],
    "ce_threshold": thresholds["ce"]["threshold"],
    "dev_queries": len(dev),
    "frozen_on": date.today().isoformat(),
}
(ROOT / "config").mkdir(exist_ok=True)
with open(ROOT / "config" / "frozen.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"\nwrote config/frozen.json; total Jev spend ${jev.spent_usd():.4f}")
