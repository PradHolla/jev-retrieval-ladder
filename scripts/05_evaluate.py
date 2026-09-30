"""M5: test-split metrics with paired bootstrap 95% CIs, threshold curves, and Jev / cross-encoder
cost, latency and determinism, all written to results/results.json. Makes no API calls."""

import json

import numpy as np

from ladder import ROOT, data, evaluate, jev, rank

SYSTEMS = {
    "bm25": "bm25",
    "dense": "dense",
    "hybrid": "hybrid",
    "ce": "ce",
    "hybrid+ce": "hybrid+ce",
    "jev": "jev",
    "hybrid+jev": "hybrid+jev",
    "ce (fp32, 512 tokens)": "ce_fp32_512/ce",
    "hybrid+ce (fp32, 512 tokens)": "ce_fp32_512/hybrid+ce",
}
DELTAS = [
    ("jev", "ce"),
    ("jev", "hybrid"),
    ("ce", "hybrid"),
    ("hybrid+jev", "jev"),
    ("hybrid+ce", "ce"),
    ("hybrid+jev", "hybrid+ce"),
]
DELTA_METRICS = ["ndcg@10", "mrr@10", "recall@10"]
MATCHED_KEPT_PER_QUERY = [0.5, 1, 2, 3, 5]
GPT_RESEARCHER_THRESHOLD = 1.5

with open(ROOT / "config" / "frozen.json") as f:
    config = json.load(f)
corpus = data.load_corpus()
queries = data.load_queries()
test = data.load_qrels("test")
query_ids = sorted(test)

runs = {name: rank.load_run(path) for name, path in SYSTEMS.items()}
values = {}
for name, run in runs.items():
    scores = evaluate.per_query(test, run)
    values[name] = {m: np.array([scores[m][q] for q in query_ids]) for m in evaluate.METRICS}
resamples = evaluate.resample_indices(len(query_ids))

systems = {
    name: {m: evaluate.bootstrap_ci(values[name][m], resamples) for m in evaluate.METRICS}
    for name in runs
}
deltas = {
    f"{a} - {b}": {m: evaluate.bootstrap_ci(values[a][m] - values[b][m], resamples) for m in DELTA_METRICS}
    for a, b in DELTAS
}

jev_scores = rank.load_run("jev_scores_test")
ce_scores = rank.load_run("ce_scores")
sweeps = {"jev": evaluate.threshold_sweep(test, jev_scores), "ce": evaluate.threshold_sweep(test, ce_scores)}
thresholds = {
    "frozen_on_dev": {
        "jev": evaluate.at_threshold(sweeps["jev"], config["jev_threshold"]),
        "ce": evaluate.at_threshold(sweeps["ce"], config["ce_threshold"]),
    },
    "jev_at_gpt_researcher_1.5": evaluate.at_threshold(sweeps["jev"], GPT_RESEARCHER_THRESHOLD),
    "matched_kept_per_query": {
        str(k): {
            name: {
                "precision": float(np.interp(k, sweep["kept_per_query"], sweep["precision"])),
                "recall": float(np.interp(k, sweep["kept_per_query"], sweep["recall"])),
            }
            for name, sweep in sweeps.items()
        }
        for k in MATCHED_KEPT_PER_QUERY
    },
    "curves": {name: {k: v.tolist() for k, v in sweep.items()} for name, sweep in sweeps.items()},
}

candidates = rank.candidates({q: rank.load_run("hybrid")[q] for q in test})
requests = [jev.build_request(jev.MODEL, config["rubric"], queries[q], corpus[d])
            for q in test for d in candidates[q]]
assert jev.estimate_usd(requests)[0] == 0, "test responses missing from cache"
entries = jev.call_many(requests)
repeats = jev.cache_entries("repeat")
originals = jev.call_many([r["request"] for r in repeats])
repeat_diffs = [abs(o["response"]["answers"]["relevance"]["score"] - r["response"]["answers"]["relevance"]["score"])
                for o, r in zip(originals, repeats)]
latency = [e["latency_ms"] for e in entries]
test_cost = sum(e["response"]["usage"]["cost"] for e in entries)
ce_latency = json.load(open(ROOT / "results" / "ce_latency_ms.json"))

dev = data.dev_qrels()
dev_hybrid = {q: runs["hybrid"][q] for q in dev}
dev_runs = {name: runs[name] for name in ["hybrid", "ce", "hybrid+ce"]}
for rubric in ["A", "B"]:
    dev_runs[f"jev rubric {rubric}"] = rank.rerank(dev_hybrid, rank.load_run(f"jev_{rubric}_scores_dev"))
    dev_runs[f"hybrid+jev rubric {rubric}"] = rank.rrf(dev_hybrid, dev_runs[f"jev rubric {rubric}"])
dev_ndcg = {name: evaluate.metrics(dev, run)["ndcg@10"] for name, run in dev_runs.items()}

results = {
    "config": config,
    "test_queries": len(query_ids),
    "bootstrap": {"samples": evaluate.BOOTSTRAP_SAMPLES, "seed": evaluate.BOOTSTRAP_SEED, "ci": "95% percentile"},
    "systems": systems,
    "deltas": deltas,
    "dev_ndcg@10_used_for_choices": {"queries": len(dev), "ndcg@10": dev_ndcg},
    "thresholds": thresholds,
    "jev": {
        "models_returned": sorted({e["response"]["model"] for e in entries}),
        "calls_on_test": len(entries),
        "test_cost_usd": test_cost,
        "cost_per_query_usd": test_cost / len(query_ids),
        "total_spend_usd_all_runs": jev.spent_usd(),
        "mean_input_tokens": float(np.mean([e["response"]["usage"]["input_tokens"] for e in entries])),
        "latency_ms_per_call": {"p50": float(np.percentile(latency, 50)), "p95": float(np.percentile(latency, 95))},
        "calls_made_between": [min(e["timestamp"] for e in entries), max(e["timestamp"] for e in entries)],
        "distinct_scores_per_query_mean": float(np.mean([len(set(s.values())) for s in jev_scores.values()])),
        "determinism": {
            "repeated_calls": len(repeats),
            "identical_scores": sum(d == 0 for d in repeat_diffs),
            "max_abs_score_difference": max(repeat_diffs),
            "mean_abs_score_difference": float(np.mean(repeat_diffs)),
        },
    },
    "cross_encoder": {
        "model": "BAAI/bge-reranker-v2-m3",
        "setup": "fp16, max_length 8192 (no pair truncated), NVIDIA A10G (EC2 g5.2xlarge)",
        "latency_ms_per_query_30_pairs": {"p50": float(np.percentile(ce_latency, 50)),
                                          "p95": float(np.percentile(ce_latency, 95))},
    },
}
(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results" / "results.json", "w") as f:
    json.dump(results, f, indent=2)


def fmt(ci):
    return f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


print(f"test ({len(query_ids)} queries), mean [95% CI]")
print(f"{'system':<30}{'ndcg@10':>24}{'mrr@10':>24}{'recall@10':>24}")
for name, metrics in systems.items():
    print(f"{name:<30}" + "".join(f"{fmt(metrics[m]):>24}" for m in DELTA_METRICS))

print("\npaired differences, mean [95% CI]")
for name, metrics in deltas.items():
    print(f"{name:<30}" + "".join(f"{fmt(metrics[m]):>24}" for m in DELTA_METRICS))

print("\nthresholds on test")
for name, point in thresholds["frozen_on_dev"].items():
    print(f"  {name} frozen t={point['threshold']:.4f}: precision {point['precision']:.3f}, "
          f"recall {point['recall']:.3f}, kept/query {point['kept_per_query']:.2f}")
point = thresholds["jev_at_gpt_researcher_1.5"]
print(f"  jev at 1.5: precision {point['precision']:.3f}, recall {point['recall']:.3f}, "
      f"kept/query {point['kept_per_query']:.2f}")
print("  precision at matched kept/query (jev vs ce):")
for k, by_system in thresholds["matched_kept_per_query"].items():
    print(f"    {k:>3}: {by_system['jev']['precision']:.3f} vs {by_system['ce']['precision']:.3f}")

d = results["jev"]["determinism"]
print(f"\njev determinism: {d['identical_scores']}/{d['repeated_calls']} identical, "
      f"mean |diff| {d['mean_abs_score_difference']:.3f}, max {d['max_abs_score_difference']:.3f}")
print(f"wrote results/results.json; total Jev spend ${jev.spent_usd():.4f}")
