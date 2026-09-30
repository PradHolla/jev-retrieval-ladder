"""M4: Jev on the test split, once, with the configuration frozen on dev. Also re-calls 20
cached pairs to check whether Jev's answers are deterministic."""

import json
import random
from collections import defaultdict

import numpy as np

from ladder import ROOT, data, evaluate, jev, rank

DETERMINISM_PAIRS = 20
DETERMINISM_SEED = 13

with open(ROOT / "config" / "frozen.json") as f:
    config = json.load(f)
assert config["jev_model"] == jev.MODEL

corpus = data.load_corpus()
queries = data.load_queries()
test = data.load_qrels("test")
hybrid = {query_id: docs for query_id, docs in rank.load_run("hybrid").items() if query_id in test}
candidates = rank.candidates(hybrid)
pairs = [(query_id, doc_id) for query_id in test for doc_id in candidates[query_id]]
requests = [jev.build_request(jev.MODEL, config["rubric"], queries[q], corpus[d]) for q, d in pairs]

n_missing, estimate = jev.estimate_usd(requests)
print(f"rubric {config['rubric']}: {n_missing} uncached calls, estimated ${estimate:.3f} "
      f"(spent so far ${jev.spent_usd():.4f})")
entries = jev.call_many(requests)

scores = defaultdict(dict)
for (query_id, doc_id), entry in zip(pairs, entries):
    scores[query_id][doc_id] = entry["response"]["answers"]["relevance"]["score"]
rank.save_run("jev_scores_test", scores)
runs = {name: rank.load_run(name) for name in ["hybrid", "ce", "hybrid+ce"]}
runs["jev"] = rank.rerank(hybrid, scores)
runs["hybrid+jev"] = rank.rrf(hybrid, runs["jev"])
rank.save_run("jev", runs["jev"])
rank.save_run("hybrid+jev", runs["hybrid+jev"])

latency = [entry["latency_ms"] for entry in entries]
print(f"  cost ${sum(e['response']['usage']['cost'] for e in entries):.4f}, "
      f"mean input tokens {np.mean([e['response']['usage']['input_tokens'] for e in entries]):.0f}, "
      f"latency p50 {np.percentile(latency, 50):.0f} ms p95 {np.percentile(latency, 95):.0f} ms")

evaluate.print_table("test", test, runs)

print("\ntest thresholds, frozen on dev (pooled over queries)")
for name, doc_scores, t in [("jev", scores, config["jev_threshold"]),
                            ("ce", rank.load_run("ce_scores"), config["ce_threshold"])]:
    point = evaluate.at_threshold(evaluate.threshold_sweep(test, doc_scores), t)
    print(f"  {name:<4} t={t:.4f} precision={point['precision']:.3f} recall={point['recall']:.3f} "
          f"f1={point['f1']:.3f} kept/query={point['kept_per_query']:.2f}")

sample = random.Random(DETERMINISM_SEED).sample(range(len(requests)), DETERMINISM_PAIRS)
repeats = jev.call_many([requests[i] for i in sample], cache="repeat")
diffs = [abs(entries[i]["response"]["answers"]["relevance"]["score"]
             - repeat["response"]["answers"]["relevance"]["score"])
         for i, repeat in zip(sample, repeats)]
print(f"\ndeterminism: {sum(d == 0 for d in diffs)}/{DETERMINISM_PAIRS} repeated calls returned "
      f"the same score; max |difference| {max(diffs):.3f}")
print(f"total Jev spend ${jev.spent_usd():.4f}")
