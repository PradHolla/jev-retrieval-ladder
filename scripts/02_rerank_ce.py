"""M2: bge-reranker-v2-m3 over the hybrid top 30, its fusion with hybrid, and its dev threshold."""

import json
import time

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from ladder import ROOT, data, evaluate, rank

CE_MODEL = "BAAI/bge-reranker-v2-m3"
# The model's position limit. The longest candidate pair is 2,298 tokens, so nothing is
# truncated and the cross-encoder reads the same full text Jev does.
MAX_LENGTH = 8192

corpus = data.load_corpus()
queries = data.load_queries()
splits = {"dev": data.dev_qrels(), "test": data.load_qrels("test")}
hybrid = rank.load_run("hybrid")
candidates = rank.candidates(hybrid)

# fp16, as in the model card's FlagReranker example (use_fp16=True).
model = CrossEncoder(CE_MODEL, max_length=MAX_LENGTH, model_kwargs={"torch_dtype": torch.float16})
assert next(model.parameters()).dtype == torch.float16
scores, latency_ms, truncated = {}, [], 0
for query_id, doc_ids in tqdm(candidates.items()):
    pairs = [(queries[query_id], corpus[doc_id]) for doc_id in doc_ids]
    truncated += sum(len(model.tokenizer(q, p)["input_ids"]) > MAX_LENGTH for q, p in pairs)
    start = time.perf_counter()
    probabilities = model.predict(pairs, batch_size=len(pairs))
    latency_ms.append((time.perf_counter() - start) * 1000)
    scores[query_id] = dict(zip(doc_ids, map(float, probabilities)))

assert all(np.isfinite(s) for doc_scores in scores.values() for s in doc_scores.values())
rank.save_run("ce_scores", scores)
(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results" / "ce_latency_ms.json", "w") as f:
    json.dump(latency_ms, f)

runs = {"hybrid": hybrid, "ce": rank.rerank(hybrid, scores)}
runs["hybrid+ce"] = rank.rrf(hybrid, runs["ce"])
for name in ["ce", "hybrid+ce"]:
    rank.save_run(name, runs[name])

for split, qrels in splits.items():
    evaluate.print_table(split, qrels, runs)

n_pairs = sum(map(len, candidates.values()))
print(f"\npairs truncated at {MAX_LENGTH} tokens: {truncated}/{n_pairs} ({truncated / n_pairs:.1%})")
print(f"latency per query (30 pairs, {model.device}, fp16): p50 {np.percentile(latency_ms, 50):.0f} ms, "
      f"p95 {np.percentile(latency_ms, 95):.0f} ms")

best = evaluate.best_f1(evaluate.threshold_sweep(splits["dev"], scores))
print(f"\ndev max-F1 threshold: t={best['threshold']:.4f} precision={best['precision']:.3f} "
      f"recall={best['recall']:.3f} f1={best['f1']:.3f} kept/query={best['kept_per_query']:.2f}")
