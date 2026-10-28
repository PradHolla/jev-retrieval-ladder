"""Claude Haiku 5.5 checks each claim against the passages one pipeline retrieved.
Run `08_verdicts.py dev` to check the prompt, then `08_verdicts.py test` once."""

import json
import sys

import numpy as np

from ladder import ROOT, claude, data, rank, verdict

split = sys.argv[1]
with open(ROOT / "config" / "frozen.json") as f:
    config = json.load(f)

data.download_claims()
corpus = data.load_corpus()
queries = data.load_queries()
hybrid = rank.load_run("hybrid")
if split == "dev":
    qrels, claims = data.dev_qrels(), data.load_claims("train")
    jev_scores = rank.load_run("jev_A_scores_dev")
    jev = rank.rerank({query_id: hybrid[query_id] for query_id in qrels}, jev_scores)
elif split == "test":
    qrels, claims = data.load_qrels("test"), data.load_claims("test")
    jev_scores = rank.load_run("jev_scores_test")
    jev = rank.load_run("jev")
else:
    raise ValueError(f"unknown split {split!r}")

contexts = verdict.contexts(qrels, hybrid, rank.load_run("ce"), jev, jev_scores, config["jev_threshold"])
requests = {pipeline: [verdict.build_request(queries[q], [corpus[d] for d in docs[q]]) for q in qrels]
            for pipeline, docs in contexts.items()}
n_missing, estimate = claude.estimate_usd([r for rs in requests.values() for r in rs])
print(f"{split}: {n_missing} uncached calls, estimated ${estimate:.3f} "
      f"(Claude spent so far ${claude.spent_usd():.4f})")
claude.call_many([r for rs in requests.values() for r in rs])

gold = np.array([claims[q]["verdict"] for q in qrels])
print(f"\n{split} ({len(qrels)} claims; majority-class accuracy "
      f"{max(np.mean(gold == v) for v in verdict.VERDICTS):.3f})")
print(f"{'pipeline':<15}{'accuracy':>9}{'macro-F1':>9}{'passages':>9}{'in tok':>8}{'out tok':>8}{'failed':>7}")
predictions = {}
for pipeline, entries in ((p, claude.call_many(rs)) for p, rs in requests.items()):
    predicted = np.array([verdict.parse(e["response"]) for e in entries], dtype=object)
    predictions[pipeline] = dict(zip(qrels, predicted.tolist()))
    usage = [e["response"]["usage"] for e in entries]
    print(f"{pipeline:<15}{np.mean(predicted == gold):>9.3f}{verdict.macro_f1(predicted, gold):>9.3f}"
          f"{np.mean([len(contexts[pipeline][q]) for q in qrels]):>9.2f}"
          f"{np.mean([u['input_tokens'] for u in usage]):>8.0f}{np.mean([u['output_tokens'] for u in usage]):>8.0f}"
          f"{sum(v is None for v in predicted):>7}")
rank.save_run(f"verdicts_{split}", predictions)
print(f"\ntotal Claude spend ${claude.spent_usd():.4f}")
