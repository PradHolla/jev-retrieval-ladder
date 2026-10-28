"""Test-split claim verification: accuracy with paired-bootstrap CIs, per gold label, whether
the evidence reached Claude, and tokens and cost per claim. Writes results/verdicts.json."""

import json

import numpy as np

from ladder import ROOT, claude, data, evaluate, rank, verdict

DIFFERENCES = [("jev", "ce"), ("jev", "hybrid"), ("ce", "hybrid"), ("jev_threshold", "jev"), ("cited", "jev")]

with open(ROOT / "config" / "frozen.json") as f:
    config = json.load(f)
corpus = data.load_corpus()
queries = data.load_queries()
qrels = data.load_qrels("test")
claims = data.load_claims("test")
query_ids = list(qrels)
predictions = rank.load_run("verdicts_test")
contexts = verdict.contexts(qrels, rank.load_run("hybrid"), rank.load_run("ce"), rank.load_run("jev"),
                            rank.load_run("jev_scores_test"), config["jev_threshold"])

gold = np.array([claims[q]["verdict"] for q in query_ids])
has_evidence = gold != "NOT_ENOUGH_INFO"
resamples = evaluate.resample_indices(len(query_ids))
correct = {p: np.array([predictions[p][q] == claims[q]["verdict"] for q in query_ids], dtype=float)
           for p in contexts}


def mean_or_none(values):
    """None for "cited", whose context never misses the evidence."""
    return float(values.mean()) if len(values) else None


systems = {}
for pipeline, docs in contexts.items():
    predicted = np.array([predictions[pipeline][q] for q in query_ids], dtype=object)
    evidence_in_context = np.array([bool(set(docs[q]) & set(claims[q]["evidence"])) for q in query_ids])
    entries = claude.call_many([verdict.build_request(queries[q], [corpus[d] for d in docs[q]])
                                for q in query_ids])
    usage = [e["response"]["usage"] for e in entries]
    systems[pipeline] = {
        "accuracy": evaluate.bootstrap_ci(correct[pipeline], resamples),
        "macro_f1": verdict.macro_f1(predicted, gold),
        "accuracy_by_gold": {v: float(correct[pipeline][gold == v].mean()) for v in verdict.VERDICTS},
        "evidence_in_context": float(evidence_in_context[has_evidence].mean()),
        "accuracy_evidence_in_context": float(correct[pipeline][has_evidence & evidence_in_context].mean()),
        "accuracy_evidence_missing": mean_or_none(correct[pipeline][has_evidence & ~evidence_in_context]),
        "cited_in_context_not_enough_info": float(np.mean([bool(set(docs[q]) & set(qrels[q]))
                                                           for q in np.array(query_ids)[~has_evidence]])),
        "refusals": int(sum(v is None for v in predicted)),
        "passages_per_claim": float(np.mean([len(docs[q]) for q in query_ids])),
        "input_tokens_per_claim": float(np.mean([u["input_tokens"] for u in usage])),
        "output_tokens_per_claim": float(np.mean([u["output_tokens"] for u in usage])),
        "cost_per_claim_usd": float(np.mean([claude.cost_usd(u) for u in usage])),
        "latency_ms_p50": float(np.percentile([e["latency_ms"] for e in entries], 50)),
    }
deltas = {f"{a} - {b}": evaluate.bootstrap_ci(correct[a] - correct[b], resamples) for a, b in DIFFERENCES}
deltas_by_evidence = {}
for subset, mask in [("has_evidence", has_evidence), ("not_enough_info", ~has_evidence)]:
    subset_resamples = evaluate.resample_indices(int(mask.sum()))
    deltas_by_evidence[subset] = {f"{a} - {b}": evaluate.bootstrap_ci((correct[a] - correct[b])[mask],
                                                                       subset_resamples)
                                  for a, b in DIFFERENCES}

results = {
    "model": claude.MODEL,
    "effort": verdict.EFFORT,
    "context_size": verdict.CONTEXT_SIZE,
    "claims": len(query_ids),
    "gold_counts": {v: int(np.sum(gold == v)) for v in verdict.VERDICTS},
    "majority_class_accuracy": float(max(np.mean(gold == v) for v in verdict.VERDICTS)),
    "claude_spend_usd": claude.spent_usd(),
    "systems": systems,
    "deltas": deltas,
    "deltas_by_evidence": deltas_by_evidence,
}
with open(ROOT / "results" / "verdicts.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"test, {len(query_ids)} claims {results['gold_counts']}")
print(f"{'pipeline':<15}{'accuracy [95% CI]':>24}{'evidence in ctx':>17}{'acc if in':>11}"
      f"{'acc if missing':>16}{'NEI acc':>9}{'in tok':>8}")
for pipeline, s in systems.items():
    a = s["accuracy"]
    missing = s["accuracy_evidence_missing"]
    print(f"{pipeline:<15}{a['mean']:>9.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]"
          f"{s['evidence_in_context']:>17.3f}{s['accuracy_evidence_in_context']:>11.3f}"
          f"{'-' if missing is None else f'{missing:.3f}':>16}{s['accuracy_by_gold']['NOT_ENOUGH_INFO']:>9.3f}"
          f"{s['input_tokens_per_claim']:>8.0f}")
for title, table in [("all claims", deltas)] + [(k, v) for k, v in deltas_by_evidence.items()]:
    print(f"\npaired accuracy differences, {title}")
    for name, d in table.items():
        print(f"  {name:<22}{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
print(f"\nwrote results/verdicts.json; total Claude spend ${claude.spent_usd():.4f}")
