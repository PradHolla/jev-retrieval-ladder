"""Two Jev calls on one test claim (a relevant and an irrelevant abstract) to confirm the
endpoint, the pinned model id, and the response shape."""

import json

from ladder import data, jev

corpus = data.load_corpus()
queries = data.load_queries()
qrels = data.load_qrels("test")

query_id = sorted(qrels)[0]
claim = queries[query_id]
relevant_id = next(iter(qrels[query_id]))
irrelevant_id = next(doc_id for doc_id in sorted(corpus) if doc_id not in qrels[query_id])

relevant = jev.call(jev.build_request(jev.MODEL, "A", claim, corpus[relevant_id]))
irrelevant = jev.call(jev.build_request(jev.MODEL, "A", claim, corpus[irrelevant_id]))

print(f"claim: {claim}")
print(json.dumps(relevant["response"], indent=2))
for label, entry in [("relevant", relevant), ("irrelevant", irrelevant)]:
    answer = entry["response"]["answers"]["relevance"]
    print(f"{label:>10}: score={answer['score']:.3f} latency={entry['latency_ms']:.0f}ms")
print(f"total Jev spend so far: ${jev.spent_usd():.6f}")
