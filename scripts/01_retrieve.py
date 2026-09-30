"""M1: BM25, dense, and hybrid runs for dev + test queries, checked against published numbers."""

from ladder import data, evaluate, rank, retrieve

corpus = data.load_corpus()
all_queries = data.load_queries()
splits = {"dev": data.dev_qrels(), "test": data.load_qrels("test")}
queries = {query_id: all_queries[query_id] for qrels in splits.values() for query_id in qrels}

runs = {"bm25": retrieve.bm25(corpus, queries), "dense": retrieve.dense(corpus, queries)}
runs["hybrid"] = rank.rrf(runs["bm25"], runs["dense"])
for name, run in runs.items():
    rank.save_run(name, run)

for split, qrels in splits.items():
    evaluate.print_table(split, qrels, runs)

test = {name: evaluate.metrics(splits["test"], run)["ndcg@10"] for name, run in runs.items()}
print("\nsanity gates (test NDCG@10)")
print(f"  bm25   {test['bm25']:.3f}  expected 0.65-0.70 (BEIR 0.665, Anserini 0.679)")
print(f"  dense  {test['dense']:.3f}  expected 0.70-0.73 (published 0.713)")
print(f"  hybrid {test['hybrid']:.3f}  expected >= roughly max(bm25, dense)")
