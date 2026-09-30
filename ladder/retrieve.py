"""First-stage retrievers over the full corpus: BM25 and dense (bge-small)."""

import bm25s
import numpy as np
import Stemmer
from sentence_transformers import SentenceTransformer

DEPTH = 100
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
# From the model card; the repo's sentence-transformers config defines no prompts.
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


def bm25(corpus, queries):
    doc_ids, query_ids = list(corpus), list(queries)
    stemmer = Stemmer.Stemmer("english")

    def tokenize(texts):
        return bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)

    # Lucene/Elasticsearch defaults, the configuration behind BEIR's published BM25 numbers.
    retriever = bm25s.BM25(k1=1.2, b=0.75, method="lucene")
    retriever.index(tokenize(list(corpus.values())), show_progress=False)
    indices, scores = retriever.retrieve(
        tokenize([queries[q] for q in query_ids]), k=DEPTH, show_progress=False
    )
    return {
        query_id: {doc_ids[i]: float(s) for i, s in zip(row_indices, row_scores) if s > 0}
        for query_id, row_indices, row_scores in zip(query_ids, indices, scores)
    }


def dense(corpus, queries):
    doc_ids, query_ids = list(corpus), list(queries)
    model = SentenceTransformer(DENSE_MODEL)
    doc_embeddings = model.encode_document(
        list(corpus.values()), normalize_embeddings=True, batch_size=64, show_progress_bar=True
    )
    query_embeddings = model.encode_query(
        [queries[q] for q in query_ids], prompt=BGE_QUERY_PROMPT, normalize_embeddings=True
    )
    similarities = query_embeddings @ doc_embeddings.T
    top = np.argsort(-similarities, axis=1)[:, :DEPTH]
    return {
        query_id: {doc_ids[j]: float(similarities[i, j]) for j in top[i]}
        for i, query_id in enumerate(query_ids)
    }
