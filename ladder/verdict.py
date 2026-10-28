"""Claim verification by Claude over the passages each pipeline hands it."""

import json

import numpy as np

from ladder import claude, rank

VERDICTS = ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
CONTEXT_SIZE = 3
EFFORT = "medium"
MAX_TOKENS = 8000

SYSTEM = (
    "You check scientific claims against abstracts returned by a search engine. Judge only from "
    "the abstracts, not from your own knowledge. Answer SUPPORT if an abstract gives evidence that "
    "the claim is true, CONTRADICT if an abstract gives evidence that it is false, and "
    "NOT_ENOUGH_INFO if no abstract settles it, including when there are no abstracts."
)
SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": VERDICTS}},
    "required": ["verdict"],
    "additionalProperties": False,
}


def build_request(claim, passages):
    abstracts = "\n\n".join(f"[{i}] {passage}" for i, passage in enumerate(passages, 1)) or "(none)"
    return {
        "model": claude.MODEL,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        "system": SYSTEM,
        "messages": [{"role": "user", "content": f"Claim: {claim}\n\nAbstracts:\n\n{abstracts}"}],
    }


def parse(response):
    """The verdict, or None when Claude refused or ran out of tokens."""
    if response["stop_reason"] != "end_turn":
        return None
    text = next(block["text"] for block in response["content"] if block["type"] == "text")
    return json.loads(text)["verdict"]


def contexts(qrels, hybrid, ce, jev, jev_scores, jev_threshold):
    """pipeline -> query_id -> the doc_ids Claude reads, best first.

    "cited" is the abstracts the claim was written from, the context an oracle retriever
    would hand over; for NOT_ENOUGH_INFO claims they hold no rationale.
    """
    def top(run):
        return {q: rank.ranked_doc_ids(run[q])[:CONTEXT_SIZE] for q in qrels}

    return {
        "hybrid": top(hybrid),
        "ce": top(ce),
        "jev": top(jev),
        "jev_threshold": {q: [d for d in rank.ranked_doc_ids(jev[q])[:rank.RERANK_DEPTH]
                              if jev_scores[q][d] >= jev_threshold] for q in qrels},
        "cited": {q: list(qrels[q]) for q in qrels},
    }


def macro_f1(predicted, gold):
    f1s = []
    for label in VERDICTS:
        hits = np.sum((predicted == label) & (gold == label))
        precision = hits / max(np.sum(predicted == label), 1)
        recall = hits / np.sum(gold == label)
        f1s.append(2 * precision * recall / (precision + recall) if hits else 0.0)
    return float(np.mean(f1s))
