"""Jev through OpenRouter's TypeSafe-compatible endpoint, with an append-only disk cache."""

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from ladder import ROOT

ENDPOINT = "https://openrouter.ai/api/v1/systemone"
MODEL = "typesafe/jev-1.13-20260917"
# "repeat" holds second calls of already-cached requests, for the determinism check.
CACHE_PATHS = {
    "main": ROOT / "cache" / "jev" / "responses.jsonl",
    "repeat": ROOT / "cache" / "jev" / "repeats.jsonl",
}
SPEND_CAP_USD = 4.0
PRICE_PER_INPUT_TOKEN = 0.042e-6
RETRY_STATUSES = {429, 500, 502, 503, 529}
MAX_ATTEMPTS = 6

# A frames SciFact queries as claims to check; B is GPT Researcher's generic QA rubric.
RUBRIC_A_CRITERIA = [
    "Unrelated to the claim",
    "Same topic, but says nothing about whether the claim is true",
    "Indirect or partial evidence about the claim",
    "Directly provides evidence that supports or refutes the claim",
]
RUBRIC_B_CRITERIA = [
    "Unrelated to the question",
    "Same topic, but does not help answer the question",
    "Partially answers the question or gives useful supporting facts",
    "Directly answers the question with specific facts",
]


def build_request(model, rubric, claim, passage):
    if rubric == "A":
        state = {"claim": claim, "passage": passage}
        question = {
            "type": "score",
            "instructions": "How useful is this passage as scientific evidence for checking the claim?",
            "criteria": RUBRIC_A_CRITERIA,
        }
    elif rubric == "B":
        state = passage
        question = {
            "type": "score",
            "instructions": f"How useful is this passage for answering the question: {claim}",
            "criteria": RUBRIC_B_CRITERIA,
        }
    else:
        raise ValueError(f"unknown rubric {rubric!r}")
    return {"model": model, "state": state, "questions": {"relevance": question}}


def _cache_key(request):
    blob = json.dumps({"endpoint": ENDPOINT, "request": request}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _read_cache(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return {entry["key"]: entry for entry in map(json.loads, f)}


_caches = {name: _read_cache(path) for name, path in CACHE_PATHS.items()}
_spent_usd = sum(e["response"]["usage"]["cost"] for cache in _caches.values() for e in cache.values())
_lock = threading.Lock()


def spent_usd():
    return _spent_usd


def cache_entries(cache):
    return list(_caches[cache].values())


def estimate_usd(requests, cache="main"):
    """(uncached request count, rough cost). 2.85 JSON characters per token was measured on
    the M3 dev run; scientific text tokenizes denser than the usual ~4."""
    missing = [r for r in requests if _cache_key(r) not in _caches[cache]]
    return len(missing), sum(len(json.dumps(r)) for r in missing) / 2.85 * PRICE_PER_INPUT_TOKEN


def _post(client, request):
    """Returns (response JSON, latency of the successful attempt in ms)."""
    for attempt in range(MAX_ATTEMPTS):
        start = time.perf_counter()
        response = client.post(ENDPOINT, json=request)
        latency_ms = (time.perf_counter() - start) * 1000
        if response.status_code not in RETRY_STATUSES:
            response.raise_for_status()
            return response.json(), latency_ms
        time.sleep(float(response.headers.get("retry-after", 2**attempt)))
    response.raise_for_status()


def _fetch(client, request, cache):
    global _spent_usd
    if _spent_usd >= SPEND_CAP_USD:
        raise RuntimeError(f"Jev spend cap of ${SPEND_CAP_USD} reached")
    response, latency_ms = _post(client, request)
    if response["model"] != request["model"]:
        raise RuntimeError(f"requested {request['model']} but got {response['model']}")

    entry = {
        "key": _cache_key(request),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round(latency_ms, 1),
        "request": request,
        "response": response,
    }
    with _lock:
        with open(CACHE_PATHS[cache], "a") as f:
            f.write(json.dumps(entry) + "\n")
        _caches[cache][entry["key"]] = entry
        _spent_usd += response["usage"]["cost"]


def call_many(requests, cache="main", workers=4):
    """Cache entries for `requests`, in order; cache misses are fetched concurrently."""
    entries = _caches[cache]
    missing = list({_cache_key(r): r for r in requests if _cache_key(r) not in entries}.values())
    if missing:
        load_dotenv(ROOT / ".env")
        headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
        CACHE_PATHS[cache].parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(headers=headers, timeout=60) as client, ThreadPoolExecutor(workers) as pool:
            list(tqdm(pool.map(lambda r: _fetch(client, r, cache), missing), total=len(missing)))
    return [entries[_cache_key(r)] for r in requests]


def call(request):
    return call_many([request])[0]
