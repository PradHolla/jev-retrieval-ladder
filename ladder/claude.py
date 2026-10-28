"""Claude through the Anthropic SDK, with an append-only disk cache and a spend cap."""

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from ladder import ROOT

# Haiku 5.5 has no dated snapshot; every cached response records the model that answered.
MODEL = "claude-haiku-5-5"
CACHE_PATH = ROOT / "cache" / "claude" / "responses.jsonl"
SPEND_CAP_USD = 2.0
# Haiku 5.5's rate card for prompts of 100K tokens or fewer; ours are a few thousand.
PRICE_PER_INPUT_TOKEN = 0.10e-6
PRICE_PER_OUTPUT_TOKEN = 0.50e-6
MAX_RETRIES = 6


def cost_usd(usage):
    return usage["input_tokens"] * PRICE_PER_INPUT_TOKEN + usage["output_tokens"] * PRICE_PER_OUTPUT_TOKEN


def _cache_key(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def _read_cache():
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH) as f:
        return {entry["key"]: entry for entry in map(json.loads, f)}


_cache = _read_cache()
_spent_usd = sum(cost_usd(e["response"]["usage"]) for e in _cache.values())
_lock = threading.Lock()


def spent_usd():
    return _spent_usd


def estimate_usd(requests):
    """(uncached request count, rough cost), scaled from the cached calls: input tokens per
    prompt character, and mean output tokens (thinking included) per call."""
    missing = [r for r in requests if _cache_key(r) not in _cache]
    entries = list(_cache.values())
    chars = sum(len(json.dumps(e["request"])) for e in entries)
    tokens_per_char = sum(e["response"]["usage"]["input_tokens"] for e in entries) / chars
    output_tokens = sum(e["response"]["usage"]["output_tokens"] for e in entries) / len(entries)
    input_tokens = sum(len(json.dumps(r)) for r in missing) * tokens_per_char
    return len(missing), (input_tokens * PRICE_PER_INPUT_TOKEN
                          + len(missing) * output_tokens * PRICE_PER_OUTPUT_TOKEN)


def _fetch(client, request):
    global _spent_usd
    if _spent_usd >= SPEND_CAP_USD:
        raise RuntimeError(f"Claude spend cap of ${SPEND_CAP_USD} reached")
    start = time.perf_counter()
    response = client.messages.create(**request).model_dump(mode="json")
    latency_ms = (time.perf_counter() - start) * 1000
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
        with open(CACHE_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        _cache[entry["key"]] = entry
        _spent_usd += cost_usd(response["usage"])


def call_many(requests, workers=8):
    """Cache entries for `requests`, in order; cache misses are fetched concurrently."""
    missing = list({_cache_key(r): r for r in requests if _cache_key(r) not in _cache}.values())
    if missing:
        load_dotenv(ROOT / ".env")
        client = anthropic.Anthropic(max_retries=MAX_RETRIES)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(workers) as pool:
            list(tqdm(pool.map(lambda r: _fetch(client, r), missing), total=len(missing)))
    return [_cache[_cache_key(r)] for r in requests]
