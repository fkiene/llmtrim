"""Pricing load + USD helper, from the pinned bench/pricing.json snapshot."""
import json

from . import lib
from .config import PRICING


def load_pricing():
    d = json.loads(PRICING.read_text())
    p = d["models"].get(lib.MODEL)
    if not p:
        raise SystemExit(f"no pricing for {lib.MODEL} in {PRICING}")
    return {"input": p["input"], "output": p["output"],
            "cache_read": p.get("cache_read", 0.0), "fetched": d.get("fetched")}


def usd(tokens, rate_per_million):
    return tokens / 1_000_000.0 * rate_per_million
