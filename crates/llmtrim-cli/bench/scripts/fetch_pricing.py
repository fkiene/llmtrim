#!/usr/bin/env python3
"""Refresh the pinned pricing snapshot (bench/pricing.json) from https://models.dev/api.json.

Pulls the registry, keeps the providers the bench actually prices against (OpenRouter
ids are slashed, native provider ids are bare), and flattens each model's cost block
to {cache_read, input, output} in USD per 1M tokens — the shape `bench::load_pricing`
parses. Run: `python3 bench/scripts/fetch_pricing.py`.
"""

import datetime
import json
import os
import urllib.request

API = "https://models.dev/api.json"
# Native providers give bare ids (claude-*, gpt-*, gemini-*…); openrouter gives the
# slashed ids the live A/B bench sends. Same id never appears in two providers today;
# if that changes, later providers in this sorted order win.
PROVIDERS = ["anthropic", "deepseek", "google", "mistral", "openai", "openrouter"]
OUT = os.path.join(os.path.dirname(__file__), "..", "pricing.json")


def main() -> None:
    # models.dev 403s the default Python-urllib agent.
    req = urllib.request.Request(API, headers={"User-Agent": "llmtrim-bench/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        registry = json.load(r)

    models = {}
    for provider in PROVIDERS:
        for model_id, model in registry[provider].get("models", {}).items():
            cost = model.get("cost") or {}
            models[model_id] = {
                "cache_read": cost.get("cache_read", 0.0),
                "input": cost.get("input", 0),
                "output": cost.get("output", 0),
            }

    try:
        previous = json.load(open(OUT)).get("models", {})
    except (OSError, json.JSONDecodeError):
        previous = {}

    snapshot = {
        "fetched": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "models": dict(sorted(models.items())),
    }
    with open(OUT, "w") as f:
        f.write(json.dumps(snapshot, indent=0))

    added = sorted(set(models) - set(previous))
    removed = sorted(set(previous) - set(models))
    changed = sorted(
        m for m in set(models) & set(previous) if models[m] != previous[m]
    )
    print(f"{len(models)} models -> {os.path.relpath(OUT)}")
    print(f"added {len(added)}, removed {len(removed)}, repriced {len(changed)}")
    for m in changed:
        o, n = previous[m], models[m]
        print(f"  {m}: in {o['input']}->{n['input']} out {o['output']}->{n['output']} cache {o['cache_read']}->{n['cache_read']}")


if __name__ == "__main__":
    main()
