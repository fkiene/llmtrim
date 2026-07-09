#!/usr/bin/env python3
"""Refresh the embedded per-model context-window snapshot.

The breakdown occupancy view (see `llmtrim-cli`'s `window_for`) shows how full a request's context
is, which needs each model's real context window. The window is model-specific and not on the wire,
so it comes from the model registry, embedded as a static snapshot in the core crate. Run this on
release to refresh it, the same way `bench/pricing.json`, the LMArena board, and the reasoning flags
are refreshed.

Source: https://models.dev/api.json (same registry the bench prices from). Keeps the providers the
proxy actually sees and writes `crates/llmtrim-core/data/model_context.json` as
`{fetched, models:{id: context_tokens}}`. Uses `limit.context` (the advertised context window).

Usage:  python3 tools/refresh_context.py
Deps:   none beyond the standard library.
"""

import datetime
import json
import os
import urllib.request

API = "https://models.dev/api.json"
# Native providers give bare ids (claude-*, gpt-*…); openrouter gives the slashed ids the live
# bench sends. Mirror `refresh_reasoning.py` so the snapshots cover the same id space.
PROVIDERS = ["anthropic", "deepseek", "google", "mistral", "openai", "openrouter"]
OUT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "crates",
    "llmtrim-core",
    "data",
    "model_context.json",
)


def main() -> None:
    # models.dev 403s the default Python-urllib agent.
    req = urllib.request.Request(API, headers={"User-Agent": "llmtrim-bench/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        registry = json.load(r)

    models = {}
    for provider in PROVIDERS:
        for model_id, model in registry[provider].get("models", {}).items():
            context = (model.get("limit") or {}).get("context")
            if isinstance(context, int) and context > 0:
                models[model_id] = context

    snapshot = {
        "fetched": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "models": dict(sorted(models.items())),
    }
    with open(OUT, "w") as f:
        f.write(json.dumps(snapshot, indent=0))

    print(f"{len(models)} models -> {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
