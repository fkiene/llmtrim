#!/usr/bin/env python3
"""Refresh the embedded reasoning-capability snapshot.

The anti-overthinking directive (see `llmtrim-core/src/capability.rs`) only injects for models
that actually run a reasoning pass. Claude Code and other harnesses send a bare model id with no
`thinking`/`reasoning` field on the wire, so the signal has to come from the model registry, not
the request. The signal is a static snapshot of the per-model `reasoning` flag from models.dev,
embedded in the core crate. Run this on release to refresh it, the same way `bench/pricing.json`
and the LMArena board are refreshed.

Source: https://models.dev/api.json (same registry the bench prices from). Keeps the providers the
proxy actually sees and writes `crates/llmtrim-core/data/model_reasoning.json` as
`{fetched, models:{id: bool}}`.

Usage:  python3 tools/refresh_reasoning.py
Deps:   none beyond the standard library.
"""

import datetime
import json
import os
import urllib.request

API = "https://models.dev/api.json"
# Native providers give bare ids (claude-*, gpt-*…); openrouter gives the slashed ids the live
# bench sends. Mirror `bench/scripts/benchkit/data/fetch_pricing.py` so the two snapshots cover
# the same id space.
PROVIDERS = ["anthropic", "deepseek", "google", "mistral", "openai", "openrouter"]
OUT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "crates",
    "llmtrim-core",
    "data",
    "model_reasoning.json",
)


def main() -> None:
    # models.dev 403s the default Python-urllib agent.
    req = urllib.request.Request(API, headers={"User-Agent": "llmtrim-bench/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        registry = json.load(r)

    models = {}
    for provider in PROVIDERS:
        for model_id, model in registry[provider].get("models", {}).items():
            models[model_id] = bool(model.get("reasoning"))

    snapshot = {
        "fetched": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "models": dict(sorted(models.items())),
    }
    with open(OUT, "w") as f:
        f.write(json.dumps(snapshot, indent=0))

    reasoning = sum(models.values())
    print(f"{len(models)} models ({reasoning} reasoning) -> {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
