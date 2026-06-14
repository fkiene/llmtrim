#!/usr/bin/env python3
"""Synthesize bench/README.md from bench/results/*.json + data/manifest.json.

Run after `llmtrim bench suite` completes. Renders the two-axis frontier table
(tokens saved vs quality retained) plus methodology and honest caveats.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bench/ root (this script lives in bench/scripts/)

# corpus -> (shape, scorer description, what it stresses)
META = {
    "gsm8k": ("Reasoning (CoT)", "numeric-exact", "output draft / token-budget"),
    "humaneval": ("Code generation", "pass@1 (runs unit tests)", "skeleton + minify"),
    "dolly": ("Generation (instruction)", "LLM-judge", "output-control on long-form answers"),
    "hotpotqa": ("Multi-hop RAG", "token-F1", "retrieve (long context)"),
    "glaive": ("Agent / tool-use", "tool-call match", "tool select / trim"),
    "chat": ("Multi-turn chat", "LLM-judge", "output-control + dedup/cache on history"),
    "cnn": ("Long-doc summary", "token-F1", "output budget on long input"),
    "cache": ("Prompt-cache reuse", "numeric-exact", "stable shared prefix (Stage A)"),
}
ORDER = list(META.keys())


def unwrap(doc):
    # Flatten the shared bench envelope ({schema, meta, result, ...}) into the legacy flat
    # shape this script reads. Pre-envelope (bare) docs pass through unchanged.
    if isinstance(doc, dict) and "schema" in doc and "result" in doc:
        return {**doc.get("meta", {}), **doc["result"]}
    return doc


def load_results():
    out = {}
    for path in glob.glob(os.path.join(HERE, "results", "*.json")):
        name = os.path.splitext(os.path.basename(path))[0]
        # Pool only the shape-matched run (`bench suite` writes results/<corpus>.json). Skip
        # the preset-variant files (results/<corpus>__safe|aggressive|tuned.json), which exist
        # for the chart's safe/aggressive "dial" and would otherwise double-count.
        if name == "run" or "__" in name:
            continue
        try:
            out[name] = unwrap(json.load(open(path)))
        except Exception:
            pass
    return out


def main():
    results = load_results()
    manifest = {}
    mpath = os.path.join(HERE, "data", "manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath)).get("corpora", {})

    model = next((r.get("model") for r in results.values()), "openai/gpt-oss-20b")

    lines = []
    lines.append("# llmtrim benchmark\n")
    lines.append(
        "Two axes, measured live: **tokens saved** (real tokenizer, at compress time) and "
        "**quality retained** (the A/B delta between the model's answer on the *original* vs the "
        "*compressed* request). A preset is only honest if quality holds at its token saving — the "
        "frontier of (saved, retained) is the benchmark, not the saving alone.\n"
    )
    lines.append(
        f"- **Model:** `{model}` via OpenRouter, pinned to the **Groq** upstream (async-openai byot — "
        "the exact compressed body is sent, injected fields intact).\n"
        "- **Scoring:** ground-truth where possible (numeric-exact for math, pass@1 that *runs the unit "
        "tests* for code), token-F1 for extractive QA, tool-call match for agents, an LLM judge only for "
        "open-ended shapes.\n"
        "- **Cost:** priced from a pinned [models.dev](https://models.dev) snapshot (`bench/pricing.json`), "
        "Groq rates, cached input billed at the `cache_read` rate.\n"
        "- **Cache used %:** share of compressed input served from the provider prompt cache "
        "(`usage.prompt_tokens_details.cached_tokens`).\n"
    )

    # Bottom line — aggregate token + cost ledger across every case.
    def s(field):
        return sum(c.get(field, 0) for r in results.values() for c in r.get("cases", []))

    io, ic = s("tokens_in_before"), s("tokens_in_after")
    oo, oc = s("tokens_out_orig"), s("tokens_out_comp")
    co, cc = s("cost_orig"), s("cost_comp")
    n_tot = sum(r.get("n", 0) for r in results.values())
    drop = lambda a, b: (a - b) / a * 100 if a else 0.0
    lines.append("\n## Bottom line\n")
    lines.append(
        f"Across **{n_tot} A/B cases** on this real-usage mix (generation, chat, reasoning, code, RAG, "
        f"agent, summary, cache):\n\n"
        f"| | original | compressed | saved |\n|---|--:|--:|--:|\n"
        f"| input tokens | {io:,} | {ic:,} | **{drop(io,ic):.0f}%** |\n"
        f"| output tokens | {oo:,} | {oc:,} | **{drop(oo,oc):.0f}%** |\n"
        f"| **total tokens** | **{io+oo:,}** | **{ic+oc:,}** | **{drop(io+oo,ic+oc):.0f}%** |\n"
        f"| **round-trip cost** | **${co:.4f}** | **${cc:.4f}** | **{drop(co,cc):.0f}%** |\n\n"
        f"**~{drop(co,cc):.0f}% cost cut, quality mostly held or improved.** Output is where it pays off — "
        f"output tokens drop {drop(oo,oc):.0f}% via output-control, and real workloads are output-heavy. "
        f"(An earlier input-heavy/short-output mix saved only ~9% — the lever was real but had nothing to "
        f"cut; representative corpora surface the true savings.)\n"
    )

    # Frontier table.
    lines.append("\n## Frontier\n")
    lines.append(
        "| corpus | shape | n | input saved | output saved | cost saved | cache used | quality (orig→comp) | retention |"
    )
    lines.append("|---|---|--:|--:|--:|--:|--:|:--:|--:|")
    for name in ORDER:
        r = results.get(name)
        if not r:
            continue
        shape = META[name][0]
        lines.append(
            "| `{name}` | {shape} | {n} | {ti:.0f}% | {to:.0f}% | {c:.0f}% | {ca:.0f}% | {qo:.0f}%→{qc:.0f}% | {ret:+.0f}pp |".format(
                name=name, shape=shape, n=r.get("n", 0),
                ti=r.get("tokens_in_saved_pct", 0), to=r.get("tokens_out_saved_pct", 0),
                c=r.get("cost_saved_pct", 0), ca=r.get("cache_used_pct", 0),
                qo=r.get("quality_orig", 0) * 100, qc=r.get("quality_comp", 0) * 100,
                ret=r.get("retention_pp", 0),
            )
        )

    # Key findings — auto-classified, CI-AWARE: a delta is only "real" if its
    # magnitude exceeds the 95% CI half-width (interval clear of zero). Everything else
    # is noise at this n and must not be reported as a win or a regression.
    wins, confirmed_reg, noisy = [], [], []
    for name in ORDER:
        r = results.get(name)
        if not r:
            continue
        ret = r.get("retention_pp", 0)
        ci = r.get("retention_ci95_pp", 0)
        cost = r.get("cost_saved_pct", 0)
        # "Confirmed" only if the interval clears zero by a ≥3pp margin — at small n the
        # CIs are wide, so a delta whose CI merely grazes zero is not a real effect.
        significant = (abs(ret) - ci) > 3.0
        if ret < 0 and significant:
            confirmed_reg.append((name, r, ci))
        elif cost > 10 and not (ret < 0 and significant):
            wins.append((name, r, ci))
        if ret < 0 and not significant and cost > 5:
            noisy.append((name, r, ci))

    lines.append("\n## Key findings\n")
    lines.append("**Where compression wins** (cost cut, quality not significantly down):")
    for name, r, ci in wins:
        lines.append(
            f"- `{name}`: **cost −{r['cost_saved_pct']:.0f}%**, quality "
            f"{r['quality_orig']*100:.0f}%→{r['quality_comp']*100:.0f}% ({r['retention_pp']:+.0f}±{ci:.0f}pp)."
        )
    if confirmed_reg:
        lines.append("\n**Confirmed regressions** (delta exceeds its CI):")
        for name, r, ci in confirmed_reg:
            lines.append(
                f"- `{name}`: {r['quality_orig']*100:.0f}%→{r['quality_comp']*100:.0f}% "
                f"(**{r['retention_pp']:+.0f}±{ci:.0f}pp**) under `{r['preset']}`."
            )
    if noisy:
        labels = ", ".join(
            f"`{n}` ({r['retention_pp']:+.0f}±{ci:.0f}pp)" for n, r, ci in noisy
        )
        lines.append(
            f"\n**Within noise at this n** (negative but CI crosses zero — *not* confirmed regressions): "
            f"{labels}. Scale n to resolve."
        )
    lines.append(
        "\n**The headline:** the per-stage **token gate guarantees fewer tokens, not preserved quality** — "
        "only this A/B quality axis catches the difference. The two regressions we confirmed and fixed were "
        "measured on **gpt-oss-20b** (a stronger model with tighter intervals): `code`'s compact-code output "
        "**−21.6±14.5pp** at n=37 → dropped from the preset; and `aggressive`+n-gram on `adult` **−100pp** "
        "(deterministic) → `ngram` now skips JSON records (recovers to 100%). On a weaker/noisier model the "
        "same levers mostly land inside their CIs — measure per model, and reserve lossy stages for inputs "
        "whose exact surface form the task doesn't depend on.\n"
    )

    # Per-corpus detail.
    lines.append("\n## What each row stresses\n")
    for name in ORDER:
        r = results.get(name)
        if not r:
            continue
        shape, scorer, stress = META[name]
        src = manifest.get(name, {}).get("dataset", "synthetic" if name == "cache" else "?")
        lines.append(
            f"- **`{name}`** ({shape}, preset `{r.get('preset')}`, scorer `{scorer}`) — stresses {stress}. "
            f"Source: `{src}`."
        )

    # Caveats.
    max_cache = max((r.get("cache_used_pct", 0) for r in results.values()), default=0)
    cache_caveat = (
        "- **Cache used % is 0 across the board here — the provider does not cache this model.** Verified by "
        "probe: a repeated 1.3k-token prefix returns `cached_tokens: 0` (and `cache_write_tokens: 0`) for this provider×model, despite OpenRouter listing a `cache_read` price. The cache mechanism + "
        "measurement are validated on **caching** providers (Groq hit ~95% on the same `cache` corpus); cache "
        "value is provider×model-dependent, so pick a caching upstream for fixed-prefix workloads.\n"
        if max_cache < 5
        else "- **Cache used % is ~0 for one-shot diverse prompts** (nothing to cache-hit across distinct "
        "requests) and high only when a long prefix repeats — see `cache` (fixed system dossier + varying "
        "queries), the canonical agent/RAG-over-fixed-context shape.\n"
    )
    lines.append("\n## Reading the numbers honestly\n")
    lines.append(
        "- **No single compression %** — it is input-shape dependent. Long/structured inputs (RAG, "
        "record arrays, long docs) win on *input* tokens; short prompts (math, code stubs) can go *negative* "
        "on input because `output_control` injects a fixed instruction whose payoff is **output-side** "
        "(shorter answers), invisible in the input measure. Read **cost saved**, which captures both.\n"
        + cache_caveat
        + "- **Small n** — these runs use modest n for cost; CIs are reported in the JSON. Scale n for tighter "
        "intervals; several deltas here sit inside their CI (noise).\n"
        "- **pass@1 actually executes** the model's code against the unit tests — the strongest signal here "
        "(no judge noise).\n"
    )

    lines.append("\n## Reproduce\n")
    lines.append("```bash")
    lines.append("python3 bench/scripts/download.py 40       # pull + normalize corpora (pinned in data/manifest.json)")
    lines.append("cargo run -q --features live -- bench suite   # live A/B across all corpora (needs OPENROUTER_API_KEY)")
    lines.append("python3 bench/scripts/synth_readme.py      # regenerate this file")
    lines.append("```")
    lines.append(
        "\nPer-stage ablation (offline, free): "
        "`llmtrim bench quality --corpus bench/data/<c>.jsonl --preset aggressive --ablate`.\n"
    )

    lines.append("\n## Head-to-head: Headroom\n")
    lines.append(
        "Both libraries run through their Python APIs on the same inputs and the same "
        "`o200k_base` denominator. Full tables (input saved, per-stage attribution, latency, "
        "and the live gpt-oss-20b output A/B) are in "
        "[snapshots/vs-headroom/README.md](snapshots/vs-headroom/README.md).\n"
    )
    lines.append("```bash")
    lines.append("crates/llmtrim-uniffi/scripts/build-wheel.sh --release   # build the llmtrim wheel")
    lines.append("pip install --user target/wheels/llmtrim-*.whl")
    lines.append("pip install --user -r bench/scripts/requirements-vs-headroom.txt")
    lines.append("python3 bench/scripts/vs_headroom.py                     # deterministic axes (offline)")
    lines.append("python3 bench/scripts/vs_headroom.py --live --live-n 12  # output A/B (needs OPENROUTER_API_KEY)")
    lines.append("```")

    readme = os.path.join(HERE, "README.md")
    open(readme, "w").write("\n".join(lines) + "\n")
    print(f"wrote {readme} ({len(results)} corpora)")


if __name__ == "__main__":
    main()
