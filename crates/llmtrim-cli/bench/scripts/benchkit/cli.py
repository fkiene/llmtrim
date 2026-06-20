#!/usr/bin/env python3
"""benchkit CLI: `bench.py <competitor> [flags]`.

Looks the competitor up in the registry and runs the generic engine (deterministic sweep +
optional live CPCA leg + report). A competitor whose comparison shape does not fit the
corpora x grid engine (e.g. caveman, which compares system-prompt strategies on output
tokens) owns a self-contained `run(argv)` that the CLI dispatches to instead.
"""
import argparse
import json
import os
import sys
import time

from . import competitors, lib
from .config import (CORPORA, LLMTRIM_PRESETS, MATCHED, RESULTS_DIR)
from .corpora import load_corpus
from .gate import provenance, run_check, write_baseline
from .live import live_leg
from .pricing import load_pricing
from .report import render
from .sweep import deterministic_sweep

# Competitors that do not fit the corpora x grid engine and own their own run(argv).
SELF_CONTAINED = {"caveman", "rtk", "snip"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    ap = argparse.ArgumentParser(prog="bench.py",
                                 description="llmtrim vs <competitor> (CPCA + Pareto)")
    ap.add_argument("competitor", help="which competitor to benchmark against "
                    f"(known: {', '.join(sorted(competitors.REGISTRY))})")
    ap.add_argument("--limit", type=int, default=40, help="cases per corpus (token axis)")
    ap.add_argument("--repeats", type=int, default=1, help="compress repeats (median); token"
                    " counts are deterministic so 1 is enough")
    ap.add_argument("--live", action="store_true", help="run the budget-capped CPCA leg")
    ap.add_argument("--live-n", type=int, default=20, help="candidate cases per matched point")
    ap.add_argument("--seeds", type=int, default=3, help="generation seeds per case (live leg)")
    ap.add_argument("--max-out", type=int, default=1024, help="max_tokens per live generation")
    ap.add_argument("--budget", type=float, default=0.90, help="hard USD cap for the live leg")
    ap.add_argument("--check", action="store_true",
                    help="CI gate: assert invariants + reduction within baseline; exit 0/1")
    ap.add_argument("--write-baseline", action="store_true",
                    help="(re)generate baseline.json from this run's deterministic numbers")
    ap.add_argument("--no-ml", action="store_true",
                    help="disable the competitor's ML (deterministic routers only) for this run")

    # Dispatch self-contained competitors before the generic engine parses the rest. They get
    # the leftover argv so they keep their own flags (e.g. caveman's --summarize).
    name = argv[0] if argv and not argv[0].startswith("-") else None
    if name in SELF_CONTAINED:
        from .competitors import caveman, rtk, snip
        dispatch = {"caveman": caveman.run, "rtk": rtk.run, "snip": snip.run}
        return dispatch[name](argv[1:])

    args = ap.parse_args(argv)
    competitor = competitors.get(args.competitor)

    # --no-ml runs the competitor's deterministic routers alone. disable_ml() flips ML
    # availability, then we rebuild the competitor so its client sees the disabled state. The
    # first client above is discarded before any compress() call, so no ML pipeline was ever
    # built (it is lazy); a per-call toggle wouldn't work once the pipeline is constructed.
    noml = args.no_ml or bool(os.environ.get("LLMTRIM_HEADROOM_NOML"))
    if noml:
        competitor.disable_ml()
        competitor = competitors.get(args.competitor)

    enc = lib.get_encoder()
    pricing = load_pricing()

    cases = []
    for cname in CORPORA:
        cases += load_corpus(cname, args.limit)
    by_corpus = {}
    for _, _, meta in cases:
        by_corpus[meta["corpus"]] = by_corpus.get(meta["corpus"], 0) + 1
    print(f"corpus: {len(cases)} cases - " + ", ".join(f"{k}={v}" for k, v in by_corpus.items()),
          file=sys.stderr)

    installed = getattr(competitor, "installed", True)
    grid = competitor.config_grid()

    # Cold start = the FIRST call to each library in the process (FFI init for llmtrim, model
    # load for the competitor). A real per-deployment cost, measured once, before the warm-up
    # below excludes it from the per-call overhead.
    import time as _t
    warm = [{"role": "user", "content": "warm-up " + "lorem ipsum dolor " * 40}]
    cold = {}
    t0 = _t.perf_counter()
    lib.llmtrim_compress(warm, LLMTRIM_PRESETS[0], 1)
    cold["llmtrim"] = round((_t.perf_counter() - t0) * 1000, 1)
    if installed and grid:
        # Use the MAX config so the ML path fires and its model load is counted - the no-op
        # (first grid entry) would undercount the real cold start.
        t0 = _t.perf_counter()
        try:
            competitor.compress(warm, grid[-1][1], 1)
            cold[competitor.name] = round((_t.perf_counter() - t0) * 1000, 1)
        except Exception as e:  # noqa: BLE001
            print(f"{competitor.name} cold-start call failed: {e}", file=sys.stderr)

    # Warm both libraries (one-time setup excluded).
    for p in LLMTRIM_PRESETS:
        lib.llmtrim_compress(warm, p, 1)
    if installed:
        for _, cfg in grid:
            try:
                competitor.compress(warm, cfg, 1)
            except Exception as e:  # noqa: BLE001
                print(f"{competitor.name} warm-up failed: {e}", file=sys.stderr)

    print("=== deterministic sweep ($0) ===", file=sys.stderr)
    det = deterministic_sweep(enc, competitor, cases, pricing, args.repeats)
    for label in LLMTRIM_PRESETS + [g[0] for g in grid]:
        a = det.get(label)
        if a:
            print(f"  {label:12} {a['tool']:8} reduction {a['reduction_pct']:5.1f}%  "
                  f"(n={a['n']}, ml={a['ml_fired']})", file=sys.stderr)

    # $0 CI/baseline modes operate on the deterministic sweep alone and exit early.
    if args.write_baseline:
        write_baseline(det, args.limit)
        return 0
    if args.check:
        return run_check(det, args.limit)

    live = None
    if args.live:
        key = lib.load_api_key()
        if not key:
            print("ERROR: --live needs OPENROUTER_API_KEY (env or .env)", file=sys.stderr)
            return 1
        print(f"=== live CPCA leg ({args.seeds} seeds, budget ${args.budget}) ===",
              file=sys.stderr)
        live = live_leg(key, enc, competitor, cases, pricing, args.budget, args.live_n,
                        args.seeds, det, args.max_out)
    else:
        # The live leg costs real money. A deterministic-only re-run must NOT wipe a paid
        # live block - carry forward whatever the last run saved (re-run with --live to
        # refresh it).
        prev = RESULTS_DIR / "results.json"
        if prev.exists():
            old_live = json.loads(prev.read_text()).get("live")
            if old_live:
                print("carrying forward saved live results (re-run with --live to refresh)",
                      file=sys.stderr)
                live = old_live

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {"model": lib.MODEL, "encoder": "o200k_base", "corpora": CORPORA,
                 "llmtrim_presets": LLMTRIM_PRESETS,
                 "headroom_grid": [g[0] for g in grid],
                 "matched": MATCHED, "pricing": pricing,
                 "headroom_installed": installed,
                 "limit": args.limit, "repeats": args.repeats,
                 "cold_start_ms": cold,
                 "provenance": provenance(competitor)},
        "deterministic": det,
        "live": live,
    }
    if noml:
        # No-ML pass: write a sidecar the main report reads for the no-ML row. Don't touch
        # results.json / README (that's the ML-on headline).
        results["meta"]["headroom_noml"] = True
        (RESULTS_DIR / "results-noml.json").write_text(json.dumps(results, indent=2))
        print(f"\nWrote {RESULTS_DIR}/results-noml.json ({competitor.display} ML disabled)\n")
        return 0
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    report = render(results, competitor)
    (RESULTS_DIR / "README.md").write_text(report + "\n")
    # Archive every paid live run under a timestamp so it survives later regenerations.
    if args.live and live:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        archive = RESULTS_DIR / "live-runs"
        archive.mkdir(exist_ok=True)
        (archive / f"live-{stamp}.json").write_text(json.dumps(
            {"meta": results["meta"], "live": live}, indent=2))
        print(f"archived paid live run to {archive}/live-{stamp}.json", file=sys.stderr)
    print(f"\nWrote {RESULTS_DIR}/results.json and README.md\n")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
