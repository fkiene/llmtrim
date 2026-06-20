"""Deterministic leg ($0): token reduction + input cost across the full sweep.

Generic over a Competitor: the llmtrim arms come from LLMTRIM_PRESETS, the competitor arms
from competitor.config_grid(), and the competitor's compress()/ml_fired() drive its rows.
"""
import statistics

from . import lib
from .config import LLMTRIM_PRESETS
from .pricing import usd
from .stats import bootstrap_weighted_reduction, percentile


def deterministic_sweep(enc, competitor, cases, pricing, repeats):
    """For every case, compress with each llmtrim preset and each competitor config. Returns
    per-arm aggregates (token reduction, input cost) - the Pareto x-axis, no API calls."""
    installed = getattr(competitor, "installed", True)
    arms = [("llmtrim", p, p) for p in LLMTRIM_PRESETS]
    arms += [(competitor.name, label, cfg) for label, cfg in competitor.config_grid()]

    # per-arm accumulators, and per-arm per-corpus
    agg = {}
    for tool, label, _ in arms:
        agg[label] = {"tool": tool, "before": 0, "after": 0, "per_corpus": {},
                      "pairs": [], "ms": [], "ml_fired": 0, "n": 0, "cases": []}

    for cname, messages, meta in cases:
        corpus = meta["corpus"]
        base = lib.count(enc, messages)
        for tool, label, cfg in arms:
            if tool == "llmtrim":
                out_msgs, _, ms = lib.llmtrim_compress(messages, cfg, repeats)
                ml = False
            else:
                if not installed:
                    continue
                out_msgs, transforms, ms = competitor.compress(messages, cfg, repeats)
                ml = competitor.ml_fired(transforms)
            after = lib.count(enc, out_msgs)
            a = agg[label]
            a["before"] += base
            a["after"] += after
            a["n"] += 1
            a["ml_fired"] += int(ml)
            a["pairs"].append((base, after))
            a["ms"].append(ms)
            # per-case raw row so a reviewer can recompute the headline (P2-6).
            a["cases"].append({"corpus": corpus, "name": cname,
                               "before": base, "after": after,
                               "saved_pct": round(lib.pct(base, after), 2)})
            pc = a["per_corpus"].setdefault(corpus, {"before": 0, "after": 0})
            pc["before"] += base
            pc["after"] += after

    # finalize: reduction %, input cost, CI
    out = {}
    for label, a in agg.items():
        if a["n"] == 0:
            continue
        red = lib.pct(a["before"], a["after"])
        lo, hi = bootstrap_weighted_reduction(a["pairs"])
        out[label] = {
            "tool": a["tool"], "n": a["n"],
            "tokens_before": a["before"], "tokens_after": a["after"],
            "reduction_pct": red, "reduction_ci": [lo, hi],
            "per_case": a["cases"],
            "input_cost_before_usd": usd(a["before"], pricing["input"]),
            "input_cost_after_usd": usd(a["after"], pricing["input"]),
            # Compress overhead: the wall-clock a Python caller waits for this library's
            # compress() - the Rust(llmtrim) vs Python+ML(competitor) latency gap.
            "overhead_ms_median": statistics.median(a["ms"]) if a["ms"] else None,
            "overhead_ms_p95": percentile(a["ms"], 95),
            "ml_fired": a["ml_fired"],
            "per_corpus": {c: {"reduction_pct": lib.pct(d["before"], d["after"]),
                               "tokens_before": d["before"], "tokens_after": d["after"]}
                           for c, d in a["per_corpus"].items()},
        }
    return out
