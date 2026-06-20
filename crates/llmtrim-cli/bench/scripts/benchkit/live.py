"""Live leg (budget-capped): accuracy, output tokens, CPCA. Generic over a Competitor.

The competitor arm is keyed by competitor.name (e.g. "headroom"), matched per run to an
llmtrim preset by achieved reduction. Compression is deterministic, so each arm's messages
are built once and reused across seeds.
"""
import json
import random
import sys
import time

from . import lib
from .config import DETERMINISTIC_SCORERS, MATCHED
from .corpora import score_v2
from .pricing import usd
from .stats import paired_bootstrap_diff, percentile


class Budget:
    def __init__(self, cap_usd, pricing):
        self.cap = cap_usd
        self.pricing = pricing
        self.spent = 0.0
        self.stopped = False

    def can_afford(self, in_tokens, max_out):
        # worst-case projection of the next call
        proj = usd(in_tokens, self.pricing["input"]) + usd(max_out, self.pricing["output"])
        return (self.spent + proj) <= self.cap

    def charge(self, in_tokens, out_tokens):
        self.spent += usd(in_tokens, self.pricing["input"]) + usd(out_tokens, self.pricing["output"])


def call_seed(key, messages, seed, max_out):
    """lib.call_model with an explicit seed + max_tokens so we can run multiple seeds and
    bound runaway generations. Returns (text, completion_tokens, finish_reason) or None."""
    import urllib.error
    import urllib.request
    payload = json.dumps({
        "model": lib.MODEL, "messages": messages, "temperature": 0, "seed": seed,
        "max_tokens": max_out, "provider": lib.PROVIDER_ROUTE,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/fkiene/llmtrim",
                 "X-Title": "llmtrim-vs-headroom"}, method="POST")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90, context=lib._SSL_CTX) as r:
                resp = json.loads(r.read())
            ch = (resp.get("choices") or [{}])[0]
            text = (ch.get("message") or {}).get("content") or ""
            ct = (resp.get("usage") or {}).get("completion_tokens") or 0
            return text, ct, ch.get("finish_reason")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 * (attempt + 1)); continue
            return None
        except Exception:  # noqa: BLE001
            if attempt < 2:
                time.sleep(2 * (attempt + 1)); continue
            return None
    return None


def closest_hr(competitor, det, target_pct):
    """The competitor config whose achieved reduction is nearest target_pct. The no-op
    `hr-default` (the first grid entry) is never the matched arm (pairing against a no-op was
    the original bias); on ties the MORE-aggressive config wins (the grid is ascending), so a
    deterministic, non-arbitrary choice - the ML often caps several configs at the same
    reduction."""
    grid = competitor.config_grid()
    noop = grid[0][0] if grid else None
    best, bd = None, 1e9
    for label, _ in grid:
        if label == noop:
            continue
        a = det.get(label)
        if not a:
            continue
        d = abs(a["reduction_pct"] - target_pct)
        if d <= bd:  # <= so a later (more aggressive) config wins ties
            best, bd = label, d
    if best is None and noop and det.get(noop):  # only the no-op was available
        best = noop
    return best


def live_leg(key, enc, competitor, cases, pricing, budget_cap, live_n, seeds, det, max_out=1024):
    """At each iso-compression point, generate original/llmtrim/competitor answers across
    `seeds` seeds, score continuously, and accumulate cost + score-sum → CPCA. The competitor
    arm is matched per run by achieved reduction. Compression is deterministic, so each arm's
    messages are built once and reused across seeds. The budget guard stops cleanly before
    any call would exceed the cap (writing partial results)."""
    installed = getattr(competitor, "installed", True)
    cname_arm = competitor.name
    grid = dict(competitor.config_grid())
    budget = Budget(budget_cap, pricing)
    scored = [(n, m, meta) for (n, m, meta) in cases
              if meta["scorer"] in DETERMINISTIC_SCORERS]
    random.Random(0).shuffle(scored)  # fixed, recorded selection

    points = {}
    for pname, preset in MATCHED.items():
        hr_label = (closest_hr(competitor, det, det.get(preset, {}).get("reduction_pct", 0))
                    if installed else None)
        hr_cfg = grid[hr_label] if hr_label else None
        match_info = {"preset": preset, "hr_label": hr_label,
                      "llmtrim_reduction_pct": det.get(preset, {}).get("reduction_pct"),
                      "headroom_reduction_pct": det.get(hr_label, {}).get("reduction_pct")
                      if hr_label else None}
        sample = scored[:live_n]
        # build the (deterministic) compressed arms once per case
        prepared = []
        for cname, messages, meta in sample:
            arms = {"original": messages}
            arms["llmtrim"] = lib.llmtrim_compress(messages, preset, 1)[0]
            if installed:
                arms[cname_arm] = competitor.compress(messages, hr_cfg, 1)[0]
            prepared.append((cname, meta, arms))

        samples = []  # one row per (case, seed, arm-set)
        for s in range(seeds):
            for cname, meta, arms in prepared:
                if budget.stopped:
                    break
                gold, scorer = meta["gold"], meta["scorer"]
                row = {"name": cname, "corpus": meta["corpus"], "scorer": scorer, "seed": s,
                       "q": {}, "in_tokens": {}, "out_tokens": {}, "cost": {}, "truncated": {}}
                for arm, amsgs in arms.items():
                    in_tok = lib.count(enc, amsgs)
                    if not budget.can_afford(in_tok, max_out):
                        print(f"  budget guard: stop at ${budget.spent:.4f}/{budget.cap} "
                              f"before {pname}/s{s}/{cname}/{arm}", file=sys.stderr)
                        budget.stopped = True
                        break
                    res = call_seed(key, amsgs, s, max_out)
                    if res is None:
                        row["q"][arm] = None
                        continue
                    text, ct, finish = res
                    budget.charge(in_tok, ct)
                    row["in_tokens"][arm] = in_tok
                    row["out_tokens"][arm] = ct
                    row["cost"][arm] = usd(in_tok, pricing["input"]) + usd(ct, pricing["output"])
                    row["q"][arm] = score_v2(scorer, text, gold)
                    row["truncated"][arm] = (finish == "length")
                    time.sleep(0.3)
                if row["q"]:
                    samples.append(row)
                    print(f"  [{pname} s{s}] {cname:18} "
                          + " ".join(f"{a}={row['q'].get(a):.2f}" if row['q'].get(a) is not None
                                     else f"{a}=na" for a in arms)
                          + f"  spent=${budget.spent:.4f}", file=sys.stderr, flush=True)
            if budget.stopped:
                break
        points[pname] = summarize_live(samples, cname_arm)
        points[pname]["match"] = match_info
        a = points[pname]["arms"]
        print(f"  [{pname}] n={points[pname]['n']} "
              + " ".join(f"{k}:q={a[k]['quality']:.2f}" for k in a if a[k]['quality'] is not None)
              + f"  spent=${budget.spent:.4f}", file=sys.stderr)
    points["_budget_spent_usd"] = round(budget.spent, 4)
    points["_budget_cap_usd"] = budget_cap
    points["_seeds"] = seeds
    points["_max_out"] = max_out
    return points


def summarize_live(samples, competitor_arm):
    """Per-arm quality (mean score), output tokens (median + total), and CPCA computed as
    total_cost / sum(score) - fractional credit, so one flipped answer no longer swings an
    integer denominator (P0-4). Significance via paired bootstrap on score differences."""
    arms = ("original", "llmtrim", competitor_arm)
    out = {"n": len(samples), "per_sample": samples, "arms": {}}
    for a in arms:
        qs = [r["q"].get(a) for r in samples if r["q"].get(a) is not None]
        if not qs:
            out["arms"][a] = {"n": 0, "quality": None, "cpca_usd": None}
            continue
        cost = sum(r["cost"].get(a, 0.0) for r in samples)
        sum_score = sum(qs)
        out["arms"][a] = {
            "n": len(qs), "quality": sum(qs) / len(qs), "score_sum": sum_score,
            "input_tokens": sum(r["in_tokens"].get(a, 0) for r in samples),
            "output_tokens_total": sum(r["out_tokens"].get(a, 0) for r in samples),
            "output_tokens_median": percentile([r["out_tokens"].get(a) for r in samples], 50),
            "truncated": sum(1 for r in samples if r["truncated"].get(a)),
            "total_cost_usd": cost,
            "cpca_usd": (cost / sum_score) if sum_score > 0 else None,
        }
    diffs = [r["q"]["llmtrim"] - r["q"][competitor_arm] for r in samples
             if r["q"].get("llmtrim") is not None and r["q"].get(competitor_arm) is not None]
    mean, lo, hi = paired_bootstrap_diff(diffs)
    if mean is not None:
        out["quality_diff"] = {"llmtrim_minus_headroom_mean": mean, "ci95": [lo, hi],
                               "significant": (lo > 0 or hi < 0), "n_paired": len(diffs)}
    return out
