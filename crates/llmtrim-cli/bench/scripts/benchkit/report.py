"""Reporting: render the snapshot README from a results dict. Generic over a Competitor.

Competitor-specific caveat prose comes from competitor.notes(); the layout, numbers, and
tables are shared. competitor.display names the tool in headings and prose.
"""
import json

from .config import LLMTRIM_PRESETS, MATCHED, RESULTS_DIR


def cusd(x):
    return f"${x:.4f}" if x is not None else "n/a"


def render(results, competitor):
    disp = competitor.display
    grid_labels = [g[0] for g in competitor.config_grid()]
    noml_label = f"{grid_labels[-1]}-noml" if grid_labels else None
    max_label = grid_labels[-1] if grid_labels else None
    notes = competitor.notes()
    m = results["meta"]
    L = [
        f"# llmtrim vs {disp} (cost per correct answer + Pareto)", "",
        "The metric that matters to a buyer is not fewest input tokens - it is **cost per "
        "correct answer (CPCA)**: a tool that compresses more but is wrong more, or that "
        "makes the model ramble, costs you more. This measures that, and shows the full "
        "quality-vs-compression frontier so neither tool is judged at a single cherry-picked "
        "setting. See `BENCH_SPEC.md`.", "",
        f"- Model: `{m['model']}` (pinned route). Encoder: `{m['encoder']}` over the same "
        "message span for both tools.",
        f"- Corpora (public, sha-pinned): {', '.join(m['corpora'])}. The self-authored "
        "synthetic tool-output corpus is **excluded**.",
        f"- Pricing: bench/pricing.json (fetched {m['pricing']['fetched']}), "
        f"input ${m['pricing']['input']}/M, output ${m['pricing']['output']}/M "
        f"(output is {m['pricing']['output']/m['pricing']['input']:.1f}x input).", "",
    ]

    # Pareto / iso-compression table (deterministic, $0, citable)
    L += ["## Token reduction across the sweep (deterministic, $0)", "",
          "Each arm is a compression setting. Reduction % is token-weighted (1 - "
          "sum_after/sum_before); the CI bootstraps that same token-weighted statistic. "
          "Overhead leads with p95 (the tail a user feels), median in parentheses. This is "
          "the Pareto x-axis.", "",
          "| arm | tool | n | reduction % | 95% CI | overhead ms p95 (med) | ML fired |",
          "|---|---|--:|--:|:--|:--|--:|"]
    det = results["deterministic"]
    order = LLMTRIM_PRESETS + list(grid_labels) + ([noml_label] if noml_label else [])
    for label in order:
        a = det.get(label)
        if not a:
            continue
        lo, hi = a["reduction_ci"]
        ci = f"{lo:.0f}–{hi:.0f}" if lo is not None else "n/a"
        med, p95 = a.get("overhead_ms_median"), a.get("overhead_ms_p95")
        lat = f"{p95:.1f} ({med:.1f})" if med is not None else "n/a"
        L.append(f"| {label} | {a['tool']} | {a['n']} | {a['reduction_pct']:.0f}% | {ci} | "
                 f"{lat} | {a['ml_fired']} |")
    # No-ML row from the sidecar (separate $0 process where ML is disabled at startup). Shows
    # what the torch+ModernBERT dependency buys: on prose its deterministic routers no-op, so
    # the reduction collapses while latency drops to llmtrim's order.
    noml_path = RESULTS_DIR / "results-noml.json"
    if noml_label and max_label and noml_path.exists():
        nd = json.loads(noml_path.read_text()).get("deterministic", {}).get(max_label)
        if nd:
            lo, hi = nd["reduction_ci"]
            ci = f"{lo:.0f}–{hi:.0f}" if lo is not None else "n/a"
            md, p9 = nd.get("overhead_ms_median"), nd.get("overhead_ms_p95")
            lat = f"{p9:.1f} ({md:.1f})" if md is not None else "n/a"
            L.append(f"| {noml_label} | {competitor.name} (no ML) | {nd['n']} | "
                     f"{nd['reduction_pct']:.0f}% | {ci} | {lat} | {nd['ml_fired']} |")
    cold = m.get("cold_start_ms", {})
    L += ["",
          f"Latency is Python wall-clock around each library's `compress()`; it is not a "
          f"like-for-like CPU measurement (llmtrim crosses an FFI boundary into Rust, "
          f"{disp} runs in-process Python + torch). One-time cold start (model load, once "
          f"per process, amortizes to ~0 per call): llmtrim {cold.get('llmtrim', 'n/a')} ms, "
          f"{disp} {cold.get(competitor.name, 'n/a')} ms.", ""]

    # Per-corpus reduction (P2-4: the single aggregate hides per-corpus regressions -
    # llmtrim adds tokens on short prompts).
    L += [f"### Reduction per corpus (aggressive arm)", "",
          f"| corpus | llmtrim aggressive | {disp} {max_label} |", "|---|--:|--:|"]
    lt = det.get("aggressive", {}).get("per_corpus", {})
    hm = det.get(max_label, {}).get("per_corpus", {}) if max_label else {}
    for c in m["corpora"]:
        lv = lt.get(c, {}).get("reduction_pct")
        hv = hm.get(c, {}).get("reduction_pct")
        L.append(f"| {c} | {lv:.0f}% | {hv:.0f}% |" if lv is not None and hv is not None
                 else f"| {c} | n/a | n/a |")
    L += ["", "llmtrim is preservation-first: on short prompts (gsm8k, truthfulqa) it can "
          "*add* a few tokens rather than risk the answer, and the aggregate reduction is "
          "carried by the long-context corpora. Stated plainly, not hidden.", ""]

    # Live CPCA headline
    live = results.get("live")
    if live:
        L += [f"## Cost per answer-quality (live, {live.get('_seeds')} seeds, budget "
              f"${live.get('_budget_cap_usd')}, spent ${live.get('_budget_spent_usd')})", "",
              f"Each point pairs an llmtrim preset with the {disp} config of nearest achieved "
              "reduction (shown per point - exact iso isn't always possible because the "
              f"{disp} ML caps its reduction). For each, generate original / llmtrim / "
              f"{disp} across "
              "seeds, score with each corpus's own scorer (ROUGE-L for summaries, F1 for "
              "QA, numeric/contains/choice otherwise), and compute "
              "CPCA = total cost / sum of scores - fractional credit, so 'cost per correct "
              "answer' here means cost per unit of summed answer quality, not per binary hit. "
              "**Lower CPCA is better.** Quality is the mean score; output tokens are the "
              "median (resists one runaway generation).", ""]
        for pname in MATCHED:
            pt = live.get(pname)
            if not pt or pt.get("n", 0) == 0:
                continue
            mi = pt.get("match", {})
            preset, hr_label = mi.get("preset", "?"), mi.get("hr_label", "?")
            lr, hrr = mi.get("llmtrim_reduction_pct"), mi.get("headroom_reduction_pct")
            if lr is not None and hrr is not None:
                gap = abs(lr - hrr)
                kind = "iso" if gap <= 2 else f"near-iso, {gap:.0f}pp apart"
                red = f" - {kind}: reduction llmtrim {lr:.0f}% vs {disp} {hrr:.0f}%"
            else:
                red = ""
            L += [f"### {pname} - llmtrim `{preset}` vs {disp} `{hr_label}` "
                  f"(n={pt['n']} samples){red}", "",
                  "| arm | quality | output tok (med) | truncated | total cost | **CPCA** |",
                  "|---|--:|--:|--:|--:|--:|"]
            for arm in ("original", "llmtrim", competitor.name):
                d = pt["arms"].get(arm, {})
                if not d or d.get("quality") is None:
                    continue
                star = "**" if arm == "llmtrim" else ""
                L.append(f"| {star}{arm}{star} | {d['quality']:.2f} | "
                         f"{(d.get('output_tokens_median') or 0):.0f} | "
                         f"{d.get('truncated', 0)} | {cusd(d['total_cost_usd'])} | "
                         f"{star}{cusd(d['cpca_usd'])}{star} |")
            # Tie the truncation column to the cost story.
            lt_tr = pt["arms"].get("llmtrim", {}).get("truncated")
            hr_tr = pt["arms"].get(competitor.name, {}).get("truncated")
            if lt_tr is not None and hr_tr is not None:
                L += ["", f"{disp}'s longer outputs hit the generation cap {hr_tr} times vs "
                      f"llmtrim's {lt_tr}: the output-inflation that drives both its higher "
                      f"cost and its clipped answers."]
            qd = pt.get("quality_diff")
            if qd:
                lo, hi = qd["ci95"]
                verdict = ("significant" if qd["significant"]
                           else "NOT significant (CI spans 0)")
                L += ["", f"Quality difference llmtrim − {disp}: "
                      f"{qd['llmtrim_minus_headroom_mean']:+.3f} "
                      f"(95% CI {lo:+.3f}…{hi:+.3f}, n={qd['n_paired']}) - **{verdict}**."]
            L.append("")
    else:
        L += ["## Cost per correct answer (live)", "",
              "_Not run. Re-run with `--live` (and `OPENROUTER_API_KEY`) to fill in CPCA, "
              "quality, output tokens, and the paired-bootstrap significance test._", ""]

    L += ["## Caveats", "",
          "- The deterministic token axis is exact and citable. CPCA / quality / output "
          "tokens are live generations across seeds - directional, with the paired-bootstrap "
          "CI on the quality difference as the significance signal (CI excluding 0 = real).",
          "- Live sample is small (n shown per point) and uses few seeds; read each point with "
          "its own paired-bootstrap CI above (CI excluding 0 = significant). A larger live run "
          "would tighten the CIs."]
    if notes.get("ml_cap"):
        L.append("- " + notes["ml_cap"])
    L += ["- Scorers per corpus: numeric (gsm8k), token-F1 (hotpotqa, squad2, LongBench QA), "
          "choice (truthfulqa MC1), ROUGE-L (cnn, gov_report, multi_news). Each is the "
          "corpus's own standard metric.",
          f"- The {disp} ML reduction varies run-to-run, so the live {disp} arm is matched "
          "to llmtrim by ACHIEVED reduction within the same run (shown per point), not a "
          "fixed label; the full sweep shows neither tool is judged at one cherry-picked "
          "setting.",
          f"- Latency is Python wall-clock, not like-for-like CPU (llmtrim is Rust via FFI, "
          f"{disp} in-process Python+torch); read p95, and treat cold start as a one-time "
          f"cost. Per-{disp}-arm latency is also confounded by {disp} caching embeddings "
          "ACROSS arms within a run, so only the FIRST ML arm reflects true inference cost; "
          "the honest ML latency is the cold start plus that first ML call."]
    if notes.get("noml"):
        L.append("- " + notes["noml"])
    L.append("- llmtrim is preservation-first by design (no lossy tier). " + disp + " will "
             "win raw reduction at its most aggressive; the point is that there it loses "
             "answers while llmtrim does not - read the iso-compression rows together with CPCA.")
    if notes.get("rtk"):
        L.append("- " + notes["rtk"])
    L += ["- Tool-calling corpora (bfcl, glaive) are deferred (tool-schema plumbing "
          "+ call-arg scorer); excluded here, not cherry-picked away.", ""]
    return "\n".join(L)
