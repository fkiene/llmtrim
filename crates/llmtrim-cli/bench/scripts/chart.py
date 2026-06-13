#!/usr/bin/env python3
"""Render the README hero — bench/frontier-dark.svg + bench/frontier-light.svg.

"Whole round-trip" framing: a before/after cost bar where llmtrim's bill stops short and the
dashed "ghost" of the original is annotated as what you don't pay. One supporting line (the
differentiator the bars can't show) + a thin credibility footer; the proof numbers and
adoption signals live in the README beside the image.

Two themes matched to GitHub's own surfaces (canvas + success-green + text/border) so the panel
blends into the README in either color scheme; the README swaps them via `<picture>` +
`prefers-color-scheme`. Static SVG (GitHub renders as <img>: no JS/web-fonts; system fonts; the
only theme adaptation is the two-file swap). Pooled from the shape-matched live A/B (same data as
bench/README), priced with the pinned models.dev snapshot. Run: `python3 bench/scripts/chart.py`.
"""
import glob
import html
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bench/ root (this script lives in bench/scripts/)
SANS = "ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"

# GitHub's own palettes: canvas, border, text (primary/muted/subtle), success-green, hairline.
# The bill bars are a neutral slate (cost is just cost); green is reserved for the win.
THEMES = {
    "dark": dict(
        bg="#0d1117", border="#30363d", hair="#21262d",
        ink="#e6edf3", mute="#8b949e", foot="#7d8590",
        green="#3fb950", green_soft="#56d364",
        in_fill="#2b3a48", bill0="#34506b", bill1="#4d7aa3",
        seg_in="#c3d2de", seg_out="#eaf2f8", glow=0.5, ghost=0.09,
    ),
    "light": dict(
        bg="#ffffff", border="#d0d7de", hair="#d8dee4",
        ink="#1f2328", mute="#59636e", foot="#6a727c",
        green="#1a7f37", green_soft="#1f883d",
        in_fill="#c8d4e0", bill0="#9fb8d2", bill1="#6e93b6",
        seg_in="#222b33", seg_out="#0d2942", glow=0.18, ghost=0.13,
    ),
}


def pooled():
    """Sum tokens + cost across the shape-matched run (results/<corpus>.json) — the same data
    bench/README pools; variant files (`__safe`/`__aggressive`/`__tuned`) excluded."""
    tin_b = tin_a = tout_b = tout_a = n = 0
    cost_b = cost_a = 0.0
    model = ""
    for path in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if "__" in name or name == "run":
            continue
        d = json.load(open(path))
        model = model or d.get("model", "")
        for c in d.get("cases", []):
            tin_b += c["tokens_in_before"]; tin_a += c["tokens_in_after"]
            tout_b += c["tokens_out_orig"]; tout_a += c["tokens_out_comp"]
            cost_b += c["cost_orig"]; cost_a += c["cost_comp"]
            n += 1
    return tin_b, tin_a, tout_b, tout_a, cost_b, cost_a, model, n


def model_rates(model):
    """(input, output) per-token rate for `model` from the pinned snapshot."""
    try:
        models = json.load(open(os.path.join(HERE, "pricing.json"))).get("models", {})
        m = models.get(model) or models.get(model.split("/", 1)[-1])
        if not m:
            m = next((v for k, v in models.items() if model.endswith(k)), None)
        return float(m["input"]), float(m["output"])
    except Exception:
        return 0.075, 0.3  # gpt-oss-20b fallback


def pct(b, a):
    return (b - a) / b * 100 if b else 0.0


def txt(x, y, size, fill, s, weight=400, anchor="start", fam=SANS, ls=0.0, extra=""):
    e = f' letter-spacing="{ls}"' if ls else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" font-family="{fam}"{e}{extra}>{s}</text>')


def rrect(x, y, w, h, r, left=True, right=True):
    """Rect path with independent left / right corner rounding, so two segments meet flush
    (square inner edges) and read as ONE bar with rounded outer ends, not two pinching pills."""
    r = min(r, h / 2.0, max(w, 0.1))
    rl, rr = (r if left else 0.0), (r if right else 0.0)
    p = [f"M{x + rl:.1f},{y:.1f}", f"H{x + w - rr:.1f}"]
    if rr:
        p.append(f"A{rr:.1f},{rr:.1f} 0 0 1 {x + w:.1f},{y + rr:.1f}")
    p.append(f"V{y + h - rr:.1f}")
    if rr:
        p.append(f"A{rr:.1f},{rr:.1f} 0 0 1 {x + w - rr:.1f},{y + h:.1f}")
    p.append(f"H{x + rl:.1f}")
    if rl:
        p.append(f"A{rl:.1f},{rl:.1f} 0 0 1 {x:.1f},{y + h - rl:.1f}")
    p.append(f"V{y + rl:.1f}")
    if rl:
        p.append(f"A{rl:.1f},{rl:.1f} 0 0 1 {x + rl:.1f},{y:.1f}")
    p.append("Z")
    return " ".join(p)


def render(T, d):
    cost_b, cost_a, model, n, fi_b, fi_a, cost_pct, out_pct, in_pct = d
    W, H = 860, 284
    x0, full, bh = 168, 540, 40
    comp = full * (cost_a / cost_b) if cost_b else full   # llmtrim bar width
    y_o, y_l = 96, 152
    end = x0 + full

    def seg(cx, y, s, fill):
        return txt(cx, y + bh / 2 + 4, 10.5, fill, s, weight=600, anchor="middle", ls=0.4)

    def cost_bar(y, width, fi, bold, money, round_right=True):
        iw, ow = width * fi, width * (1 - fi)
        return [
            f'<path d="{rrect(x0, y, iw, bh, 6, left=True, right=False)}" fill="{T["in_fill"]}"/>',
            f'<path d="{rrect(x0 + iw - 0.5, y, ow + 0.5, bh, 6, left=False, right=round_right)}" fill="url(#bill)"/>',
            txt(x0 + width + 14, y + bh / 2 + 6, 17, (T["ink"] if bold else T["mute"]),
                money, weight=(800 if bold else 600), fam=MONO),
        ]

    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        f'<linearGradient id="bill" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="{T["bill0"]}"/><stop offset="100%" stop-color="{T["bill1"]}"/></linearGradient>',
        f'<filter id="win" x="-45%" y="-65%" width="190%" height="230%"><feDropShadow dx="0" dy="0" stdDeviation="5" flood-color="{T["green"]}" flood-opacity="{T["glow"]}"/></filter>',
        "</defs>",
        # panel matched to the GitHub surface: canvas fill + 1px border, like a native card
        f'<rect width="{W}" height="{H}" fill="{T["bg"]}"/>',
        f'<rect x="1" y="1" width="{W - 2}" height="{H - 2}" rx="12" fill="none" stroke="{T["border"]}" stroke-width="1"/>',
        # header — the value prop: ink (not muted) + bold so it reads, but sized below the
        # −46% and kept all-ink so the one green number stays the sole focal point
        f'<text x="44" y="52" font-family="{SANS}"><tspan font-size="19" font-weight="800" letter-spacing="0.2" fill="{T["ink"]}">llmtrim</tspan><tspan font-size="18" font-weight="700" fill="{T["ink"]}"> cuts the whole LLM bill, </tspan><tspan font-size="18" font-weight="800" fill="{T["ink"]}">both ends</tspan></text>',
        # the comparison (the hero)
        txt(x0 - 16, y_o + bh / 2 + 5, 14, T["mute"], "original", weight=600, anchor="end"),
        *cost_bar(y_o, full, fi_b, False, f"${cost_b:.4f}"),
        seg(x0 + full * fi_b / 2, y_o, "input", T["seg_in"]),
        seg(x0 + full * fi_b + full * (1 - fi_b) / 2, y_o, "output", T["seg_out"]),
        txt(x0 - 16, y_l + bh / 2 + 5, 14, T["ink"], "llmtrim", weight=800, anchor="end"),
        *cost_bar(y_l, comp, fi_a, True, f"${cost_a:.4f}", round_right=False),
        # per-axis cut on the llmtrim (after) segments: how much each end shrank
        seg(x0 + comp * fi_a / 2, y_l, f"−{in_pct:.0f}%", T["seg_in"]),
        seg(x0 + comp * fi_a + comp * (1 - fi_a) / 2, y_l, f"−{out_pct:.0f}%", T["seg_out"]),
        # savings ghost: open on the LEFT (flush with the solid bar), rounded outer right, so
        # the llmtrim row reads as ONE full-width bar — solid paid → dashed saved.
        f'<path d="M{x0 + comp:.1f},{y_l} H{end - 6:.1f} A6,6 0 0 1 {end:.1f},{y_l + 6} V{y_l + bh - 6:.1f} A6,6 0 0 1 {end - 6:.1f},{y_l + bh} H{x0 + comp:.1f}" fill="{T["green"]}" fill-opacity="{T["ghost"]}" stroke="{T["green"]}" stroke-opacity="0.55" stroke-width="1.4" stroke-dasharray="5 4"/>',
        txt(end - 16, y_l + bh / 2 + 12, 34, T["green"], f"−{cost_pct:.0f}%", weight=800,
            anchor="end", ls=-1, extra=' filter="url(#win)"'),
        txt(end - 16, y_l + bh + 14, 12.5, T["green_soft"], "round-trip cost you don't pay",
            weight=600, anchor="end", ls=0.2),
        # one supporting line (the differentiator the bars can't show) + a thin footer; the
        # proof numbers and adoption signals live in the README right beside this image.
        f'<text x="44" y="240" font-size="14.5" font-family="{SANS}" fill="{T["mute"]}">caveman cuts <tspan fill="{T["ink"]}">output</tspan> · rtk cuts <tspan fill="{T["ink"]}">tool output</tspan> · llmtrim cuts <tspan fill="{T["green"]}" font-weight="800">both ends</tspan></text>',
        f'<line x1="44" y1="258" x2="{W - 44}" y2="258" stroke="{T["hair"]}" stroke-width="1"/>',
        txt(44, 272, 11, T["foot"], f"measured: {n} live A/B cases · {html.escape(model)} · full results in bench/README", fam=MONO),
        "</svg>",
    ]
    return "\n".join(s)


def main():
    tin_b, tin_a, tout_b, tout_a, cost_b, cost_a, model, n = pooled()
    ri, ro = model_rates(model)
    fi_b = (tin_b * ri) / (tin_b * ri + tout_b * ro)   # input share of cost, original
    fi_a = (tin_a * ri) / (tin_a * ri + tout_a * ro)   # ... compressed
    d = (cost_b, cost_a, model, n, fi_b, fi_a,
         pct(cost_b, cost_a), pct(tout_b, tout_a), pct(tin_b, tin_a))
    for name, T in THEMES.items():
        open(os.path.join(HERE, f"frontier-{name}.svg"), "w").write(render(T, d))
    print(f"wrote frontier-dark.svg + frontier-light.svg "
          f"(cost −{d[6]:.0f}% · output −{d[7]:.0f}% · input −{d[8]:.0f}% · n={n})")
    # Projection: the SAME measured token deltas, re-priced at frontier-model rates from
    # the pinned snapshot. No new measurement — frontier models weight output (the −75%
    # side) far more heavily than gpt-oss-20b does, so the cost % rises. Labeled as a
    # projection wherever published; the hero number stays the measured one.
    for fid in ("openai/gpt-4o", "anthropic/claude-sonnet-4.5"):
        try:
            fri, fro = model_rates(fid)
        except Exception:
            continue
        pb = tin_b * fri + tout_b * fro
        pa = tin_a * fri + tout_a * fro
        if pb:
            print(f"projection @ {fid} rates: cost −{pct(pb, pa):.0f}%")


if __name__ == "__main__":
    main()
