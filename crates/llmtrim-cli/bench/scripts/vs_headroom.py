#!/usr/bin/env python3
"""Head-to-head: llmtrim vs Headroom, both driven through their Python libraries.

This benchmark is built to be FAIR, not to manufacture a win. Earlier versions ran a
moderate llmtrim preset against Headroom's *most aggressive* config (the setting that most
hurts Headroom's answer quality) and scored quality only on llmtrim's own synthetic
tool-output corpus. Both biases are removed here:

MATCHED AGGRESSIVENESS — two comparison points, each pairing like with like:
  * Point A "moderate":   llmtrim preset `agent`      vs Headroom DEFAULTS
      (compress_user_messages=False, compress_system_messages=True, protect_recent=4,
       target_ratio=None, min_tokens_to_compress=250 — Headroom's actual CompressConfig
       defaults, headroom/compress.py).
  * Point B "aggressive": llmtrim preset `aggressive` vs Headroom MAX-aggression
      (compress_user_messages=True, compress_system_messages=True, protect_recent=0,
       target_ratio=0.2, min_tokens_to_compress=50).
  For each point we report token savings AND live answer quality for BOTH tools.

NEUTRAL CORPUS — two groups:
  * `general` — the REAL golden corpora (gsm8k, hotpotqa, squad2, contains-scored cases)
    with their own ground-truth answers. This is the less-biased quality signal: the golds
    are not on lines llmtrim is designed to keep.
  * `tool-output` — llmtrim's own synthetic tool-output corpus (synth_toolout.py). Clearly
    labelled as llmtrim-authored; it is Headroom's home turf on token savings but its golds
    favour llmtrim on quality, so it is the MORE-biased signal and is reported separately.

Fairness rules (unchanged):
- Both tools' before/after token counts use the SAME encoder (`o200k_base`) over the SAME
  span (the concatenated message contents), not each library's own internal metric.
- Both libraries see the SAME messages for each case.
- Latency excludes one-time model load (a warm-up runs first).
- Adversarial cases (gold deliberately in a line aggressive windowing elides) are reported
  SEPARATELY from faithful cases so neither tool's mean is distorted.

Setup (reproducible):
    crates/llmtrim-uniffi/scripts/build-wheel.sh --release       # build the llmtrim wheel
    pip install --user crates/../target/wheels/llmtrim-*.whl
    pip install --user -r bench/scripts/requirements-vs-headroom.txt
    python3 bench/scripts/download.py 40                         # the golden corpora
    HEADROOM_SRC=../headroom python3 bench/scripts/vs_headroom.py            # offline axes
    OPENROUTER_API_KEY=... HEADROOM_SRC=../headroom \
        python3 bench/scripts/vs_headroom.py --live --live-n 13  # + quality A/B

Outputs land in bench/snapshots/vs-headroom/: results.json and README.md.
"""
import argparse
import json
import os
import re
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]  # crates/llmtrim-cli (bench/ lives here)
WORKSPACE_ROOT = CRATE_ROOT.parents[1]  # the repo root, where .env sits
DATA_DIR = CRATE_ROOT / "bench" / "data"
RESULTS_DIR = CRATE_ROOT / "bench" / "snapshots" / "vs-headroom"

HEADROOM_SRC = Path(os.environ.get("HEADROOM_SRC", WORKSPACE_ROOT.parent / "headroom"))

# The model + route the project pins for every OpenRouter call (see CLAUDE.md).
MODEL = "openai/gpt-oss-20b"
PROVIDER_ROUTE = {"order": ["wandb/fp4"], "allow_fallbacks": False}

# The `model` field in the request body handed to each library's local compress(). NOT an
# API call: both libraries read it to pick a tokenizer. An OpenAI id makes llmtrim select its
# exact `o200k_base` tokenizer, the same encoder this script scores both tools with.
BODY_MODEL = "gpt-4o"

# ── The two matched comparison points ─────────────────────────────────────────
# Each pairs a llmtrim preset with a Headroom config of MATCHED aggressiveness, so neither
# tool is run against the other's worst-case setting.
HEADROOM_DEFAULTS = dict(
    compress_user_messages=False,
    compress_system_messages=True,
    protect_recent=4,
    target_ratio=None,
    min_tokens_to_compress=250,
)
HEADROOM_MAX = dict(
    compress_user_messages=True,
    compress_system_messages=True,
    protect_recent=0,
    target_ratio=0.2,
    min_tokens_to_compress=50,
)
POINTS = {
    "moderate": {"preset": "agent", "headroom": HEADROOM_DEFAULTS,
                 "desc": "llmtrim `agent` vs Headroom defaults"},
    "aggressive": {"preset": "aggressive", "headroom": HEADROOM_MAX,
                   "desc": "llmtrim `aggressive` vs Headroom max-aggression"},
}

# TLS verification stays ON. The live call goes through the llmtrim MITM proxy
# (HTTPS_PROXY=127.0.0.1:8788), so trust its CA via CURL_CA_BUNDLE / SSL_CERT_FILE
# (e.g. ~/.llmtrim/ca.pem) rather than disabling verification. cafile=None falls back to
# the system trust store. See CLAUDE.md: "Do not bypass the proxy to dodge TLS errors."
_CA_FILE = os.environ.get("CURL_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
_SSL_CTX = ssl.create_default_context(cafile=_CA_FILE)


# ── Shared tokenizer (the single fair denominator) ────────────────────────────
def get_encoder():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def span_text(messages):
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(c)
        elif c is not None:
            parts.append(json.dumps(c, separators=(",", ":")))
    return "\n".join(parts)


def count(enc, messages):
    return len(enc.encode(span_text(messages)))


# ── Corpora ───────────────────────────────────────────────────────────────────
def synthetic_tool_cases():
    """llmtrim's own synthetic tool-output corpus (synth_toolout.py): logs, diffs, grep,
    JSON, stack traces. Golds live on lines a faithful compressor keeps — llmtrim's home
    turf, so this is the MORE-biased quality signal. Reported separately from `general`."""
    path = DATA_DIR / "toolout.jsonl"
    if not path.exists():
        print(f"WARNING: {path} missing — run synth_toolout.py; tool-output group empty",
              file=sys.stderr)
        return []
    cases = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        v = json.loads(ln)
        question = v.get("question", "What is the single error and its resolution?")
        messages = [
            {"role": "user", "content": "Investigate the tool output and answer the question."},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "fetch", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": v["context"]},
            {"role": "user", "content": question},
        ]
        meta = {"question": question, "gold": v.get("gold"),
                "scorer": v.get("scorer", "contains"), "adversarial": bool(v.get("adversarial"))}
        cases.append((v["name"], messages, meta))
    return cases


# Golden corpora whose scorers we can compute deterministically and honestly here. We skip
# `judge` (needs an LLM judge) and `tool` (needs call-arg parsing) so every reported quality
# number is a scorer this script actually implements. gold lives in a real answer, NOT on a
# line llmtrim is designed to keep — this is the neutral quality signal.
GENERAL_CORPORA = ["gsm8k", "hotpotqa", "squad2"]
SUPPORTED_SCORERS = {"numeric", "f1", "contains"}


def general_cases(limit):
    """Real golden corpora in natural request shape (context + question). Each carries the
    corpus's own gold + scorer. Mirrors src/bench/mod.rs::build_request."""
    cases = []
    for name in GENERAL_CORPORA:
        path = DATA_DIR / f"{name}.jsonl"
        if not path.exists():
            continue
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        kept = 0
        for ln in lines:
            if kept >= limit:
                break
            v = json.loads(ln)
            scorer = v.get("scorer", "contains")
            if scorer not in SUPPORTED_SCORERS or v.get("gold") is None:
                continue
            msgs = []
            if v.get("system"):
                msgs.append({"role": "system", "content": v["system"]})
            ctx = next((v[k] for k in ("context", "input", "passage", "document") if v.get(k)), None)
            if ctx:
                msgs.append({"role": "user", "content": ctx})
            q = next((v[k] for k in ("question", "query", "prompt") if v.get(k)), None)
            if q:
                msgs.append({"role": "user", "content": q})
            if not msgs:
                continue
            meta = {"question": q, "gold": v["gold"], "scorer": scorer, "adversarial": False}
            cases.append((v["name"], msgs, meta))
            kept += 1
    return cases


# ── The two libraries ─────────────────────────────────────────────────────────
def llmtrim_compress(messages, preset, repeats):
    import llmtrim

    req = json.dumps({"model": BODY_MODEL, "messages": messages, "max_tokens": 300})
    durations = []
    out = None
    for _ in range(repeats):
        t = time.perf_counter()
        out = llmtrim.compress(req, llmtrim.Provider.OPEN_AI, preset)
        durations.append((time.perf_counter() - t) * 1000)
    out_messages = json.loads(out.request_json).get("messages", [])
    stages = [
        {"name": s.name, "applied": s.applied,
         "tokens_before": s.tokens_before, "tokens_after": s.tokens_after, "note": s.note}
        for s in out.stages
    ]
    return out_messages, stages, statistics.median(durations)


def headroom_ml_fired(transforms):
    return any(("router:text" in t) or ("kompress" in t.lower()) for t in transforms)


def headroom_compress(client, messages, kwargs, repeats):
    durations = []
    res = None
    for _ in range(repeats):
        t = time.perf_counter()
        res = client(messages, model=BODY_MODEL, **kwargs)
        durations.append((time.perf_counter() - t) * 1000)
    return res.messages, list(res.transforms_applied), statistics.median(durations)


def make_headroom_client():
    if HEADROOM_SRC.exists() and str(HEADROOM_SRC) not in sys.path:
        sys.path.insert(0, str(HEADROOM_SRC))
    try:
        from headroom import compress
        from headroom.transforms.kompress_compressor import is_kompress_available
    except Exception as e:  # noqa: BLE001
        print(f"headroom not importable: {e}", file=sys.stderr)
        return None
    if not is_kompress_available():
        print("headroom: Kompress ML path NOT available (install headroom-ai[ml]); "
              "running its deterministic path only", file=sys.stderr)
    return compress


# ── Live output A/B (gpt-oss-20b) ─────────────────────────────────────────────
def load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env = WORKSPACE_ROOT / ".env"
    if env.exists():
        for raw in env.read_text().splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if line.startswith("OPENROUTER_API_KEY="):
                val = line.split("=", 1)[1].strip()
                # Strip an inline comment (unquoted) and surrounding quotes.
                if val[:1] not in ("'", '"'):
                    val = val.split("#", 1)[0].strip()
                val = val.strip("'\"")
                print(f"WARNING: OPENROUTER_API_KEY not in env; using fallback from {env}",
                      file=sys.stderr)
                return val
    return None


def call_model(api_key, messages):
    """Returns the parsed response, or None on a transient failure (429 / timeout / 5xx /
    network) so a single flaky call skips that case instead of aborting the whole sweep."""
    payload = json.dumps({
        "model": MODEL, "messages": messages, "temperature": 0,
        "max_tokens": 2048, "provider": PROVIDER_ROUTE,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/fkiene/llmtrim", "X-Title": "llmtrim-vs-headroom"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90, context=_SSL_CTX) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  live call HTTPError {e.code}: skipping", file=sys.stderr)
            return None
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, OSError) as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  live call error {e}: skipping", file=sys.stderr)
            return None
    return None


def completion_tokens(resp):
    return (resp.get("usage") or {}).get("completion_tokens")


def answer_text(resp):
    choices = resp.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content") or ""


# ── Scorers (deterministic; named in the README) ──────────────────────────────
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm_tokens(s):
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return [t for t in s.split() if t]


def score(scorer, answer, gold):
    """Compute the corpus's own scorer over (answer, gold). Returns a float in [0,1]."""
    answer = answer or ""
    if scorer == "numeric":
        # Numbers may carry thousands separators ("1,234"); strip the comma from each
        # extracted token before float(), else gsm8k-style answers raise ValueError and
        # silently score 0.
        nums = _NUM_RE.findall(answer)
        g = str(gold).replace(",", "").strip()
        try:
            gv = float(g)
        except ValueError:
            return 1.0 if g.lower() in answer.lower() else 0.0
        for n in nums:
            try:
                if abs(float(n.replace(",", "")) - gv) < 1e-6:
                    return 1.0
            except ValueError:
                continue
        return 0.0
    if scorer == "f1":
        at, gt = _norm_tokens(answer), _norm_tokens(gold)
        if not gt:
            return 1.0 if not at else 0.0
        if not at:
            return 0.0
        common = 0
        gt_pool = list(gt)
        for t in at:
            if t in gt_pool:
                gt_pool.remove(t)
                common += 1
        if common == 0:
            return 0.0
        prec, rec = common / len(at), common / len(gt)
        return 2 * prec * rec / (prec + rec)
    # contains (default)
    g = str(gold).strip().lower()
    return 1.0 if (g == "" or g in answer.lower()) else 0.0


# ── Reporting ─────────────────────────────────────────────────────────────────
def pct(before, after):
    return 100.0 * (1 - after / before) if before else 0.0


def spread(vals):
    if not vals:
        return "n/a"
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{min(vals):.0f} / {statistics.median(vals):.0f} / {max(vals):.0f} (σ {sd:.0f})"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def qfmt(v):
    return f"{v * 100:.0f}%" if v is not None else "n/a"


def render(results):
    pts = results["points"]
    lines = ["# llmtrim vs Headroom (matched-config, fair)", "",
             "Both libraries are driven through their Python APIs (`llmtrim.compress`, "
             "`headroom.compress`). Before/after token counts use the **same** `o200k_base` "
             "encoder over the **same** message-content span. Latency is the median compress "
             f"time over {results['meta']['repeats']} runs (one-time model load excluded by a "
             "warm-up).", "",
             "**This comparison is matched, not rigged.** Each tool is run at TWO points, and "
             "at each point llmtrim's preset is paired with a Headroom config of the same "
             "aggressiveness, not against Headroom's worst-case setting:", "",
             "| point | llmtrim preset | Headroom config |", "|---|---|---|",
             "| moderate | `agent` | **defaults** (`compress_user_messages=False`, "
             "`protect_recent=4`, `target_ratio=None`, `min_tokens_to_compress=250`) |",
             "| aggressive | `aggressive` | **max** (`compress_user_messages=True`, "
             "`protect_recent=0`, `target_ratio=0.2`, `min_tokens_to_compress=50`) |", "",
             "Two corpora: **general** is the real golden corpora (gsm8k, hotpotqa, squad2) "
             "with their own ground-truth answers, the neutral quality signal. "
             "**tool-output** is llmtrim's own synthetic corpus (`synth_toolout.py`); its "
             "golds sit on lines llmtrim is built to keep, so it is llmtrim-favouring on "
             "quality and is reported separately.", ""]

    for pname, p in pts.items():
        lines += [f"## Point: {pname}, {POINTS[pname]['desc']}", ""]
        for group in ("general", "tool-output"):
            rows = [r for r in p["cases"] if r["group"] == group]
            if not rows:
                continue
            has_hr = bool(rows[0]["headroom"])
            lib = sum(r["llmtrim"]["before"] for r in rows)
            laf = sum(r["llmtrim"]["after"] for r in rows)
            hib = sum(r["headroom"]["before"] for r in rows) if has_hr else 0
            haf = sum(r["headroom"]["after"] for r in rows) if has_hr else 0
            lms = statistics.median([r["llmtrim"]["ms"] for r in rows])
            hms = statistics.median([r["headroom"]["ms"] for r in rows]) if has_hr else None
            l_sp = spread([r["llmtrim"]["saved_pct"] for r in rows])
            h_sp = spread([r["headroom"]["saved_pct"] for r in rows]) if has_hr else "n/a"
            label = ("real golden corpora, neutral signal" if group == "general"
                     else "llmtrim-authored synthetic, llmtrim-favouring")
            lines += [f"### {group} (n={len(rows)}, {label})", "",
                      "| tool | tokens before→after | saved | per-case spread % | median ms |",
                      "|---|--:|--:|:--|--:|",
                      f"| **llmtrim** | {lib:,} → {laf:,} | **{pct(lib, laf):.0f}%** | {l_sp} | {lms:.1f} |"]
            if has_hr:
                lines.append(f"| Headroom | {hib:,} → {haf:,} | {pct(hib, haf):.0f}% | {h_sp} | {hms:.1f} |")
                if group == "tool-output":
                    n_ml = sum(1 for r in rows if r["headroom"].get("ml_fired"))
                    lines.append(f"\nHeadroom's ML Kompress (ModernBERT) path fired on "
                                 f"{n_ml}/{len(rows)} tool-output cases at this point.")
            else:
                lines.append("| Headroom | (not installed) | n/a | n/a | n/a |")
            lines.append("")

        live = p.get("live")
        if live:
            lines += [f"### Live quality A/B at point `{pname}` (gpt-oss-20b)", "",
                      f"Each case sent to `{MODEL}` three ways (original / llmtrim / Headroom); "
                      "the answer is scored with the **corpus's own scorer** "
                      "(numeric / token-F1 / contains). Faithful and adversarial cases are "
                      "separated so neither tool's mean is distorted.", ""]
            for split in ("faithful", "adversarial"):
                s = live["splits"].get(split)
                if not s or s["n"] == 0:
                    continue
                lines += [f"**{split} cases (n={s['n']})**", "",
                          "| arm | answer accuracy | output tokens |", "|---|--:|--:|",
                          f"| original (uncompressed) | {qfmt(s['quality']['original'])} | "
                          f"{s['output_tokens']['original']:,} |",
                          f"| **llmtrim** | **{qfmt(s['quality']['llmtrim'])}** | "
                          f"{s['output_tokens']['llmtrim']:,} |",
                          f"| Headroom | {qfmt(s['quality']['headroom'])} | "
                          f"{s['output_tokens']['headroom']:,} |", ""]
            lines += ["<details><summary>Per-case quality (adv flagged)</summary>", "",
                      "| case | group | adv | scorer | original | llmtrim | Headroom |",
                      "|---|---|:-:|---|:-:|:-:|:-:|"]
            for r in live["per_case"]:
                def cell(k):
                    v = r.get(k)
                    return "n/a" if v is None else ("OK" if v >= 0.5 else "miss")
                lines.append(f"| {r['name']} | {r['group']} | {'Y' if r['adversarial'] else ''} "
                             f"| {r['scorer']} | {cell('q_original')} | {cell('q_llmtrim')} "
                             f"| {cell('q_headroom')} |")
            lines += ["", "</details>", ""]

    # Honest verdict, computed from the data.
    lines += ["## Verdict: who wins each axis", "", _verdict(results), ""]

    lines += ["## Caveats (read these)", "",
              "- **Matched configs, stated plainly.** At `moderate`, llmtrim runs its `agent` "
              "preset and Headroom runs its library DEFAULTS (which protect user messages and "
              "recent turns, so Headroom no-ops on many cases by design). At `aggressive`, "
              "llmtrim runs `aggressive` and Headroom runs its max config. Neither tool is "
              "pitted against the other's worst-case setting.",
              "- **Corpus bias.** The `tool-output` group is llmtrim's own synthetic corpus; "
              "its golds sit on lines llmtrim keeps, so it flatters llmtrim on quality. Treat "
              "the `general` group (real golden corpora) as the less-biased quality signal.",
              "- **Scorer.** Quality uses each corpus's own deterministic scorer "
              "(numeric / token-F1 / contains). We deliberately skip `judge` and `tool` cases "
              "(they need an LLM judge / call-arg parsing) so every number is a scorer this "
              "script actually computes. Token-F1 with a 0.5 OK threshold is lenient; read it "
              "as 'kept enough of the answer', not exact match.",
              "- **Small n.** The live A/B is a budget sweep (a dozen-ish scored cases per "
              "point). Numbers are directional, not a significance test. Transient API errors "
              "(429/timeout) skip that case rather than abort the run.",
              "- **Headroom's ML path.** Headroom runs with its `[ml]` extra enabled "
              "(ModernBERT Kompress + deterministic JSON/log/diff routers); no generative LLM "
              "call, model load excluded from latency.",
              "- **Reproducibility.** Only the token-count axis (before/after/saved) is "
              "deterministic and citable. The `median ms` latency is machine-specific and the "
              "live `output tokens` are single-run, non-deterministic generations; read both "
              "as directional, not point estimates.",
              "- Model is `openai/gpt-oss-20b` via the pinned `wandb/fp4` route (CLAUDE.md).",
              ""]
    return "\n".join(lines)


def _verdict(results):
    """Plain-language summary of who wins each axis, computed from the results."""
    out = []
    for pname, p in results["points"].items():
        for group in ("general", "tool-output"):
            rows = [r for r in p["cases"] if r["group"] == group]
            if not rows or not rows[0]["headroom"]:
                continue
            lt = pct(sum(r["llmtrim"]["before"] for r in rows),
                     sum(r["llmtrim"]["after"] for r in rows))
            hr = pct(sum(r["headroom"]["before"] for r in rows),
                     sum(r["headroom"]["after"] for r in rows))
            who = "llmtrim" if lt > hr + 1 else ("Headroom" if hr > lt + 1 else "tie")
            out.append(f"- **{pname} / {group} tokens:** llmtrim {lt:.0f}% vs Headroom "
                       f"{hr:.0f}% → **{who}**.")
        live = p.get("live")
        if live:
            for split in ("faithful", "adversarial"):
                s = live["splits"].get(split)
                if not s or s["n"] == 0:
                    continue
                lq, hq = s["quality"]["llmtrim"], s["quality"]["headroom"]
                oq = s["quality"]["original"]
                if lq is None or hq is None:
                    continue
                who = "llmtrim" if lq > hq + 0.01 else ("Headroom" if hq > lq + 0.01 else "tie")
                out.append(f"- **{pname} / {split} quality (n={s['n']}):** original "
                           f"{qfmt(oq)}, llmtrim {qfmt(lq)}, Headroom {qfmt(hq)} → **{who}**.")
    return "\n".join(out) if out else "(no Headroom data - install Headroom to compare.)"


# ── Driver ────────────────────────────────────────────────────────────────────
def compress_point(enc, hr, preset, hr_kwargs, all_cases, repeats):
    """Token-savings axis for one matched point. Returns (case_records, ml_any)."""
    recs, ml_any = [], False
    for group, name, messages, meta in all_cases:
        lt_msgs, stages, lt_ms = llmtrim_compress(messages, preset, repeats)
        l_before, l_after = count(enc, messages), count(enc, lt_msgs)
        rec = {"group": group, "name": name, "meta": meta,
               "llmtrim": {"before": l_before, "after": l_after,
                           "saved_pct": pct(l_before, l_after), "ms": lt_ms, "stages": stages},
               "headroom": None}
        if hr is not None:
            hr_msgs, transforms, hr_ms = headroom_compress(hr, messages, hr_kwargs, repeats)
            h_before, h_after = count(enc, messages), count(enc, hr_msgs)
            ml = headroom_ml_fired(transforms)
            ml_any = ml_any or ml
            rec["headroom"] = {"before": h_before, "after": h_after,
                               "saved_pct": pct(h_before, h_after), "ms": hr_ms,
                               "transforms": transforms, "ml_fired": ml}
        recs.append(rec)
    return recs, ml_any


def live_point(key, enc, hr, preset, hr_kwargs, all_cases, live_n):
    """Quality A/B for one matched point. Picks a balanced budget across groups, scores each
    arm with the corpus's own scorer, splits faithful vs adversarial."""
    # Balance the budget: general (neutral signal) + faithful tool-output, and ALWAYS
    # include some adversarial tool-output cases so the adversarial split is non-empty and
    # both tools are tested on the hard cases (gold on a line aggressive windowing elides).
    scored_budget = [c for c in all_cases if (c[3] or {}).get("gold") is not None]
    gen = [c for c in scored_budget if c[0] == "general"]
    tool_f = [c for c in scored_budget if c[0] == "tool-output" and not (c[3] or {}).get("adversarial")]
    tool_a = [c for c in scored_budget if c[0] == "tool-output" and (c[3] or {}).get("adversarial")]
    n_gen = max(1, live_n // 3)
    n_adv = min(len(tool_a), max(2, live_n // 4))
    n_faith = live_n - n_gen - n_adv
    budget = gen[:n_gen] + tool_f[:n_faith] + tool_a[:n_adv]

    per_case = []
    for group, name, messages, meta in budget:
        gold = meta["gold"]
        scorer = meta.get("scorer", "contains")
        lt_msgs, _, _ = llmtrim_compress(messages, preset, 1)
        arms = {"original": messages, "llmtrim": lt_msgs}
        ml_fired = None
        if hr is not None:
            arms["headroom"], hr_transforms, _ = headroom_compress(hr, messages, hr_kwargs, 1)
            ml_fired = headroom_ml_fired(hr_transforms)
        row = {"name": name, "group": group, "scorer": scorer,
               "adversarial": bool(meta.get("adversarial")),
               "headroom_ml_fired": ml_fired,
               "out_tokens": {}}
        # On a transient (429/timeout) failure of ONE arm, keep the arms that succeeded and
        # mark the failed arm's quality null for this row, rather than dropping the whole case.
        any_ok = False
        for arm, arm_msgs in arms.items():
            resp = call_model(key, arm_msgs)
            if resp is None:
                row[f"q_{arm}"] = None
                continue
            any_ok = True
            ct = completion_tokens(resp) or 0
            row["out_tokens"][arm] = ct
            row[f"q_{arm}"] = score(scorer, answer_text(resp), gold)
            time.sleep(1)
        if not any_ok:
            print(f"  live skip {name} (all arms failed)", file=sys.stderr)
            continue
        per_case.append(row)
        print(f"  qa[{preset:10}] {name:22} "
              + " ".join(f"{a}={row.get('q_'+a)}" for a in arms))

    def split_stats(rows):
        ot = {"original": 0, "llmtrim": 0, "headroom": 0}
        for r in rows:
            for a in ot:
                ot[a] += r["out_tokens"].get(a, 0)
        return {"n": len(rows),
                "quality": {a: mean([r.get(f"q_{a}") for r in rows])
                            for a in ("original", "llmtrim", "headroom")},
                "output_tokens": ot}

    faithful = [r for r in per_case if not r["adversarial"]]
    adversarial = [r for r in per_case if r["adversarial"]]
    return {"splits": {"faithful": split_stats(faithful),
                       "adversarial": split_stats(adversarial)},
            "per_case": per_case}


def main():
    ap = argparse.ArgumentParser(description="llmtrim vs Headroom (matched-config) benchmark")
    ap.add_argument("--limit", type=int, default=8, help="cases per general corpus")
    ap.add_argument("--repeats", type=int, default=5, help="latency samples per case (median)")
    ap.add_argument("--live", action="store_true", help="run the gpt-oss-20b quality A/B")
    ap.add_argument("--live-n", type=int, default=13, help="scored cases per point")
    args = ap.parse_args()

    enc = get_encoder()
    hr = make_headroom_client()

    # Warm both libraries (one-time setup excluded from latency).
    warm = [
        {"role": "user", "content": "warm-up"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_w", "type": "function",
                         "function": {"name": "fetch", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_w",
         "content": json.dumps({"results": [{"x": i} for i in range(50)]})},
    ]
    for p in POINTS.values():
        llmtrim_compress(warm, p["preset"], 1)
    if hr is not None:
        warm_text = warm + [{"role": "user", "content": " ".join(
            ["the quarterly report shows revenue growth across regions and controlled costs"] * 40)}]
        for p in POINTS.values():
            try:
                hr(warm_text, model=BODY_MODEL, **p["headroom"])
            except Exception as e:  # noqa: BLE001
                print(f"headroom warm-up failed: {e}", file=sys.stderr)

    all_cases = [("general", n, m, meta) for n, m, meta in general_cases(args.limit)]
    all_cases += [("tool-output", n, m, meta) for n, m, meta in synthetic_tool_cases()]
    print(f"corpus: {sum(1 for c in all_cases if c[0]=='general')} general, "
          f"{sum(1 for c in all_cases if c[0]=='tool-output')} tool-output", file=sys.stderr)

    key = None
    if args.live:
        key = load_api_key()
        if not key:
            print("ERROR: --live needs OPENROUTER_API_KEY (env or .env)", file=sys.stderr)
            sys.exit(1)

    points_out = {}
    for pname, p in POINTS.items():
        print(f"\n=== point {pname}: {p['desc']} ===", file=sys.stderr)
        recs, ml_any = compress_point(enc, hr, p["preset"], p["headroom"], all_cases, args.repeats)
        for r in recs:
            print(f"  {r['group']:11} {r['name']:22} llmtrim {r['llmtrim']['saved_pct']:4.0f}% "
                  + (f"headroom {r['headroom']['saved_pct']:4.0f}%"
                     + (" [ml]" if r["headroom"]["ml_fired"] else "")
                     if r["headroom"] else "headroom n/a"))
        live = None
        if args.live:
            live = live_point(key, enc, hr, p["preset"], p["headroom"], all_cases, args.live_n)
        points_out[pname] = {"preset": p["preset"], "headroom_kwargs": p["headroom"],
                             "headroom_ml_fired": ml_any if hr is not None else None,
                             "cases": recs, "live": live}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {"model": MODEL, "repeats": args.repeats, "encoder": "o200k_base",
                 "note": ("Token-count fields (before/after/saved_pct) are deterministic and "
                          "citable. The latency `ms` fields are machine-specific and the live "
                          "`out_tokens` are single-run non-deterministic generations — treat "
                          "both as directional noise, NOT point estimates to cite."),
                 "headroom_installed": hr is not None,
                 "general_corpora": GENERAL_CORPORA,
                 "general_scorers": sorted(SUPPORTED_SCORERS),
                 "points": {k: {"preset": v["preset"], "headroom": v["headroom"]}
                            for k, v in POINTS.items()}},
        "points": points_out,
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    report = render(results)
    (RESULTS_DIR / "README.md").write_text(report + "\n")
    print(f"\nWrote {RESULTS_DIR}/results.json and README.md\n")
    print(report)


if __name__ == "__main__":
    main()
