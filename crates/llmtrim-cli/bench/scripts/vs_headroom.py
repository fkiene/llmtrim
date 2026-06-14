#!/usr/bin/env python3
"""Head-to-head: llmtrim vs Headroom, both driven through their Python libraries.

What this measures, per case, on a single shared token denominator:

1. Input-token reduction — how much each library removes from the request.
2. Per-stage attribution — llmtrim's `stages` (real per-stage token deltas) vs
   Headroom's `transforms_applied` (the strategies it ran).
3. Compression overhead — wall-clock milliseconds per compress call (median of K runs,
   model load excluded by a warm-up).
4. (`--live`) Output-token savings on a real model — original vs llmtrim-compressed
   request sent to gpt-oss-20b. Headroom is input-only, so its output column is 0 by
   construction; that asymmetry is the point, stated plainly.

Fairness rules (so the comparison is apples-to-apples):
- Both tools' before/after token counts use the SAME encoder (`o200k_base`) over the SAME
  span (the concatenated message contents), not each library's own internal metric.
- Both libraries see the SAME messages for each case.
- Two corpus groups make the coverage difference explicit:
    * `tool-output` — Headroom's home turf (large JSON / logs / shell / grep as tool
      results); both libraries compress here.
    * `general` — our golden corpora (gsm8k, hotpotqa, dolly, …) in their natural request
      shape. Headroom protects user/system messages, so it mostly no-ops; llmtrim still
      compresses. This shows breadth of coverage, not a rigged input.

Setup (reproducible):
    cargo build --release                                   # not required (uses the lib)
    crates/llmtrim-uniffi/scripts/build-wheel.sh            # build the llmtrim wheel
    pip install --user crates/../target/wheels/llmtrim-*.whl
    pip install --user -r bench/scripts/requirements-vs-headroom.txt
    python3 bench/scripts/download.py 40                    # our golden corpora
    python3 bench/scripts/vs_headroom.py                    # deterministic axes (offline)
    OPENROUTER_API_KEY=... python3 bench/scripts/vs_headroom.py --live --live-n 12

Outputs land in bench/results-vs-headroom/: results.json (machine-readable) and README.md
(the rendered tables + the drop-in snippet for the main README "How does it compare").
"""
import argparse
import json
import os
import ssl
import statistics
import sys
import time
import urllib.request
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[2]  # crates/llmtrim-cli (bench/ lives here)
WORKSPACE_ROOT = CRATE_ROOT.parents[1]  # the repo root, where .env sits
DATA_DIR = CRATE_ROOT / "bench" / "data"
RESULTS_DIR = CRATE_ROOT / "bench" / "results-vs-headroom"

# The model + route the project pins for every OpenRouter call (see CLAUDE.md). Used only
# by the live A/B (call_model); the deterministic axes make no network calls.
MODEL = "openai/gpt-oss-20b"
PROVIDER_ROUTE = {"order": ["wandb/fp4"], "allow_fallbacks": False}

# The `model` field placed in the request *body* handed to each library's local compress().
# It is NOT an API call: both libraries read it to pick a tokenizer / context limits. We use
# an OpenAI model on purpose so llmtrim selects its exact `o200k_base` tokenizer, the same
# encoder this script scores both tools with (get_encoder); a non-OpenAI id would make
# llmtrim fall back to an estimated tokenizer and blur the denominator.
BODY_MODEL = "gpt-4o"
LLMTRIM_PRESET_DEFAULT = "agent"  # general agent traffic — the closest analog to Headroom's use

# Direct OpenRouter calls for the live A/B bypass any local proxy (otherwise the llmtrim
# daemon would re-compress the "original" arm and contaminate the baseline) and tolerate a
# missing CA chain (same posture as bench/scripts/caveman_ab.py and the `bench suite` proxy guard).
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


# ── Shared tokenizer (the single fair denominator) ────────────────────────────
def get_encoder():
    import tiktoken

    return tiktoken.get_encoding("o200k_base")


def span_text(messages):
    """The content span both libraries are scored over: every message's content,
    flattened to text (objects JSON-dumped), joined. Same definition for both tools."""
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
    """Headroom's published content types, sized to mirror its benchmark, each delivered
    as a tool result (the shape Headroom's smart-crusher targets)."""

    def recs(n):
        a = []
        for i in range(n):
            r = {
                "ts": f"2026-04-{i % 28 + 1:02d}T10:{i % 60:02d}:00Z",
                "level": "INFO",
                "service": f"svc-{i % 5}",
                "msg": f"request {i} handled ok",
                "code": 200,
                "ms": i % 50,
            }
            if i == 67:
                r.update({"level": "ERROR", "msg": "payment gateway declined",
                          "code": 402, "resolution": "retry with backup PSP", "affected": 1432})
            a.append(r)
        return json.dumps({"results": a})

    raw = {
        "json_100": recs(100),
        "json_500": recs(500),
        "shell_200": "\n".join(
            f"drwxr-xr-x 2 u g {1024 * i:8d} Apr {i % 28 + 1:02d} file_{i}.txt" for i in range(200)
        ),
        "buildlog_200": "\n".join(f"[{i:03d}] INFO compiling crate mod_{i} ok" for i in range(198))
        + "\nERROR undefined reference to render_frame\nERROR build failed",
        "grep_150": "\n".join(
            f"src/{'abcde'[i % 5]}.rs:{i + 1}:    let v = connect({i});" for i in range(150)
        ),
    }
    cases = []
    for name, content in raw.items():
        # A realistic tool-call turn: the model asked, the tool answered (big), the user
        # follows up. Both libraries get this identical message list.
        messages = [
            {"role": "user", "content": "Investigate the service and summarize what happened."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "fetch", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": content},
            {"role": "user", "content": "What is the single error and its resolution?"},
        ]
        cases.append((name, messages))
    return cases


def general_cases(limit):
    """Our golden corpora in their natural request shape (system + context + question).
    Mirrors crates/llmtrim-cli/src/bench/mod.rs::build_request."""
    corpora = ["gsm8k", "humaneval", "dolly", "hotpotqa", "glaive", "chat", "cnn", "toolout"]
    cases = []
    for name in corpora:
        path = DATA_DIR / f"{name}.jsonl"
        if not path.exists():
            continue
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()][:limit]
        for i, ln in enumerate(lines):
            v = json.loads(ln)
            if "request" in v:  # explicit request form
                try:
                    msgs = json.loads(v["request"]).get("messages", [])
                except (json.JSONDecodeError, AttributeError):
                    continue
            else:  # friendly form
                msgs = []
                if v.get("system"):
                    msgs.append({"role": "system", "content": v["system"]})
                ctx = next((v[k] for k in ("context", "input", "passage", "document") if v.get(k)), None)
                if ctx:
                    msgs.append({"role": "user", "content": ctx})
                q = next((v[k] for k in ("question", "query", "prompt") if v.get(k)), None)
                if q:
                    msgs.append({"role": "user", "content": q})
            if msgs:
                cases.append((f"{name}-{i}", msgs))
    return cases


# ── The two libraries ─────────────────────────────────────────────────────────
def llmtrim_compress(messages, preset, repeats):
    """Compress with the llmtrim wheel. Returns (out_messages, stages, ms) where stages is
    the per-stage breakdown now exposed by the binding."""
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


def headroom_compress(client, messages, repeats):
    """Compress with Headroom's library. Returns (out_messages, transforms, ms)."""
    durations = []
    res = None
    for _ in range(repeats):
        t = time.perf_counter()
        res = client(messages, model=BODY_MODEL)
        durations.append((time.perf_counter() - t) * 1000)
    return res.messages, list(res.transforms_applied), statistics.median(durations)


def make_headroom_client():
    """The offline compress entrypoint. Imported lazily so the deterministic-only path can
    still run if Headroom is absent (it is then reported as not installed)."""
    try:
        from headroom import compress
    except Exception as e:  # noqa: BLE001 - report, don't crash the whole run
        print(f"headroom not importable: {e}", file=sys.stderr)
        return None
    return compress


# ── Live output A/B (gpt-oss-20b) ─────────────────────────────────────────────
def load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env = WORKSPACE_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def call_model(api_key, messages):
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 2048,
        "provider": PROVIDER_ROUTE,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/fkiene/llmtrim", "X-Title": "llmtrim-vs-headroom"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90, context=_SSL_CTX) as r:
        return json.loads(r.read())


def completion_tokens(resp):
    return (resp.get("usage") or {}).get("completion_tokens")


# ── Reporting ─────────────────────────────────────────────────────────────────
def pct(before, after):
    return 100.0 * (1 - after / before) if before else 0.0


def render(results, live):
    lines = ["# llmtrim vs Headroom", "",
             "Both libraries driven through their Python APIs (`llmtrim.compress` and "
             "`headroom.compress`). Before/after token counts use the **same** `o200k_base` "
             "encoder over the **same** message-content span. Latency is the median compress "
             f"time over {results['meta']['repeats']} runs (model load excluded). "
             f"llmtrim preset: `{results['meta']['preset']}`.", ""]

    for group in ("tool-output", "general"):
        rows = [r for r in results["cases"] if r["group"] == group]
        if not rows:
            continue
        has_hr = bool(rows[0]["headroom"])
        lib = sum(r["llmtrim"]["before"] for r in rows)
        laf = sum(r["llmtrim"]["after"] for r in rows)
        hib = sum(r["headroom"]["before"] for r in rows) if has_hr else 0
        haf = sum(r["headroom"]["after"] for r in rows) if has_hr else 0
        lms = statistics.median([r["llmtrim"]["ms"] for r in rows])
        hms = statistics.median([r["headroom"]["ms"] for r in rows]) if has_hr else None
        lines += [f"## {group} (n={len(rows)})", "",
                  "| tool | input tokens before→after | input saved | median ms |",
                  "|---|--:|--:|--:|",
                  f"| **llmtrim** | {lib:,} → {laf:,} | **{pct(lib, laf):.0f}%** | {lms:.1f} |"]
        if has_hr:
            lines.append(f"| Headroom | {hib:,} → {haf:,} | {pct(hib, haf):.0f}% | {hms:.1f} |")
        else:
            lines.append("| Headroom | (not installed) | n/a | n/a |")
        lines.append("")

    # llmtrim per-stage (its differentiator), summed over the tool-output group.
    tool_rows = [r for r in results["cases"] if r["group"] == "tool-output"]
    if tool_rows:
        agg = {}
        for r in tool_rows:
            for s in r["llmtrim"]["stages"]:
                a = agg.setdefault(s["name"], {"before": 0, "after": 0, "applied": 0})
                a["before"] += s["tokens_before"]
                a["after"] += s["tokens_after"]
                a["applied"] += 1 if s["applied"] else 0
        lines += ["## llmtrim per-stage attribution (tool-output group)", "",
                  "Each stage's own token delta, the breakdown the binding now exposes "
                  "and Headroom does not.", "",
                  "| stage | applied | tokens removed |", "|---|--:|--:|"]
        for name, a in agg.items():
            lines.append(f"| {name} | {a['applied']}/{len(tool_rows)} | {a['before'] - a['after']:,} |")
        lines.append("")

    if live:
        lo = live["llmtrim_output"]
        bo = live["baseline_output"]
        lines += ["## Live output A/B (gpt-oss-20b)", "",
                  f"Original vs llmtrim-compressed request through `{MODEL}` "
                  f"(n={live['n']}). Headroom is input-only, so it has no output column.", "",
                  "| arm | output tokens | output saved |", "|---|--:|--:|",
                  f"| original | {bo:,} | n/a |",
                  f"| llmtrim | {lo:,} | **{pct(bo, lo):.0f}%** |", ""]

    lines += [
        "## Method notes", "",
        "- Latency is the median compress call, with a warm-up first so neither library is "
        "charged for one-time setup. llmtrim must run on the **release** wheel "
        "(`build-wheel.sh --release`); the debug build is several times slower and not "
        "representative.",
        "- Headroom's `compress` runs a **local ModernBERT encoder** "
        "(`answerdotai/ModernBERT-base`, the multi-hundred-MB model it downloads on first "
        "use) for its semantic smart-crusher. It makes no generative LLM API call; verified "
        "by running compress with all network sockets blocked. llmtrim is purely algorithmic "
        "(BPE counting plus deterministic stages), which is why its warm latency is lower "
        "despite removing more tokens.",
        "- Headroom protects user and system messages, so on the `general` corpora (natural "
        "request shapes, no tool results) it mostly no-ops; the `tool-output` group is its "
        "home turf.",
        "- Output tokens are out of this head-to-head. Headroom is input-only, and llmtrim's "
        "output shaping is a preset feature (e.g. `aggressive`, `reasoning`) measured in the "
        "main benchmark on a non-reasoning model. The `--live` arm exists for spot checks, "
        "but gpt-oss-20b bills hidden reasoning as output, so it is not a fair output "
        "denominator (see the main bench README).",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="llmtrim vs Headroom benchmark")
    ap.add_argument("--preset", default=LLMTRIM_PRESET_DEFAULT)
    ap.add_argument("--limit", type=int, default=12, help="cases per general corpus")
    ap.add_argument("--repeats", type=int, default=5, help="latency samples per case (median)")
    ap.add_argument("--live", action="store_true", help="run the gpt-oss-20b output A/B")
    ap.add_argument("--live-n", type=int, default=12, help="cases for the live A/B")
    args = ap.parse_args()

    enc = get_encoder()
    hr = make_headroom_client()
    # Warm both libraries on a valid tool-call turn so neither is charged for one-time
    # setup (Headroom's model load, llmtrim's first-call init) on its first real case.
    warm = [
        {"role": "user", "content": "warm-up"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_w", "type": "function",
                         "function": {"name": "fetch", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_w",
         "content": json.dumps({"results": [{"x": i} for i in range(50)]})},
    ]
    llmtrim_compress(warm, args.preset, 1)
    if hr is not None:
        try:
            hr(warm, model=BODY_MODEL)
        except Exception as e:  # noqa: BLE001
            print(f"headroom warm-up failed: {e}", file=sys.stderr)

    cases = [("tool-output", n, m) for n, m in synthetic_tool_cases()]
    cases += [("general", n, m) for n, m in general_cases(args.limit)]

    out_cases = []
    for group, name, messages in cases:
        lt_msgs, stages, lt_ms = llmtrim_compress(messages, args.preset, args.repeats)
        rec = {
            "group": group, "name": name,
            "llmtrim": {"before": count(enc, messages), "after": count(enc, lt_msgs),
                        "ms": lt_ms, "stages": stages},
            "headroom": None,
        }
        if hr is not None:
            hr_msgs, transforms, hr_ms = headroom_compress(hr, messages, args.repeats)
            rec["headroom"] = {"before": count(enc, messages), "after": count(enc, hr_msgs),
                               "ms": hr_ms, "transforms": transforms}
        out_cases.append(rec)
        print(f"  {group:11} {name:16} llmtrim {pct(rec['llmtrim']['before'], rec['llmtrim']['after']):4.0f}% "
              + (f"headroom {pct(rec['headroom']['before'], rec['headroom']['after']):4.0f}%" if rec["headroom"] else "headroom n/a"))

    live = None
    if args.live:
        key = load_api_key()
        if not key:
            print("ERROR: --live needs OPENROUTER_API_KEY (env or .env)", file=sys.stderr)
            sys.exit(1)

        bo = lo = 0
        n = 0
        for group, name, messages in cases[: args.live_n]:
            comp_msgs, _, _ = llmtrim_compress(messages, args.preset, 1)
            base = call_model(key, messages)
            trim = call_model(key, comp_msgs)
            bt, lt = completion_tokens(base), completion_tokens(trim)
            if bt is None or lt is None:
                print(f"  live skip {name} (no usage)", file=sys.stderr)
                continue
            bo += bt
            lo += lt
            n += 1
            print(f"  live {name:16} original {bt:5} → llmtrim {lt:5}")
            time.sleep(1)
        live = {"n": n, "baseline_output": bo, "llmtrim_output": lo}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {"model": MODEL, "preset": args.preset, "repeats": args.repeats,
                 "encoder": "o200k_base", "headroom": hr is not None},
        "cases": out_cases,
        "live": live,
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    report = render(results, live)
    (RESULTS_DIR / "README.md").write_text(report + "\n")
    print(f"\nWrote {RESULTS_DIR}/results.json and README.md\n")
    print(report)


if __name__ == "__main__":
    main()
