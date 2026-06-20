#!/usr/bin/env python3
"""Shared primitives for the benchkit benchmark engine.

Every library is driven through its Python API over the same o200k_base encoder and the
same message span. This module holds only the competitor-agnostic pieces - the encoder, the
llmtrim driver, the deterministic scorers, and the OpenRouter client. Competitor adapters
(see `benchkit.competitors`) bring their own compress wrapper; the engine and its reporting
live in `benchkit.sweep` / `benchkit.live` / `benchkit.report`.

Fairness rules these primitives enforce:
- Both tools' before/after token counts use the SAME encoder (`o200k_base`) over the SAME
  span (the concatenated message contents), not each library's own internal metric.
- Both libraries see the SAME messages for each case.
"""
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

# Package layout: this file is scripts/benchkit/lib.py, so parents[3] is crates/llmtrim-cli.
CRATE_ROOT = Path(__file__).resolve().parents[3]  # crates/llmtrim-cli (bench/ lives here)
WORKSPACE_ROOT = CRATE_ROOT.parents[1]  # the repo root, where .env sits
DATA_DIR = CRATE_ROOT / "bench" / "data"
RESULTS_DIR = CRATE_ROOT / "bench" / "snapshots" / "vs-headroom"

# The model + route the project pins for every OpenRouter call (see CLAUDE.md).
MODEL = "openai/gpt-oss-20b"
PROVIDER_ROUTE = {"order": ["wandb/fp4"], "allow_fallbacks": False}

# The `model` field in the request body handed to each library's local compress(). NOT an
# API call: both libraries read it to pick a tokenizer. An OpenAI id makes llmtrim select its
# exact `o200k_base` tokenizer, the same encoder this script scores both tools with.
BODY_MODEL = "gpt-4o"

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


# ── The library under test (llmtrim) ──────────────────────────────────────────
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
