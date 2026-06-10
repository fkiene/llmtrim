#!/usr/bin/env python3
"""Head-to-head: llmtrim vs Headroom on Headroom's own content types.

Generates the content types from Headroom's published benchmark (JSON arrays, shell
output, build logs, grep results, source) and compresses each with **both** tools,
reporting tokens saved + latency side by side — on the SAME token denominator.

Fairness rules (so the comparison is apples-to-apples):
- Both tools' before/after token counts use the **same encoder** (`o200k_base`) over the
  **same span** (the user-message content), not each tool's own internal metric.
- llmtrim is timed by running the **prebuilt release binary** `target/release/llmtrim`
  (NOT `cargo run`, which times a debug build + compile). Build it first:
  `cargo build --release`.
- Headroom runs only if importable (`pip install "headroom-ai[all]"`). Skipped otherwise —
  fill from its published /docs/benchmarks table.

Usage: cargo build --release && python3 bench/scripts/vs_headroom.py
"""
import json, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLMTRIM_BIN = ROOT / "target" / "release" / "llmtrim"


def encoder():
    """The shared encoder for BOTH tools (Headroom uses o200k_base in its docs)."""
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


def user_content(messages_or_obj):
    """Extract the user-message content span we tokenize for both tools."""
    msgs = messages_or_obj
    if isinstance(msgs, dict):
        msgs = msgs.get("messages", [])
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content", "")
                return c if isinstance(c, str) else json.dumps(c)
        # fall back to the last message's content
        if msgs and isinstance(msgs[-1], dict):
            c = msgs[-1].get("content", "")
            return c if isinstance(c, str) else json.dumps(c)
    return str(messages_or_obj)


def gen():
    """The six content types, sized to mirror Headroom's benchmark table."""
    def recs(n):
        a = []
        for i in range(n):
            r = {"ts": f"2026-04-{i%28+1:02d}T10:{i%60:02d}:00Z", "level": "INFO",
                 "service": f"svc-{i%5}", "msg": f"request {i} handled ok", "code": 200, "ms": i % 50}
            if i == 67:
                r.update({"level": "ERROR", "msg": "payment gateway declined",
                          "code": 402, "resolution": "retry with backup PSP", "affected": 1432})
            a.append(r)
        return json.dumps(a)
    return {
        "json_100": recs(100),
        "json_500": recs(500),
        "shell_200": "\n".join(f"drwxr-xr-x 2 u g {1024*i:8d} Apr {i%28+1:02d} file_{i}.txt" for i in range(200)),
        "buildlog_200": "\n".join(f"[{i:03d}] INFO compiling crate mod_{i} ok" for i in range(198))
                        + "\nERROR undefined reference to render_frame\nERROR build failed",
        "grep_150": "\n".join(f"src/{'abcde'[i%5]}.rs:{i+1}:    let v = connect({i});" for i in range(150)),
    }


def llmtrim_savings(enc, content):
    """Compress one case with the prebuilt release binary, timing it, and tokenize the
    user-content span with the shared encoder → (before, after, pct, ms). The binary's
    `compress` reads a request body on stdin and writes the compressed body to stdout."""
    req = json.dumps({"model": "gpt-4o",
                      "messages": [{"role": "user", "content": content}],
                      "max_tokens": 300})
    t = time.perf_counter()
    proc = subprocess.run([str(LLMTRIM_BIN), "compress", "--provider", "openai"],
                          input=req, cwd=ROOT, capture_output=True, text=True)
    ms = (time.perf_counter() - t) * 1000
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        out_obj = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    before = len(enc.encode(content))
    after = len(enc.encode(user_content(out_obj)))
    pct = 100 * (1 - after / before) if before else 0.0
    return before, after, pct, ms


def headroom_savings(enc, content):
    """Run Headroom if importable → (before, after, pct, ms) on the SAME encoder/span, else None."""
    try:
        from headroom import compress  # type: ignore
    except Exception:
        return None
    msgs = [{"role": "user", "content": content}]
    t = time.perf_counter()
    out = compress(msgs)
    ms = (time.perf_counter() - t) * 1000
    before = len(enc.encode(content))
    after = len(enc.encode(user_content(out)))
    pct = 100 * (1 - after / before) if before else 0.0
    return before, after, pct, ms


def main():
    if not LLMTRIM_BIN.exists():
        print(f"missing {LLMTRIM_BIN} — build it first: cargo build --release")
        return
    enc = encoder()
    rows = []
    for name, content in gen().items():
        lt = llmtrim_savings(enc, content)
        hr = headroom_savings(enc, content)
        rows.append((name, lt, hr))
    # Same `before` denominator for both (computed from the identical span/encoder above).
    print(f"{'content':14} {'tok':>7} {'llmtrim':>14} {'headroom':>14}")
    for name, lt, hr in rows:
        before = (lt or hr or (0, 0, 0.0, 0.0))[0]
        lt_s = f"{lt[2]:.0f}% {lt[3]:.0f}ms" if lt else "(failed)"
        hr_s = f"{hr[2]:.0f}% {hr[3]:.0f}ms" if hr else "(not installed)"
        print(f"{name:14} {before:>7} {lt_s:>14} {hr_s:>14}")
    if not any(r[2] for r in rows):
        print("\nHeadroom not installed — compare against its published /docs/benchmarks table.")
    print("\nNote: both columns use o200k_base over the user-content span; llmtrim is the "
          "prebuilt release binary. Run only verified for syntax here — needs the built binary "
          "(and optionally headroom) to produce numbers.")


if __name__ == "__main__":
    main()
