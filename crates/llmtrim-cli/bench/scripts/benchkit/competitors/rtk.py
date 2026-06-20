#!/usr/bin/env python3
"""RTK adapter (self-contained).

RTK (github.com/rtk-ai/rtk) is a local CLI proxy that rewrites the OUTPUT of shell commands
(test runners, grep, git, logs, ...) before it reaches an LLM. It is NOT a `compress(messages)`
library, so it does not fit the corpora x grid engine the Competitor interface drives. The
benchkit corpora are prose QA/summarization (gsm8k, hotpotqa, cnn, longbench); RTK's filters
target tool output and have nothing to bite on there, so routing it through the generic engine
would report a flat 0% that misrepresents the tool.

The honest fit is tool output. RTK ships `rtk pipe --filter <name>`, which reads arbitrary text
on stdin and emits the filtered version with no network call (verified: zero inet syscalls in
`strace -e trace=network`). So this adapter measures RTK in its BEST mode: each native filter
running on the content type it was built for. The fixtures are real command output captured
once into competitors/rtk_fixtures/ (see that folder's README for how), so RTK parses exactly
the text it was designed for, not a shorthand. Token counts use the same `o200k_base` encoder
the rest of the benchmark scores with.

RTK is lossy by design: the pytest filter keeps the failures and drops the per-test pass
detail, grep groups and truncates matches. So the reduction reported here is the token win,
not a quality-matched score; a downstream task that needs the dropped detail loses it.

Like caveman, this module stays self-contained: it keeps its own snapshot folder
(snapshots/vs-rtk) and exposes `run(argv)` that the CLI dispatches to. A no-op `RTKCompetitor`
is registered so `bench.py rtk` resolves and the registry lists it; its Competitor methods
raise so a wrong dispatch to the generic engine is loud rather than a fabricated grid.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import register
from .base import Competitor

# This file is scripts/benchkit/competitors/rtk.py, so parents[4] is crates/llmtrim-cli.
CRATE_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = Path(__file__).resolve().parent / "rtk_fixtures"
RESULTS_DIR = CRATE_ROOT / "bench" / "snapshots" / "vs-rtk"

# RTK BEST mode: each fixture file is real output for the RTK filter of the same name. The
# filter names are RTK's own (`rtk pipe --filter ...`; the unknown-filter error lists them all).
FILTERS = ["pytest", "grep", "git-log", "git-diff"]


@register
class RTKCompetitor(Competitor):
    """Registry stub: RTK is dispatched to run() by the CLI, not to the engine. The Competitor
    methods raise so a wrong dispatch is loud rather than silently fabricating a grid/compress()
    that RTK (a tool-output rewriter, not a message compressor) does not have."""
    name = "rtk"
    display = "RTK"

    def compress(self, messages, cfg, repeats):
        raise NotImplementedError("rtk is self-contained; the CLI dispatches to run()")

    def config_grid(self):
        raise NotImplementedError("rtk is self-contained; the CLI dispatches to run()")

    def ml_fired(self, transforms):
        return False

    def notes(self):
        return {}


def find_rtk():
    """Locate the rtk binary on PATH or in the default installer dir (~/.local/bin)."""
    exe = shutil.which("rtk")
    if exe:
        return exe
    fallback = Path.home() / ".local" / "bin" / "rtk"
    return str(fallback) if fallback.exists() else None


def load_fixtures(limit):
    """Return [(filter, text)] for each rtk_fixtures/<filter>.txt that exists, capped at
    `limit` (0 = all). Order follows FILTERS so a capped run is stable and predictable, not
    sorted by size (which would let a cap drop the less-flattering cases)."""
    cases = []
    for flt in FILTERS:
        path = FIXTURE_DIR / f"{flt}.txt"
        if not path.exists():
            continue
        text = path.read_text()
        if text.strip():
            cases.append((flt, text))
    return cases[:limit] if limit else cases


def rtk_pipe(exe, flt, text):
    """Run `rtk pipe --filter <flt>` on `text` over stdin and return the filtered output. Local,
    deterministic, no network. RTK writes a one-line nudge to stderr when no hook is installed;
    we capture stderr separately so only the filtered stdout is measured."""
    proc = subprocess.run(
        [exe, "pipe", "--filter", flt],
        input=text,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"rtk pipe --filter {flt} exit {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def run(argv):
    """Entry point the CLI dispatches to for `bench.py rtk [--limit N]`."""
    limit = 0
    if "--limit" in argv:
        i = argv.index("--limit")
        if i + 1 >= len(argv):
            print("--limit needs a value", file=sys.stderr)
            return 1
        try:
            limit = int(argv[i + 1])
        except ValueError:
            print(f"--limit must be an integer, got {argv[i + 1]!r}", file=sys.stderr)
            return 1

    exe = find_rtk()
    if not exe:
        print("RTK not installed. Install: "
              "curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/"
              "install.sh | sh   (binary lands in ~/.local/bin)", file=sys.stderr)
        return 1

    from .. import lib
    enc = lib.get_encoder()

    cases = load_fixtures(limit)
    if not cases:
        print(f"no RTK fixtures found under {FIXTURE_DIR}", file=sys.stderr)
        return 1

    records = []
    tot_in = tot_out = 0
    for flt, text in cases:
        out = rtk_pipe(exe, flt, text)
        in_tok = len(enc.encode(text))
        out_tok = len(enc.encode(out))
        tot_in += in_tok
        tot_out += out_tok
        red = (1 - out_tok / in_tok) * 100 if in_tok else 0.0
        records.append({"filter": flt, "in_tokens": in_tok, "out_tokens": out_tok,
                        "reduction_pct": round(red, 1)})
        print(f"  filter={flt:9} {in_tok:6} -> {out_tok:6} tok  ({red:5.1f}%)", file=sys.stderr)

    overall = (1 - tot_out / tot_in) * 100 if tot_in else 0.0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    version = raw_version.removeprefix("rtk").strip() or raw_version
    results = {
        "meta": {
            "tool": "rtk", "version": version, "binary": exe,
            "encoder": "o200k_base", "mode": "BEST (native filter per tool-output type)",
            "fixtures": "competitors/rtk_fixtures/*.txt (real command output, same o200k span)",
            "scope": ("self-contained: RTK rewrites tool output, not message arrays, so it does "
                      "not run on the prose corpora the generic engine uses"),
            "limit": limit,
        },
        "records": records,
        "totals": {"in_tokens": tot_in, "out_tokens": tot_out,
                   "reduction_pct": round(overall, 1)},
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    write_summary(results)
    print(f"\nWrote {RESULTS_DIR}/results.json and README.md\n", file=sys.stderr)
    return 0


def write_summary(results):
    m = results["meta"]
    t = results["totals"]
    lines = [
        "# RTK vs llmtrim (tool-output mode)",
        "",
        f"Tool: rtk {m['version']} | encoder: `{m['encoder']}` | mode: {m['mode']}",
        "",
        "RTK rewrites shell-command OUTPUT, not message arrays, so it is measured on real "
        "command output (not the prose corpora). Each native filter runs on the content type it "
        "was built for, scored on the same `o200k_base` encoder.",
        "",
        "## Per-filter reduction",
        "",
        "| Filter | In (tok) | Out (tok) | Reduction |",
        "|--------|---------:|----------:|----------:|",
    ]
    for r in results["records"]:
        lines.append(f"| {r['filter']} | {r['in_tokens']} | {r['out_tokens']} | "
                     f"{r['reduction_pct']:.1f}% |")
    lines += [
        f"| **total** | **{t['in_tokens']}** | **{t['out_tokens']}** | "
        f"**{t['reduction_pct']:.1f}%** |",
        "",
        "## Caveats",
        "",
        "- RTK is lossy by design: the pytest filter keeps the failures and drops the per-test "
        "pass detail, grep groups and truncates matches. Reduction here is the token win, not a "
        "quality-matched comparison.",
        "- A small diff can grow rather than shrink: the git-diff filter adds a header, so on a "
        "two-line diff the output is larger than the input. That negative is real, not hidden.",
        "- The total is input-token-weighted, so the largest fixture dominates it. `--limit` "
        "keeps the FILTERS order, so a capped run reports the first N filters, not the largest.",
        "- Scope is tool output only. RTK has no filter for prose, file contents, or web-search "
        "results, so the prose corpora are out of scope rather than counted as 0%.",
        "- All numbers are local and deterministic (`rtk pipe --filter`, no network).",
    ]
    (RESULTS_DIR / "README.md").write_text("\n".join(lines) + "\n")
