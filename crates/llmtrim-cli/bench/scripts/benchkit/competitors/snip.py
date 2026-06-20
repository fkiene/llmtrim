#!/usr/bin/env python3
"""snip adapter (self-contained).

snip (github.com/edouard-claude/snip) is a Go CLI proxy in the same family as RTK: it sits
between an AI coding tool and the shell, runs a command, and filters that command's raw output
through a matched YAML pipeline before the bytes reach the model's context. The reduction is
LOCAL and deterministic. There is no LLM, no network call, and no API key in its filter path
(re-confirmed: internal/engine, internal/filter, and internal/config import neither net nor
net/http, and the tree carries no api-key or URL constant), so snip is not an LLM-based
"shrink your bill" proxy.

snip filters command OUTPUT, not message arrays, so it does not fit the corpora x grid engine
the Competitor interface drives. The benchkit corpora are prose QA/summarization (gsm8k,
hotpotqa, cnn, longbench); snip's filters target tool output and have nothing to bite on
there, so routing it through the generic engine would report a flat 0% that misrepresents the
tool. The honest fit is tool output, so this adapter measures snip in its BEST mode: each
native filter running on the content type it was built for.

How it is driven faithfully. snip has no stdin pipe mode; its real interface is
`snip run -- <command>`, which runs the command and filters its stdout. The selected fixtures
are snip's own tests/fixtures/*_raw.txt for filters that do NOT inject args (cargo-test, ls,
rspec, rails db:migrate, rails routes, bundle install). For a non-injecting filter the
pipeline runs on exactly the command's raw output, so replaying that captured output through
`snip run --` (via a one-line shim that emits the fixture) reproduces snip's live behavior
byte for byte. Injecting filters (git-log, git-diff, git-status, go-test) are left out on
purpose: snip re-runs the command with extra flags (--pretty=format:..., -json, ...) so the
pipeline sees a different, compact stream a static raw fixture cannot reproduce. snip's own
integration test makes the same distinction (it feeds a "simulated post-injection output" for
those), so measuring an injecting filter on a raw fixture would distort the result. Token
counts use the same o200k_base encoder the rest of the benchmark scores with.

snip is lossy by design: cargo-test keeps the pass/fail tallies and drops the per-test detail,
rails routes reports a count instead of the table. So the reduction reported here is the token
win, not a quality-matched score; a downstream task that needs the dropped detail loses it.

Like caveman and rtk, this module stays self-contained: it keeps its own snapshot folder
(snapshots/vs-snip) and exposes run(argv) that the CLI dispatches to. A no-op SnipCompetitor
is registered so `bench.py snip` resolves and the registry lists it; its Competitor methods
raise so a wrong dispatch to the generic engine is loud rather than a fabricated grid.
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from . import register
from .base import Competitor

# This file is scripts/benchkit/competitors/snip.py, so parents[4] is crates/llmtrim-cli.
CRATE_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = Path(__file__).resolve().parent / "snip_fixtures"
RESULTS_DIR = CRATE_ROOT / "bench" / "snapshots" / "vs-snip"

# snip BEST mode: each entry is (filter, fixture file, argv that matches the filter). Every
# filter here is non-injecting, so snip's pipeline runs on exactly the fixture bytes. The argv
# command name is what snip matches on; the shim ignores the rest of the args. Order is fixed
# so a --limit run is stable and predictable, not sorted by size (which would let a cap drop
# the less-flattering cases).
CASES = [
    ("cargo-test", "cargo-test.txt", ["cargo", "test"]),
    ("rspec", "rspec.txt", ["rspec"]),
    ("rails-routes", "rails-routes.txt", ["rails", "routes"]),
    ("rails-migrate", "rails-migrate.txt", ["rails", "db:migrate"]),
    ("bundle-install", "bundle-install.txt", ["bundle", "install"]),
    ("ls", "ls.txt", ["ls", "-la", "-R"]),
]


@register
class SnipCompetitor(Competitor):
    """Registry stub: snip is dispatched to run() by the CLI, not to the engine. The Competitor
    methods raise so a wrong dispatch is loud rather than silently fabricating a grid/compress()
    that snip (a tool-output filter, not a message compressor) does not have."""
    name = "snip"
    display = "snip"

    def compress(self, messages, cfg, repeats):
        raise NotImplementedError("snip is self-contained; the CLI dispatches to run()")

    def config_grid(self):
        raise NotImplementedError("snip is self-contained; the CLI dispatches to run()")

    def ml_fired(self, transforms):
        return False

    def notes(self):
        return {}


def find_snip():
    """Locate the snip binary on PATH, in ~/.local/bin (its installer dir), in ~/go/bin (go
    install), or as the built /tmp/snip-bin used during integration."""
    exe = shutil.which("snip")
    if exe:
        return exe
    for cand in (Path.home() / ".local" / "bin" / "snip",
                 Path.home() / "go" / "bin" / "snip",
                 Path("/tmp/snip-bin")):
        if cand.exists():
            return str(cand)
    return None


def load_cases(limit):
    """Return [(filter, text, argv)] for each CASES entry whose fixture exists, capped at
    limit (0 = all). Order follows CASES so a capped run reports the first N filters."""
    out = []
    for flt, fixture, argv in CASES:
        path = FIXTURE_DIR / fixture
        if not path.exists():
            continue
        text = path.read_text()
        if text.strip():
            out.append((flt, text, argv))
    return out[:limit] if limit else out


def snip_filter(exe, fixture_path, argv, home):
    """Run `snip run -- <argv>` with a shim that emits the fixture, returning snip's filtered
    stdout. Local, deterministic, no network. The shim is the command snip matches (argv[0]),
    so snip runs it, captures its stdout (the fixture), and applies the matched pipeline."""
    with tempfile.TemporaryDirectory() as shim_dir:
        shim = Path(shim_dir) / argv[0]
        shim.write_text(f'#!/bin/sh\ncat {json.dumps(str(fixture_path))}\n')
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        env = dict(os.environ)
        env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
        env["HOME"] = home  # isolate snip's config/tracking from the user's
        proc = subprocess.run(
            [exe, "run", "--", *argv],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"snip run -- {' '.join(argv)} exit {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def run(argv):
    """Entry point the CLI dispatches to for `bench.py snip [--limit N]`."""
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
        if limit < 0:
            print(f"--limit must be >= 0, got {limit}", file=sys.stderr)
            return 1

    exe = find_snip()
    if not exe:
        print("snip not installed. Build it: install Go 1.25+, then "
              "`go install github.com/edouard-claude/snip/cmd/snip@latest` "
              "(binary lands in ~/go/bin or ~/.local/bin).", file=sys.stderr)
        return 1

    from .. import lib
    enc = lib.get_encoder()

    cases = load_cases(limit)
    if not cases:
        print(f"no snip fixtures found under {FIXTURE_DIR}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as home:
        records = []
        tot_in = tot_out = 0
        for flt, text, argv_cmd in cases:
            out = snip_filter(exe, FIXTURE_DIR / f"{flt}.txt", argv_cmd, home)
            in_tok = len(enc.encode(text))
            out_tok = len(enc.encode(out))
            tot_in += in_tok
            tot_out += out_tok
            red = (1 - out_tok / in_tok) * 100 if in_tok else 0.0
            records.append({"filter": flt, "in_tokens": in_tok, "out_tokens": out_tok,
                            "reduction_pct": round(red, 1)})
            print(f"  filter={flt:14} {in_tok:6} -> {out_tok:6} tok  ({red:5.1f}%)",
                  file=sys.stderr)

    overall = (1 - tot_out / tot_in) * 100 if tot_in else 0.0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    version = raw_version.removeprefix("snip").strip() or raw_version
    results = {
        "meta": {
            "tool": "snip", "version": version, "binary": exe,
            "encoder": "o200k_base", "mode": "BEST (native filter per tool-output type)",
            "fixtures": ("competitors/snip_fixtures/*.txt (snip's own tests/fixtures, "
                         "non-injecting filters only, same o200k span)"),
            "scope": ("self-contained: snip filters tool output, not message arrays, so it "
                      "does not run on the prose corpora the generic engine uses"),
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
        "# snip vs llmtrim (tool-output mode)",
        "",
        f"Tool: snip {m['version']} | encoder: `{m['encoder']}` | mode: {m['mode']}",
        "",
        "snip filters shell-command OUTPUT, not message arrays, so it is measured on real "
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
        "- snip is lossy by design: cargo-test keeps the pass/fail tallies and drops the "
        "per-test detail, rails routes reports a count instead of the table. Reduction here is "
        "the token win, not a quality-matched comparison.",
        "- Only non-injecting filters are measured. snip's injecting filters (git-log, "
        "git-diff, git-status, go-test) re-run the command with extra flags, so the pipeline "
        "sees a different stream than a static raw fixture; measuring them on raw output would "
        "distort the result, so they are out of scope rather than faked.",
        "- The total is input-token-weighted, so the largest fixture dominates it. --limit "
        "keeps the CASES order, so a capped run reports the first N filters, not the largest.",
        "- Scope is tool output only. snip has no filter for prose, file contents, or "
        "web-search results, so the prose corpora are out of scope rather than counted as 0%.",
        "- All numbers are local and deterministic (snip run --, no network).",
    ]
    (RESULTS_DIR / "README.md").write_text("\n".join(lines) + "\n")
