"""The $0 CI gate (--check), baseline writer, data integrity, and provenance."""
import hashlib
import json
import sys
import time

from .config import BASELINE, CORPORA, DATA_DIR, LLMTRIM_PRESETS


def data_integrity():
    """Every committed corpus file's sha256 matches the manifest. Catches a swapped or
    edited corpus - the cheapest credibility check there is."""
    manifest = DATA_DIR / "manifest.json"
    if not manifest.exists():
        return ["data: manifest.json missing - run `make data` (or the CI download step)"]
    man = json.loads(manifest.read_text()).get("corpora", {})
    fails = []
    for c in CORPORA:
        f = DATA_DIR / f"{c}.jsonl"
        if not f.exists():
            fails.append(f"data {c}: file missing")
            continue
        exp = man.get(c, {}).get("sha256")
        if not exp:
            fails.append(f"data {c}: not pinned in manifest")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != exp:
            fails.append(f"data {c}: sha256 != manifest")
    return fails


def write_baseline(det, limit):
    base = {
        "limit": limit,
        "tolerance_pp": 3,
        "generated": time.strftime("%Y-%m-%d"),
        "llmtrim": {p: round(det[p]["reduction_pct"], 1)
                    for p in LLMTRIM_PRESETS if p in det},
        "note": ("The CI gate (bench.py <competitor> --check) asserts llmtrim's deterministic "
                 "reduction stays within tolerance_pp of these. Regenerate intentionally "
                 "with `make baseline` when an engine change moves the numbers."),
    }
    BASELINE.write_text(json.dumps(base, indent=2) + "\n")
    print(f"wrote {BASELINE}: {base['llmtrim']}", file=sys.stderr)


def run_check(det, limit):
    """The $0 CI gate. Asserts deterministic invariants + llmtrim reduction within tolerance
    of baseline.json. The competitor is NOT gated (its ML is non-deterministic). Returns exit
    code."""
    fails = []
    safe = det.get("safe", {}).get("reduction_pct")
    auto = det.get("auto", {}).get("reduction_pct")
    agg = det.get("aggressive", {}).get("reduction_pct")
    # `safe` is lossless input; a sub-1% drift from hygiene normalization is fine, but a
    # real lossy leak would be percent-scale.
    if safe is None or abs(safe) > 1.0:
        fails.append(f"invariant: safe must be ~lossless (<1%), got {safe}")
    if auto is None or auto <= 0:
        fails.append(f"invariant: auto must compress (>0%), got {auto}")
    if agg is None or auto is None or agg < auto - 0.01:
        fails.append(f"invariant: aggressive must be >= auto, got {agg} vs {auto}")
    fails += data_integrity()
    if not BASELINE.exists():
        fails.append(f"no baseline.json - run `make baseline` (limit {limit}) and commit it")
    else:
        base = json.loads(BASELINE.read_text())
        tol = base.get("tolerance_pp", 3)
        if base.get("limit") != limit:
            fails.append(f"baseline limit {base.get('limit')} != check limit {limit}")
        for p, exp in base.get("llmtrim", {}).items():
            got = det.get(p, {}).get("reduction_pct")
            if got is None:
                fails.append(f"baseline: {p} missing from run")
            elif abs(got - exp) > tol:
                fails.append(f"regression: {p} reduction {got:.1f}% vs baseline {exp:.1f}% "
                             f"(> ±{tol}pp) - engine changed; update baseline if intended")
    if fails:
        print("BENCH CHECK FAILED:", file=sys.stderr)
        for f in fails:
            print(f"  ✗ {f}", file=sys.stderr)
        return 1
    print(f"BENCH CHECK PASSED (safe={safe:.1f}% auto={auto:.1f}% aggressive={agg:.1f}%, "
          f"data + baseline OK)", file=sys.stderr)
    return 0


def provenance(competitor):
    """Everything a third party needs to reproduce or distrust a number: tool versions, the
    competitor's git commit (when an explicit checkout is used), platform, and the data
    manifest hash. Recorded in results.json so a reviewer never has to take an aggregate on
    faith."""
    import platform
    import subprocess

    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
        try:
            from importlib.metadata import version
            return version(mod)
        except Exception:  # noqa: BLE001
            return None

    # The competitor usually runs from its pip-installed package; the version is the citable
    # identifier. A git commit is a bonus when a source checkout is present.
    hr_version = None
    try:
        from importlib.metadata import version
        hr_version = version("headroom-ai")
    except Exception:  # noqa: BLE001
        pass
    # Only record a git commit when running an EXPLICIT checkout; with the pinned PyPI build
    # the version above is the citable identifier (a dev-tree commit would be misleading).
    hr_src = getattr(competitor, "HEADROOM_SRC", None) or getattr(
        __import__("benchkit.competitors.headroom", fromlist=["HEADROOM_SRC"]),
        "HEADROOM_SRC", None)
    hr_commit = None
    for cand in ([hr_src] if hr_src else []):
        try:
            hr_commit = subprocess.check_output(
                ["git", "-C", str(cand.resolve()), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL, text=True).strip()
            if hr_commit:
                break
        except Exception:  # noqa: BLE001
            continue

    manifest = DATA_DIR / "manifest.json"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tiktoken": _v("tiktoken"),
        "llmtrim": _v("llmtrim") or _v("llmtrim_ffi"),
        "headroom_version": hr_version,
        "headroom_commit": hr_commit,
        "headroom_installed": getattr(competitor, "installed", False),
        "data_manifest_fetched": (json.loads(manifest.read_text()).get("fetched")
                                  if manifest.exists() else None),
    }
