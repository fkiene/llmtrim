#!/usr/bin/env python3
"""Caveman adapter (self-contained).

Caveman is NOT a `compress(messages)` library, so it does not fit the corpora x grid engine
the Competitor interface drives. It compares three SYSTEM-PROMPT strategies live - baseline
(no system prompt), caveman (its full SKILL.md as a system prompt), and llmtrim (a 19-token
terse instruction) - and measures OUTPUT-token reduction over a fixed prompt set. There is no
deterministic token-reduction sweep, no per-config grid, and no corpora here.

Forcing this into config_grid()/compress() would either misrepresent caveman or silently
change its behavior, so per the refactor brief this module stays self-contained: it keeps its
own snapshot folder (snapshots/vs-caveman) and exposes `run(argv)` that the CLI dispatches to.
The logic below is the former `caveman_ab.py` verbatim, only restructured into `run()`.

A no-op `CavemanCompetitor` is registered so `bench.py caveman` resolves and so the registry
lists it; the CLI routes "caveman" to `run()` before the generic engine, never to the engine.
"""
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import register
from .base import Competitor

# Allow self-signed / missing CA chains (common on dev Linux boxes)
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# ── Paths ────────────────────────────────────────────────────────────────────
# This file is scripts/benchkit/competitors/caveman.py, so parents[4] is crates/llmtrim-cli -
# the same absolute path the former scripts/caveman_ab.py resolved as REPO_ROOT (parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[4]
CAVEMAN_ROOT = REPO_ROOT.parent / "caveman"
RESULTS_DIR = REPO_ROOT / "bench" / "snapshots" / "vs-caveman"

PROMPTS_FILE = CAVEMAN_ROOT / "benchmarks" / "prompts.json"
SKILL_FILE = CAVEMAN_ROOT / "skills" / "caveman" / "SKILL.md"
TERSE_FILE = REPO_ROOT / "prompts" / "output_terse.txt"

# ── Config ───────────────────────────────────────────────────────────────────
MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0
MAX_TOKENS = 2048


@register
class CavemanCompetitor(Competitor):
    """Registry stub: caveman is dispatched to run() by the CLI, not to the engine. The
    Competitor methods raise to make a wrong dispatch loud rather than silently fabricate a
    grid/compress() that caveman does not have."""
    name = "caveman"
    display = "caveman"

    def compress(self, messages, cfg, repeats):
        raise NotImplementedError("caveman is self-contained; the CLI dispatches to run()")

    def config_grid(self):
        raise NotImplementedError("caveman is self-contained; the CLI dispatches to run()")

    def ml_fired(self, transforms):
        return False

    def notes(self):
        return {}


# Load API key from env or .env file
def load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val:
                    return val
    return None


# ── Load fixtures ─────────────────────────────────────────────────────────────
def load_prompts():
    with open(PROMPTS_FILE) as f:
        data = json.load(f)
    return data["prompts"]


def load_caveman_system():
    """Strip YAML front-matter from SKILL.md and return the body."""
    text = SKILL_FILE.read_text()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()


def load_terse_instruction():
    return TERSE_FILE.read_text().strip()


# ── API call ──────────────────────────────────────────────────────────────────
def call_openrouter(api_key, messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fkiene/llmtrim",
            "X-Title": "llmtrim-ab-bench",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
        return json.loads(resp.read())


def call_with_retry(api_key, messages, label):
    for attempt in range(2):
        try:
            resp = call_openrouter(api_key, messages)
            return resp
        except Exception as e:
            print(f"  attempt {attempt+1} failed for {label}: {e}", file=sys.stderr)
            if attempt == 0:
                time.sleep(5)
    return None


# ── Build messages per arm ────────────────────────────────────────────────────
def make_messages(arm, prompt_text, caveman_system, terse_instruction):
    if arm == "baseline":
        return [{"role": "user", "content": prompt_text}]
    elif arm == "caveman":
        return [
            {"role": "system", "content": caveman_system},
            {"role": "user", "content": prompt_text},
        ]
    elif arm == "llmtrim":
        return [
            {"role": "system", "content": terse_instruction},
            {"role": "user", "content": prompt_text},
        ]
    else:
        raise ValueError(f"Unknown arm: {arm}")


# ── Extract usage ─────────────────────────────────────────────────────────────
def extract_record(prompt_id, arm, resp):
    usage = resp.get("usage", {})
    text = ""
    choices = resp.get("choices", [])
    if choices:
        text = choices[0].get("message", {}).get("content", "")
    return {
        "prompt_id": prompt_id,
        "arm": arm,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "response": text,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key()
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment or .env", file=sys.stderr)
        sys.exit(1)

    prompts = load_prompts()
    caveman_system = load_caveman_system()
    terse_instruction = load_terse_instruction()

    print(f"Caveman system prompt length: {len(caveman_system.split())} words")
    print(f"Terse instruction: {repr(terse_instruction)}")
    print(f"Running {len(prompts)} prompts x 3 arms = {len(prompts)*3} calls")
    print()

    arms = ["baseline", "caveman", "llmtrim"]
    all_records = []
    failed_prompts = []

    for p in prompts:
        pid = p["id"]
        prompt_text = p["prompt"]
        print(f"Prompt: {pid}")
        arm_records = {}
        ok = True

        for arm in arms:
            messages = make_messages(arm, prompt_text, caveman_system, terse_instruction)
            label = f"{pid}/{arm}"
            print(f"  calling {arm}...", end=" ", flush=True)
            resp = call_with_retry(api_key, messages, label)
            if resp is None:
                print("FAILED")
                ok = False
                break
            rec = extract_record(pid, arm, resp)
            print(f"completion_tokens={rec['completion_tokens']}")
            arm_records[arm] = rec
            time.sleep(1)  # gentle rate limiting

        if ok and len(arm_records) == 3:
            for arm in arms:
                all_records.append(arm_records[arm])
        else:
            failed_prompts.append(pid)
            print(f"  *** {pid} excluded from totals (incomplete) ***")

    # Save raw results
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved {len(all_records)} records to {results_path}")

    # Compute summary
    write_summary(all_records, failed_prompts, caveman_system, terse_instruction)


def write_summary(records, failed_prompts, caveman_system, terse_instruction):
    from collections import defaultdict

    # Group by arm; only include prompts where all 3 arms are present
    by_prompt_arm = defaultdict(dict)
    for r in records:
        by_prompt_arm[r["prompt_id"]][r["arm"]] = r

    valid_prompts = [pid for pid, arms in by_prompt_arm.items() if len(arms) == 3]

    totals = {"baseline": {"prompt_tokens": 0, "completion_tokens": 0},
              "caveman":  {"prompt_tokens": 0, "completion_tokens": 0},
              "llmtrim":  {"prompt_tokens": 0, "completion_tokens": 0}}

    per_prompt_notes = []

    for pid in valid_prompts:
        arms = by_prompt_arm[pid]
        for arm in ["baseline", "caveman", "llmtrim"]:
            totals[arm]["prompt_tokens"]     += arms[arm]["prompt_tokens"] or 0
            totals[arm]["completion_tokens"] += arms[arm]["completion_tokens"] or 0

        # Simple quality judgment: does each compressed answer mention the same key terms as baseline?
        baseline_words = set((arms["baseline"]["response"] or "").lower().split())
        notes = []
        for arm in ["caveman", "llmtrim"]:
            resp_text = arms[arm]["response"] or ""
            resp_words = set(resp_text.lower().split())
            # Sample a few key technical terms from baseline (words > 5 chars, not common English)
            common = {"which", "where", "there", "their", "about", "would", "could", "should",
                      "using", "makes", "every", "after", "before", "first", "when", "then",
                      "that", "this", "with", "have", "from", "into", "will", "your"}
            technical_terms = [w for w in baseline_words if len(w) > 5 and w.isalpha() and w not in common][:15]
            overlap = len([t for t in technical_terms if t in resp_words])
            ratio = overlap / len(technical_terms) if technical_terms else 1.0
            quality = "OK" if ratio >= 0.5 else "DEGRADED"
            notes.append(f"{arm}={quality}({overlap}/{len(technical_terms)} key terms)")
        per_prompt_notes.append((pid, notes))

    n = len(valid_prompts)
    base_comp = totals["baseline"]["completion_tokens"]
    cav_comp  = totals["caveman"]["completion_tokens"]
    ltrim_comp = totals["llmtrim"]["completion_tokens"]

    def pct_reduction(comp, base):
        if base == 0:
            return "n/a"
        return f"{(1 - comp/base)*100:.1f}%"

    caveman_overhead = len(caveman_system.split())  # rough word count used as proxy
    terse_overhead = len(terse_instruction.split())

    readme = f"""# Caveman vs llmtrim A/B Benchmark

Model: `{MODEL}` | temperature=0 | max_tokens={MAX_TOKENS}
Prompts: {n} valid (of 10) | {len(failed_prompts)} failed: {failed_prompts if failed_prompts else "none"}

## Summary Table

| Arm       | Total prompt tokens | Total completion tokens | Output reduction vs baseline | Instr. overhead/req |
|-----------|---------------------|------------------------|------------------------------|----------------------|
| baseline  | {totals['baseline']['prompt_tokens']:>19} | {totals['baseline']['completion_tokens']:>22} | -                            | 0 tokens             |
| caveman   | {totals['caveman']['prompt_tokens']:>19} | {totals['caveman']['completion_tokens']:>22} | {pct_reduction(cav_comp, base_comp):>28} | ~949 tokens (SKILL.md)|
| llmtrim   | {totals['llmtrim']['prompt_tokens']:>19} | {totals['llmtrim']['completion_tokens']:>22} | {pct_reduction(ltrim_comp, base_comp):>28} | 19 tokens (terse.txt)|

Numbers are sum across {n} prompts where all 3 arms succeeded.

## Per-Prompt Quality Check

Quality = compressed answer retains ≥50% of key technical terms from baseline.

| Prompt ID             | caveman quality | llmtrim quality |
|-----------------------|-----------------|-----------------|
"""
    for pid, notes in per_prompt_notes:
        cav_note = next((n for n in notes if n.startswith("caveman")), "-")
        ltrim_note = next((n for n in notes if n.startswith("llmtrim")), "-")
        readme += f"| {pid:<21} | {cav_note:<15} | {ltrim_note:<15} |\n"

    readme += f"""
## Key Numbers

- Baseline total output tokens: {base_comp}
- Caveman total output tokens:  {cav_comp} ({pct_reduction(cav_comp, base_comp)} reduction)
- llmtrim total output tokens:  {ltrim_comp} ({pct_reduction(ltrim_comp, base_comp)} reduction)

### Instruction overhead per request
- caveman: ~949 tokens (full SKILL.md body as system prompt)
- llmtrim: 19 tokens (`output_terse.txt`)

Net efficiency = output savings minus instruction overhead amortized over session length.

## Caveats
- Quality judgment uses keyword overlap (≥50% of key technical terms), not semantic evaluation.
- "Key technical terms" = words >5 chars, alphabetic, non-common-English, sampled from baseline (up to 15).
- Token counts are from OpenRouter's `usage` fields (real API counts, not estimates).
- Model: {MODEL} may apply its own brevity tendencies independent of the system prompt.
"""

    readme_path = RESULTS_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)
    print(f"Saved summary to {readme_path}")
    print()
    print(readme)


def summarize_only():
    """Re-generate README from existing results.json without calling the API."""
    caveman_system = load_caveman_system()
    terse_instruction = load_terse_instruction()
    results_path = RESULTS_DIR / "results.json"
    with open(results_path) as f:
        records = json.load(f)
    write_summary(records, [], caveman_system, terse_instruction)


def run(argv):
    """Entry point the CLI dispatches to for `bench.py caveman [--summarize]`."""
    if "--summarize" in argv:
        summarize_only()
    else:
        main()
    return 0
