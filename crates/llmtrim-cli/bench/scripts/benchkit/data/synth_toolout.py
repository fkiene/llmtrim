#!/usr/bin/env python3
"""Synthesize a tool-output benchmark corpus for the toolout stage.

Each case puts a realistic log / diff / grep blob in `context` (assembled into a user
message by the bench loader) and asks a question whose answer lives in a line the
toolout stage *keeps* (an error, a changed/`+`/`-` line, or a query-relevant match).
A faithful compressor should let the model answer from the windowed output just as well
as from the full blob - that's the quality axis; the token delta is the savings axis.

Writes bench/data/toolout.jsonl. Deterministic (no randomness), so re-running is a no-op.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "data" / "toolout.jsonl"

# A pool of distinct words so "noise" lines vary lexically (exercise windowing, not just
# template collapse).
WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee "
    "zulu amber blaze cobalt dune ember flint granite harbor ivory jade kelp lotus "
    "maple nectar opal pearl quartz rust slate topaz umber violet willow"
).split()


def noise(prefix, n):
    return [f"{prefix} {WORDS[i % len(WORDS)]}-{i}" for i in range(n)]


def case(name, context, question, gold, adversarial=False):
    c = {"name": name, "context": context, "question": question, "gold": gold,
         "scorer": "contains"}
    if adversarial:
        # Tag (ignored by the loader) so a reader can see these are the windowing-stress cases
        # whose gold sits in DROPPED lines - retention SHOULD fall if windowing is too greedy.
        c["adversarial"] = True
    return c


cases = []

# ---- logs: the answer is in a force-kept ERROR/FAILED line --------------------------
build = "\n".join(
    [f"INFO  [{i:03d}] compiling crate {WORDS[i % len(WORDS)]}_core" for i in range(80)]
)
build += "\nERROR linker failed: undefined reference to `render_frame`\n"
build += "INFO  build finished with 1 error"
cases.append(case("log-build-undef",
                  build,
                  "What symbol is reported as an undefined reference in the build error?",
                  "render_frame"))

pytest = "\n".join(
    [f"tests/test_{WORDS[i % len(WORDS)]}.py::test_case_{i} PASSED" for i in range(90)]
)
pytest += ("\ntests/test_auth.py::test_login FAILED - AssertionError: "
           "expected status 200 but got 401")
pytest += "\n90 passed, 1 failed in 12.40s"
cases.append(case("log-pytest-fail",
                  pytest,
                  "Which test FAILED in the pytest run?",
                  "test_login"))

timeout = "\n".join(noise("DEBUG worker idle", 60))
timeout += "\nERROR connection to host db-7 timed out after 30000ms\n"
timeout += "\n".join(noise("DEBUG worker idle", 20))
cases.append(case("log-db-timeout",
                  timeout,
                  "Which database host timed out?",
                  "db-7"))

# templatable noise (folds losslessly), error kept
ratelog = "\n".join([f"INFO request {1000+i} served in {5+i%40}ms" for i in range(100)])
ratelog += "\nERROR rate limit exceeded for tenant acme-corp"
cases.append(case("log-rate-limit",
                  ratelog,
                  "Which tenant exceeded the rate limit?",
                  "acme-corp"))

panic = "\n".join(noise("INFO  handled request", 50))
panic += "\nthread 'main' panicked at src/engine.rs:88: index out of bounds: len is 3"
panic += "\nnote: run with RUST_BACKTRACE=1"
cases.append(case("log-panic-file",
                  panic,
                  "In which source file did the thread panic?",
                  "src/engine.rs"))

# ---- diffs: the answer is in a kept changed (+/-) line ------------------------------
def diff_file(path, old, new, ctx_lines):
    head = (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -1,{len(ctx_lines)+1} +1,{len(ctx_lines)+1} @@\n")
    body = "\n".join(f" {c}" for c in ctx_lines[:5])
    body += f"\n-{old}\n+{new}\n"
    body += "\n".join(f" {c}" for c in ctx_lines[5:])
    return head + body


ctx = [f"// surrounding line {WORDS[i]}" for i in range(20)]
sig = diff_file("src/auth.rs",
                "fn login(user: &User) -> Result<Session>",
                "fn login(user: &User, mfa_token: &str) -> Result<Session>",
                ctx)
cases.append(case("diff-signature",
                  sig,
                  "What new parameter was added to the login function signature?",
                  "mfa_token"))

deletion = diff_file("src/token.rs",
                     "pub fn validate_token(t: &str) -> bool { check(t) }",
                     "// token validation moved to middleware",
                     ctx)
cases.append(case("diff-deletion",
                  deletion,
                  "Which function was removed in the diff to src/token.rs?",
                  "validate_token"))

# multi-file diff: only one file carries the meaningful change; others are noise
multi = ""
for i in range(8):
    multi += diff_file(f"src/mod_{WORDS[i]}.rs", f"const N = {i}", f"const N = {i+1}", ctx[:8])
    multi += "\n"
multi += diff_file("Cargo.toml", 'serde = "1.0.190"', 'serde = "1.0.210"', ctx[:8])
cases.append(case("diff-version-bump",
                  multi,
                  "What version was the serde dependency bumped to?",
                  "1.0.210"))

# ---- grep: the answer is the file of a query-relevant match -------------------------
grep = []
for i in range(70):
    grep.append(f"src/{WORDS[i % len(WORDS)]}/handler.rs:{10+i}:    connect({i});")
grep.insert(40, "src/db/pool.rs:42:pub fn connect_db(cfg: &Config) -> Pool {")
cases.append(case("grep-define-fn",
                  "\n".join(grep),
                  "Which file defines the connect_db function?",
                  "src/db/pool.rs"))

grep2 = []
for i in range(60):
    grep2.append(f"src/{WORDS[i % len(WORDS)]}/use.rs:{i+1}:    let k = read(API_KEY);")
grep2.insert(33, "config/secrets.rs:7:pub const API_KEY: &str = env!(\"API_KEY\");")
cases.append(case("grep-const-decl",
                  "\n".join(grep2),
                  "In which file is the API_KEY constant declared?",
                  "config/secrets.rs"))

# ====================================================================================
# ADVERSARIAL cases - the gold sits in a line the toolout stage DROPS by construction
# (a noise/INFO line, a diff CONTEXT line, a middle grep match, or a global aggregate that
# needs lines the windowing elides). The non-adversarial cases above can't catch
# over-aggressive windowing because their answer is always in a force-kept line; these can.
# A faithful windower keeps enough to answer → retention holds. A too-greedy one drops the
# answer line → these cases fail → the eval finally has teeth on windowing regressions.
# ====================================================================================

# --- answer in a buried INFO (noise) line, not in any error -------------------------
# No error at all: the signal-only/aggressive path keeps "errors + boundaries", so a fact
# living in a middle INFO line is exactly what gets elided. The target table name is unique
# (inserted once, not from the cycling WORDS pool) and the row count is 5-digit, so the
# `contains` gold is unambiguous.
target_rows = 84017
infolog_lines = [f"INFO  [{i:03d}] migrated table {WORDS[i % len(WORDS)]} ({1000+i} rows)"
                 for i in range(120)]
infolog_lines.insert(57, f"INFO  [057] migrated table audit_ledger ({target_rows} rows)")
infolog = "\n".join(infolog_lines)
cases.append(case("adv-info-rowcount",
                  infolog,
                  "How many rows were migrated for the audit_ledger table? Answer with the number.",
                  str(target_rows),
                  adversarial=True))

# Buried INFO carrying a config value, surrounded by templatable noise that folds away.
cfglog = "\n".join([f"INFO loading plugin {WORDS[i % len(WORDS)]}" for i in range(70)])
cfglog = (cfglog.split("\n")[:35] +
          ["INFO  effective max_connections = 384 (from config)"] +
          cfglog.split("\n")[35:])
cfglog = "\n".join(cfglog)
cases.append(case("adv-info-config",
                  cfglog,
                  "What is the effective max_connections value? Answer with the number.",
                  "384",
                  adversarial=True))

# --- answer in a diff CONTEXT (unchanged, space-prefixed) line ----------------------
# The signal-only path keeps +/- changed lines; an unchanged context line is dropped, so a
# fact that lives only in surrounding context vanishes.
ctx_named = [f"// owner: team-{WORDS[i]}" for i in range(20)]
ctx_named[8] = "    pub const RETRY_LIMIT: u32 = 5;  // unchanged context, not part of the hunk"
ctxdiff = diff_file("src/net.rs",
                    "fn send(req: Req) -> Resp",
                    "fn send(req: Req, deadline: Instant) -> Resp",
                    ctx_named)
cases.append(case("adv-diff-context-const",
                  ctxdiff,
                  "What is the value of RETRY_LIMIT shown in the context around the change?",
                  "5",
                  adversarial=True))

# --- answer in the Nth grep match (middle), not the first/last ----------------------
# Per-file/first-match sampling keeps a representative match per file; a specific middle hit
# in a long single-file run is exactly what gets sampled out.
gmid = [f"src/store.rs:{100+i}:    cache.put(key_{i}, val_{i});" for i in range(80)]
gmid[39] = "src/store.rs:139:    cache.put(SENTINEL_KEY, poison_value);"  # the one that matters
cases.append(case("adv-grep-middle-match",
                  "\n".join(gmid),
                  "Which key is paired with poison_value in a cache.put call?",
                  "SENTINEL_KEY",
                  adversarial=True))

# --- global aggregate that needs the elided body, not just kept boundaries ----------
# "How many …" over noise: the count can only be derived from lines the windower drops, so a
# windowed view literally cannot answer it. Job ids are prefixed (job-J<i>) so the small WARN
# count can't collide with an index inside a job line (the `contains` gold must be unambiguous).
warns = [f'{"WARN" if i % 5 == 0 else "INFO"}  job-J{i} finished' for i in range(50)]
n_warn = sum(1 for i in range(50) if i % 5 == 0)  # == 10
cases.append(case("adv-aggregate-count",
                  "\n".join(warns),
                  "How many WARN lines are in this log? Reply with exactly: count=<number>.",
                  f"count={n_warn}",
                  adversarial=True))

# ====================================================================================
# Broader non-adversarial cases - raise n and vary the content shape (large JSON,
# mixed log+grep, big multi-file diff, structured table, stack trace). Each gold sits
# in a line a faithful compressor keeps, so all three tools (original / llmtrim /
# Headroom) get a fair shot at the answer; these widen the corpus so a couple of fat
# blobs can't carry the token-savings mean.
# ====================================================================================

# Large JSON tool result with one anomalous record carrying the answer.
json_recs = []
for i in range(220):
    r = {"id": i, "ts": f"2026-05-{i % 28 + 1:02d}T08:{i % 60:02d}:00Z",
         "service": f"svc-{i % 6}", "status": "ok", "latency_ms": 5 + i % 40}
    if i == 133:
        r = {"id": 133, "ts": "2026-05-22T08:13:00Z", "service": "billing",
             "status": "error", "latency_ms": 9100,
             "detail": "stripe webhook signature mismatch for account acct_9F2K"}
    json_recs.append(r)
cases.append(case("json-error-account",
                  json.dumps({"results": json_recs}),
                  "Which account had a stripe webhook signature mismatch?",
                  "acct_9F2K"))

json_recs2 = []
for i in range(180):
    json_recs2.append({"region": f"r{i % 8}", "requests": 1000 + i, "p99_ms": 20 + i % 30})
json_recs2.insert(91, {"region": "eu-west-1", "requests": 1000000, "p99_ms": 4200,
                       "alert": "p99 SLA breach", "owner": "team-payments"})
cases.append(case("json-sla-owner",
                  json.dumps({"metrics": json_recs2}),
                  "Which team owns the region with the p99 SLA breach?",
                  "team-payments"))

# Mixed blob: build log, then a grep block, then a stack frame - answer in the stack.
mixed = "\n".join(f"INFO  [{i:03d}] linking object obj_{WORDS[i % len(WORDS)]}.o" for i in range(60))
mixed += "\n" + "\n".join(
    f"src/{WORDS[i % len(WORDS)]}/route.rs:{20 + i}:    dispatch(req_{i});" for i in range(40))
mixed += "\nthread 'worker-3' panicked at src/scheduler.rs:204: deadlock detected on lock `queue_mutex`"
cases.append(case("mixed-panic-lock",
                  mixed,
                  "Which lock was reported as deadlocked in the panic?",
                  "queue_mutex"))

# Big multi-file diff where exactly one hunk renames a public symbol.
bigdiff = ""
for i in range(12):
    bigdiff += diff_file(f"src/leaf_{WORDS[i]}.rs", f"let x = {i};", f"let x = {i} + 1;", ctx[:6])
    bigdiff += "\n"
bigdiff += diff_file("src/api.rs",
                     "pub fn get_user(id: u64) -> User",
                     "pub fn fetch_user(id: u64) -> User", ctx[:6])
cases.append(case("diff-rename-public",
                  bigdiff,
                  "What is the new name of the renamed public function get_user?",
                  "fetch_user"))

# docker ps / ls style table with one container in a crash loop.
table_rows = [f"{'a1b2c3d4e5f' + str(i):14}  svc-{i % 7:<8}  Up {i % 24}h  0.0.0.0:{8000 + i}->8080/tcp"
              for i in range(90)]
table_rows.insert(48, "deadbeefcafe   payments  Restarting (137) 12s ago   0.0.0.0:9090->8080/tcp")
cases.append(case("table-crashloop",
                  "CONTAINER ID    IMAGE     STATUS    PORTS\n" + "\n".join(table_rows),
                  "Which image is in a Restarting (crash loop) state?",
                  "payments"))

# Long stack trace with the root cause line at the bottom.
stack = ["Traceback (most recent call last):"]
for i in range(40):
    stack.append(f'  File "/app/svc/{WORDS[i % len(WORDS)]}.py", line {100 + i}, in handler_{i}')
    stack.append(f"    return process_{i}(payload)")
stack.append("ValueError: invalid currency code 'XZZ' in order ord-5521")
cases.append(case("stack-root-cause",
                  "\n".join(stack),
                  "Which invalid currency code raised the ValueError?",
                  "XZZ"))

OUT.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
n_adv = sum(1 for c in cases if c.get("adversarial"))
print(f"wrote {len(cases)} cases ({n_adv} adversarial) -> {OUT}")
