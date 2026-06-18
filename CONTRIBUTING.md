# Contributing to llmtrim

Thanks for your interest! llmtrim is a static, deterministic LLM prompt compressor:
zero auxiliary model calls, every transform measured with the real target tokenizer.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ground rules

Four principles guide every change:

- **Simplicity first:** minimum code that solves the problem; no speculative abstraction.
- **Surgical changes:** touch only what the change requires.
- **Code universally:** text processing must handle any language/locale, not just English
  (detect the language, use the matching resource; prefer Unicode-aware operations).
- **Goal-driven:** turn the task into a verifiable test, then make it pass.

## Development

The standard check loop (run all three before pushing):

```bash
cargo fmt
cargo clippy --features intercept
cargo nextest run --features intercept
```

Notes:

- Use `cargo nextest run`, not `cargo test`: it runs in parallel and prints compact output.
- Do not pass `-- -D warnings` to clippy. `warnings = "deny"` is already set in the
  workspace `[lints.rust]`, and adding the flag forks the build cache between clippy and test.
- Keep `--features intercept` consistent across commands so the incremental cache is reused.

The suite is deterministic and hits no network. `cargo build --features live` builds the
optional bench network path (async-openai + tokio); async is confined to that path and is
not allowed in the core compression stages.

Before the first push that opens a PR, also check coverage of the files you changed:

```bash
cargo llvm-cov --features intercept,mcp --summary-only
```

CI runs fmt, clippy, and the test suite on Linux, macOS, and Windows. Keep all three green.

Enable the local git hooks (mirror of CI plus a gitleaks secret scan; needs `gitleaks` or Docker):

```bash
git config core.hooksPath .githooks
```

## Adding a compression stage

Stages implement the `Transform` trait (`src/gate.rs`) and are assembled in
`stages_for` (`src/lib.rs`). A stage:

1. Declares a `GateKind`: `InputTokens` (reverted if it doesn't reduce tokens),
   `OutputShaping`, or `Structural`.
2. Mutates the JSON-backed `Request` at JSON-pointer addresses (lossless by construction
   for any field it doesn't touch).
3. Adds a config flag in `DenseConfig` (`src/config.rs`) and is wired into a preset.
4. Ships with unit tests proving the token win and quality behavior.

Lossy stages stay **off by default** and are quality-checked offline (see README §6).

## Before a large change

Open an issue first for anything beyond a small fix. Describe the problem and the approach
you have in mind, and wait for a maintainer to confirm the direction. A short agreement up
front saves a large PR from a redesign in review. New user-facing surfaces (a CLI mode, an
MCP tool, an output channel) and any change to the published `llmtrim-core` public API need
this step.

## Reporting vs. implementing

Filing an issue does not commit you to writing the code. Tell us which you mean: the
feature-request template has a checkbox, or just say so in the issue.

If you want to implement it, comment to claim it and a maintainer will assign you; an
assigned issue is yours until it goes quiet for about two weeks. If you only want to
report it, a maintainer or another contributor may pick it up. Either way, for a feature
we agree the approach in the issue before a PR is opened, so a maintainer implementing
your request will post the intended design on the issue first and give you a chance to
respond.

## Pull requests

- One logical change per PR; every changed line should trace to the stated goal.
- Link the issue the PR addresses (`Closes #123`).
- Add or update tests for new behavior.
- Add a `CHANGELOG.md` entry under `## [Unreleased]` for any user-observable change, written
  for users. Skip it only for invisible changes (cosmetic, docs, tests, no-op refactors).
- Run the check loop above before pushing.
- Fill in the PR template.

## Commit hygiene

One coherent change per commit, with a message that explains it for a reviewer. Fold every
fixup, "wip", or review-fix tweak into the commit it belongs to before you push. Force-push a
rewritten branch with `--force-with-lease`, never a bare `--force`.

## Review

A maintainer reviews each PR and may ask you to split a large change into smaller pieces or to
move ecosystem-specific logic out of `llmtrim-core`. Expect a first response within a week. A
request for changes is about the code, not about you.

## Sign your commits (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/). Sign off
every commit:

```bash
git commit -s -m "your message"
```

The `Signed-off-by` line certifies you wrote the patch or have the right to submit it under
the project license.

## License

By contributing, you agree your contributions are licensed under MPL-2.0.

You also grant François Kiene, and his successors and assigns, the right to license your
contribution under any other license, including proprietary or commercial terms, so the
project can relicense or offer a commercial dual-license later without contacting every
contributor. You keep the copyright to your contribution. This is separate from your DCO
sign-off, which only certifies the origin of your work.
