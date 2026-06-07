# Contributing to llmtrim

Thanks for your interest! llmtrim is a static, deterministic LLM prompt compressor —
zero auxiliary model calls, every transform measured with the real target tokenizer.

## Ground rules

The behavioral guidelines in [`CLAUDE.md`](CLAUDE.md) apply to humans too:

- **Simplicity first** — minimum code that solves the problem; no speculative abstraction.
- **Surgical changes** — touch only what the change requires.
- **Code universally** — text processing must handle any language/locale, not just English
  (detect the language, use the matching resource; prefer Unicode-aware operations).
- **Goal-driven** — turn the task into a verifiable test, then make it pass.

## Development

```bash
cargo build                  # debug build
cargo test                   # run the suite (deterministic, no network)
cargo clippy --all-targets -- -D warnings
cargo fmt --all
cargo build --features live  # the bench network path (async-openai + tokio)
```

CI runs fmt, clippy (`-D warnings`), and the test suite on Linux, macOS, and Windows. Keep all three green.

## Adding a compression stage

Stages implement the `Transform` trait (`src/gate.rs`) and are assembled in
`stages_for` (`src/lib.rs`). A stage:

1. Declares a `GateKind` — `InputTokens` (reverted if it doesn't reduce tokens),
   `OutputShaping`, or `Structural`.
2. Mutates the JSON-backed `Request` at JSON-pointer addresses (lossless by construction
   for any field it doesn't touch).
3. Adds a config flag in `DenseConfig` (`src/config.rs`) and is wired into a preset.
4. Ships with unit tests proving the token win and quality behavior.

Lossy stages stay **off by default** and are quality-checked offline (see README §6).

## Pull requests

- One logical change per PR; every changed line should trace to the stated goal.
- Add or update tests for new behavior.
- Run fmt + clippy + tests before pushing.
- Fill in the PR template.

## Sign your commits (DCO)

We use the [Developer Certificate of Origin](https://developercertificate.org/). Sign off
every commit:

```bash
git commit -s -m "your message"
```

The `Signed-off-by` line certifies you wrote the patch or have the right to submit it under
the project license.

## License

By contributing, you agree your contributions are licensed under AGPL-3.0-only.
