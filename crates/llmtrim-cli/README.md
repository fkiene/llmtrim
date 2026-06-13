# llmtrim

Cut your LLM bill. `llmtrim` is a drop-in proxy that compresses LLM API requests in
flight — input prompts, tool outputs, schemas — with static, deterministic algorithms and
**zero extra model calls**, typically saving 30–90% of input tokens. It speaks the OpenAI,
Anthropic and Google wire shapes and never changes the model's output behavior by default.

```bash
# install (or: brew install fkiene/tap/llmtrim · scoop install llmtrim · npm i -g @llmtrim/cli)
cargo install llmtrim

llmtrim setup          # configure the local interceptor + env
llmtrim doctor         # verify the setup
```

Point your app's HTTPS traffic through the local interceptor and requests are compressed
before they reach the provider; the response comes back untouched. See the
[project README](https://github.com/fkiene/llmtrim) for the full walkthrough, benchmarks,
and configuration.

## Library / bindings

This crate is the CLI and proxy. To embed the compression engine directly:

- Rust: [`llmtrim-core`](https://crates.io/crates/llmtrim-core) — the deterministic engine,
  no network, no async.
- Python / Ruby / Swift / Kotlin: the `llmtrim-uniffi` bindings in the
  [repository](https://github.com/fkiene/llmtrim).

## License

AGPL-3.0-only.
