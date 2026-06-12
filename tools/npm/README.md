# llmtrim

**Cut ~66% off your LLM bill.** A drop-in local HTTPS proxy that compresses every LLM
request and reply — input, output, and cache — with zero extra model calls. Works with
Claude Code, Cursor, Cline, and any tool that talks to OpenAI / Anthropic / Google /
DeepSeek / Mistral & co. Answers unchanged: every cut is re-counted with the provider's
real tokenizer and auto-reverted if it doesn't save.

```bash
npm install -g @llmtrim/cli && llmtrim setup
# open a new shell, then watch the bill shrink:
llmtrim status --watch
```

`setup` is transparent and fully reversible (`llmtrim uninstall`): a local CA, a proxy
block in your shell profile, a background service. Everything runs locally — nothing is
ever sent to us.

This package installs a prebuilt native binary for your platform (Linux, macOS, Windows;
x64 & arm64) — no Rust toolchain needed.

Docs, benchmarks (112 live A/B cases), and source: **https://github.com/fkiene/llmtrim**

License: AGPL-3.0-only
