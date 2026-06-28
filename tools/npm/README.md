# llmtrim

**llmtrim is a local proxy that compresses your LLM API requests so you pay less, with no
change to the answers.** It strips wasted tokens (verbose tool output, resent schemas,
bulky JSON, long context) out of every request before it reaches the provider: −31% input
and −74% output tokens, measured live across 112 A/B cases. Works with Claude Code, Cursor,
Cline, and any tool that talks to OpenAI / Anthropic / Google / DeepSeek / Mistral & co.

Every cut is re-counted with the provider's real tokenizer and auto-reverted if it doesn't
save, so it can never increase your bill or break a request.

```bash
npm install -g @llmtrim/cli@latest && llmtrim setup
# open a new shell, then watch the bill shrink:
llmtrim status
```

`setup` is transparent and fully reversible (`llmtrim uninstall`): a local CA, a proxy
block in your shell profile, a background service. Everything runs locally; nothing is
ever sent to us.

This package installs a prebuilt native binary for your platform (Linux, macOS, Windows;
x64 & arm64). No Rust toolchain needed.

Docs, benchmarks (112 live A/B cases), and source: **https://github.com/fkiene/llmtrim**

License: MPL-2.0
