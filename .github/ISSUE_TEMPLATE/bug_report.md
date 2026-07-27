---
name: Bug report
about: Something compressed wrong, the proxy misbehaved, or a stage broke
title: ""
labels: bug
assignees: ""
---

## What happened

<!-- What went wrong, in plain terms. -->

## How are you running llmtrim?

- [ ] Proxy / daemon (`llmtrim setup` / `start` / `serve` / `wrap`): intercepting a tool's traffic
- [ ] CLI (`llmtrim compress` / `send`)
- [ ] Subscription reroute (`llmtrim sub` → codex | kimi | grok)
- [ ] MCP (`llmtrim mcp`)

- Tool / client (proxy path): <!-- e.g. Claude Code, Cursor, Codex, a Node app -->
- Provider: <!-- openai | anthropic | google (`gemini` is an alias for google) -->
- Preset / config: <!-- auto (default), safe, agent, code, rag, aggressive, cache, reasoning, frugal; or paste ~/.config/llmtrim/config.toml if custom -->

## Reproduce

```bash
# Proxy: the steps + the tool action that triggers it.
# For mangled requests, set LLMTRIM_CAPTURE_DIR=~/llmtrim-capture and re-run, then attach the before/after pair.
# CLI: the command + a MINIMAL request body, e.g.
echo '<request json>' | llmtrim compress --provider openai
```

> ⚠️ **Redact secrets first.** Strip API keys, `authorization` headers, and private prompt
> content. llmtrim never needs them to reproduce a compression bug. Security *vulnerabilities*
> go through the private advisory link, not a public issue (see SECURITY.md).

## Expected vs actual

- Expected:
- Actual:

<!-- Paste error output or proxy log here. Daemon log is usually ~/.llmtrim/serve.log; or run `llmtrim serve` in the foreground. -->

## Environment

- `llmtrim --version`:
- `llmtrim status` (daemon + CA + savings):
- `llmtrim doctor` (paste failing checks only):
- Install method: <!-- npm | curl installer | homebrew | scoop | cargo install | docker | source -->
- OS / arch:
