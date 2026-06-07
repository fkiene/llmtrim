---
name: Bug report
about: Something compressed wrong, the proxy misbehaved, or a stage broke
title: ""
labels: bug
assignees: ""
---

## What happened

<!-- A clear description of the bug. -->

## How are you running llmtrim?

- [ ] Proxy (`llmtrim setup` / `serve`) — intercepting a tool's traffic
- [ ] CLI (`llmtrim compress` / `send` / `batch`)

- Tool / client (proxy path): <!-- e.g. Claude Code, Cursor, Codex, a Node app -->
- Provider: <!-- openai | anthropic | gemini | … -->
- Preset / config: <!-- auto (default), rag, code, … — paste your config.toml if custom -->

## Reproduce

```bash
# Proxy: the steps + the tool action that triggers it.
# CLI:  the command + a MINIMAL request body, e.g.
echo '<request json>' | llmtrim compress --provider openai
```

> ⚠️ **Redact secrets first** — strip API keys, `authorization` headers, and private prompt
> content. llmtrim never needs them to reproduce a compression bug. (Security *vulnerabilities*
> go through the private advisory link, not a public issue — see SECURITY.md.)

## Expected vs actual

- Expected:
- Actual:

<!-- Paste any error output / proxy log (run `llmtrim serve` in the foreground) here. -->

## Environment

- `llmtrim --version`:
- `llmtrim monitor` (daemon + CA + savings state):
- Install method: <!-- prebuilt binary | cargo install | homebrew | source -->
- OS / arch:
