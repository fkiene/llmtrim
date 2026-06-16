# llmtrim

<strong>llmtrim is a local proxy that compresses your LLM API requests so you pay less, with no change to the answers.</strong>

It sits between your AI tools and the provider, strips the wasted tokens out of every request, and forwards it on. Same answers, smaller bill. **−31% input and −74% output tokens**, measured live across 112 A/B cases, with no change in answer quality.

[![crates.io](https://img.shields.io/crates/v/llmtrim)](https://crates.io/crates/llmtrim)
[![license](https://img.shields.io/badge/license-AGPL--3.0-blue)](https://www.gnu.org/licenses/agpl-3.0.txt)

```
  before:  your tool ───── full request ─────▶  OpenAI / Anthropic / …
                    ◀──────── reply ──────────

  after:   your tool ──▶ llmtrim ──smaller──▶  OpenAI / Anthropic / …
                            (on your machine)
                    ◀──────── reply ──────────  (same answer)
```

A lot of what your tools send is waste: a 200-line build log where 2 lines are errors, a tool schema resent identically 50 times, a JSON array with 500 near-identical rows. llmtrim removes it before it's sent. Everything runs locally; nothing is ever sent to us.

> [!IMPORTANT]
> **It can never make your bill bigger or break a request.** Every compression step is re-measured with the provider's real tokenizer; if a step doesn't save tokens, it's reverted. If the provider rejects the compressed request, the original is resent verbatim. Worst case is zero savings, never a worse outcome.

## Install

```bash
cargo install llmtrim   # or: cargo binstall llmtrim
llmtrim setup           # configure the local interceptor + environment
llmtrim doctor          # verify it
```

Other channels (same binary): `brew install fkiene/tap/llmtrim` · `scoop install llmtrim` · `npm i -g @llmtrim/cli@latest` · `docker run ghcr.io/fkiene/llmtrim`. See [INSTALL.md](https://github.com/fkiene/llmtrim/blob/main/INSTALL.md).

## Use it

After `llmtrim setup`, any tool that honors `HTTPS_PROXY` routes through it automatically: Claude Code, Codex, Cursor, Aider, Gemini CLI, your own app. (GitHub Copilot pins its certificates and can't be intercepted.)

Or run the compression directly, no proxy:

```bash
echo '{"model":"gpt-4o","messages":[...]}' | llmtrim compress --provider openai > out.json
echo '{"model":"gpt-4o","messages":[...]}' | llmtrim send     --provider openai   # compress, call, print
```

**Zero config needed.** The default `auto` mode inspects each request and picks the right compressors for its shape (tool-heavy → `agent`, code → `code`, long context → `rag`, else `aggressive`). Force one with `LLMTRIM_PRESET=<name>`.

Or expose the engine to an MCP client. `llmtrim mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io) server over stdin/stdout with five tools: `llmtrim_compress` (full request body), `llmtrim_compress_text` (single text blob), `llmtrim_read_file_compressed` (read and compress a local file, rejecting secrets and binaries), `llmtrim_read_folder_compressed` (read and compress multiple source files in a folder with code-aware skeletonization, importance sorting, and a structured `summary` UX field), and `llmtrim_stats` (savings ledger). All compression tools accept optional `client` and `model` arguments for accurate stats grouping. Every call records to the same ledger as the proxy. Run `llmtrim mcp install` to register it with Claude Code, or `llmtrim mcp install --print` to get the config for any other client.

## As a library

The compression engine is the [`llmtrim-core`](https://crates.io/crates/llmtrim-core) crate (no network, no async), with native bindings for **Python, Ruby, Swift and Kotlin** (see the [project README](https://github.com/fkiene/llmtrim)).

## License

AGPL-3.0-only.
