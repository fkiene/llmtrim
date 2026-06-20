# llmtrim gateway plugin (Proxy-Wasm)

One Proxy-Wasm module that compresses LLM API request bodies with llmtrim as they pass through
the gateway. It buffers the request body, runs it through the llmtrim engine, and forwards the
smaller body upstream. The same `.wasm` runs on **Kong** and **Higress** (both are Proxy-Wasm
0.2 ABI hosts); only the deployment config differs.

It is fail-open: a non-JSON body, an undetectable provider, an oversized body, or any engine
error forwards the original request unchanged. A gateway that fronts one provider should set
`provider` so it does not depend on shape auto-detection.

## Get the module

Each release publishes the same `.wasm` to both channels:

- **OCI artifact** (for Higress): `ghcr.io/fkiene/llmtrim-gateway-plugin:<version>` (version tags
  only, no `:latest`, so a control plane never picks up a new build silently)
- **Release asset** (for Kong): `llmtrim-gateway-plugin.wasm` on the matching GitHub Release;
  download it to the host and point `wasm_filters_path` at it (Kong loads from disk, not a URL)

Or build it yourself:

```sh
cargo build -p llmtrim-gateway-plugin --target wasm32-wasip1 --release
# artifact: target/wasm32-wasip1/release/llmtrim_gateway_plugin.wasm
```

## Configuration

The host passes a JSON config object. Every field is optional:

| Field            | Type   | Default            | Meaning                                                        |
| ---------------- | ------ | ------------------ | -------------------------------------------------------------- |
| `provider`       | string | auto-detect        | `openai`, `anthropic`, or `google`. Set it when fronting one.  |
| `preset`         | string | `auto`             | `auto`, `aggressive`, `agent`, `code`, `rag`, `safe`.          |
| `max_body_bytes` | number | 4194304 (4 MiB)    | Bodies larger than this are forwarded uncompressed; `0` disables the guard. |

A missing or malformed config falls back to these defaults rather than disabling the plugin.

The module targets `wasm32-wasip1`, not `wasm32-unknown-unknown`: the engine needs an entropy
source (for hash seeding), which on `wasm32-unknown-unknown` only resolves to a browser JS
backend that a gateway has no runtime for. `wasm32-wasip1` uses the WASI `random_get` import,
which the Envoy-based (Higress) and Kong wasm runtimes provide.

## Deploy on Kong

Kong loads Proxy-Wasm filters from `wasm_filters_path`. Point Kong at the `.wasm`, then attach
the filter to a service or route with the config above:

```yaml
# kong.conf
wasm = on
wasm_filters_path = /etc/kong/wasm
```

```yaml
# a route's filter chain
filters:
  - name: llmtrim_gateway_plugin
    config: '{"provider":"openai","preset":"auto"}'
```

The filter-config field name and encoding have moved across Kong Gateway releases. Verify the
exact `filters[].config` form against the Kong version you run (the plugin reads the bytes the
host passes to `on_configure` as JSON); a wrong field silently leaves the plugin on defaults.

## Deploy on Higress

Higress (Envoy-based) loads it as a `WasmPlugin`:

```yaml
apiVersion: extensions.higress.io/v1alpha1
kind: WasmPlugin
metadata:
  name: llmtrim
  namespace: higress-system
spec:
  url: oci://ghcr.io/fkiene/llmtrim-gateway-plugin:VERSION
  defaultConfig:
    provider: openai
    preset: auto
```

Higress assigns plugin execution order itself; the Istio `WasmPlugin` `phase`/`priority`
fields are not part of the Higress (`extensions.higress.io/v1alpha1`) schema. Use `matchRules`
to scope the plugin to specific routes or domains if needed.

## What lives where

All logic (fail-open rules, the size guard, config parsing) is in the `llmtrim-gateway` crate,
which is unit-tested on the host. This crate is only the Proxy-Wasm lifecycle glue and builds
solely for wasm, so it carries no host tests.
