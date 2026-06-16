"""Compress a request in-process, then send it yourself: no proxy, no CA trust.

This is the integration route for environments where the `HTTPS_PROXY` interception
can't run: sandboxed / serverless functions, and certificate-pinned HTTP clients that
reject a MITM CA. You call `compress()` natively, then POST the shaped body with your
own HTTP client. Nothing intercepts the connection.

Run after building + installing the wheel:
    crates/llmtrim-uniffi/scripts/build-wheel.sh --release
    pip install target/wheels/llmtrim-*.whl
    OPENAI_API_KEY=sk-... python crates/llmtrim-uniffi/examples/compress_then_send.py
"""

import json
import os
import urllib.request

import llmtrim

# 1. A normal provider-shaped request body (here: OpenAI chat completions).
request = json.dumps(
    {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Summarize the build log above."},
        ],
    }
)

# 2. Compress in-process. The provider is Provider.OPEN_AI (or None to auto-detect
#    from the body); the preset is a workload name such as "aggressive" / "agent" /
#    "code" / "rag" / "safe" (None uses the environment / config-file defaults).
out = llmtrim.compress(request, llmtrim.Provider.OPEN_AI, "aggressive")

print(f"input tokens {out.input_tokens_before} -> {out.input_tokens_after}", flush=True)
print(f"provider={out.provider} model={out.model} tokenizer={out.tokenizer_label}")

# 3. Send the compressed body yourself. No proxy, no CURL_CA_BUNDLE, no env-var setup.
#    This works inside a sandbox and against a pinned TLS client.
req = urllib.request.Request(
    "https://api.openai.com/v1/chat/completions",
    data=out.request_json.encode("utf-8"),
    headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    body = json.load(resp)

print(body["choices"][0]["message"]["content"])
