# Hermes Agent + llmtrim

[Hermes Agent](https://hermes-agent.nousresearch.com) (Nous Research) calls an OpenAI-compatible
model under the hood. llmtrim is a local proxy that sits in front of that call, compresses the
request (tool schemas resent every turn, conversation history, MCP tool output, scraped pages),
and forwards it. You pay less for the same answers. No code change to Hermes; no Hermes-specific
build of llmtrim.

## Quick start

This is the whole setup when Hermes uses a mainstream provider (OpenRouter, OpenAI, Anthropic,
or Google). A self-hosted or `custom:` endpoint needs one extra step, covered in
[Custom endpoints](#custom-openai-compatible-endpoints).

1. Install llmtrim and trust its certificate:

   ```bash
   llmtrim setup
   ```

   `setup` starts the proxy, writes a CA certificate, and wires `HTTPS_PROXY` into your shell
   profile. Open a new shell (or re-source the profile) so the change takes effect.

2. Start Hermes in that shell and use it as usual.

   If Hermes runs as a service or container rather than from your shell, copy these into *its*
   environment so it routes through llmtrim and trusts the certificate:

   ```bash
   export HTTPS_PROXY=http://127.0.0.1:43117
   export HTTP_PROXY=http://127.0.0.1:43117
   export SSL_CERT_FILE=~/.llmtrim/ca.pem
   export REQUESTS_CA_BUNDLE=~/.llmtrim/ca.pem
   export CURL_CA_BUNDLE=~/.llmtrim/ca.pem
   export NODE_EXTRA_CA_CERTS=~/.llmtrim/ca.pem
   ```

3. Check it is working:

   ```bash
   llmtrim status      # tokens trimmed and dollars saved off your bill
   ```

If `status` shows no traffic, run `llmtrim doctor` for an end-to-end diagnosis. llmtrim forwards
Hermes' own API key untouched and never stores keys.

## Details

### Custom OpenAI-compatible endpoints

llmtrim intercepts the known provider hosts automatically. A `custom:` endpoint on your own host
is not in that set, so add it. Check the host Hermes dials with `hermes model` (or
`inference.provider` / `inference.model` in its `config.yaml`), then list it.

Prefer the config file. The llmtrim daemon reads it at startup, so it works no matter which
shell or service launched the daemon:

```toml
# ~/.config/llmtrim/config.toml  (honors $XDG_CONFIG_HOME if set)
extra_hosts = ["llm.mycorp.com"]
```

The env-var form works too, but it must be set in the **daemon's** environment, not the shell
that runs Hermes:

```bash
export LLMTRIM_EXTRA_HOSTS=llm.mycorp.com    # comma-separated for several
```

The intercept set is read once when the daemon starts. After changing it, restart llmtrim, then
restart Hermes so it picks up the refreshed certificate (the CA regenerates on restart to cover
the new host). For a systemd or container daemon, the var or config file has to be in that
daemon's environment, not your interactive shell.

List the **exact** host (`llm.mycorp.com`), never a bare apex like `mycorp.com`. A bare apex is
worse than imprecise: by RFC 5280, that entry lets the local CA sign certificates for every
`*.mycorp.com` host, even though the proxy still intercepts only the exact name. The exact host
keeps the certificate authority's reach as narrow as what is actually intercepted. Invalid
entries (a bare TLD, an IP address, a wildcard, or a value with a scheme like `https://...`) are
dropped silently rather than rejected, so if a custom host produces no captures, a dropped or
misspelled entry is the first thing to check.

### Confirm interception, then measure

Do not judge savings from one Hermes session: Hermes chooses its own iteration count and
providers cache unevenly, so a single run is noise. Confirm the traffic flows first, then
measure on a fixed set of tasks.

Confirm interception by capturing the before/after request bodies:

```bash
LLMTRIM_CAPTURE_DIR=~/.llmtrim/capture llmtrim    # or set capture_dir in config.toml
# ... run a Hermes task that calls the model ...
ls ~/.llmtrim/capture/                             # one JSON per compressed request
```

No files means the traffic is not reaching llmtrim. Re-check that Hermes inherited the env vars
from the Quick start and, for a `custom:` endpoint, that its host is in `extra_hosts`.

To test a custom host on its own, before involving Hermes, send one request through the proxy
and confirm a capture lands:

```bash
HTTPS_PROXY=http://127.0.0.1:43117 curl --cacert ~/.llmtrim/ca.pem \
  https://llm.mycorp.com/v1/chat/completions \
  -H "authorization: Bearer $YOUR_KEY" -H 'content-type: application/json' \
  -d '{"model":"...","messages":[{"role":"user","content":"hi"}]}'
ls ~/.llmtrim/capture/
```

For a savings number you can trust, drive a fixed set of representative tasks and compare totals
from `llmtrim status` across runs. Caching only helps on a provider tier that caches; a flex or
batch tier reports none, so a cache-stability gain cannot show up there.

### Notes

- The connection to the provider stays under verifying TLS end to end. Do not bypass the proxy
  or disable certificate checks to dodge a TLS error; trust the CA instead (Quick start step 1),
  or compression and tracking are skipped.
- A rejected compressed request is resent verbatim, so the worst case is zero savings, never a
  broken call.
- Prompt-cache discounts are preserved: llmtrim keeps the cached prefix stable across turns.
- To send Hermes' egress through a second proxy as well (a corporate gateway, for example), set
  `LLMTRIM_UPSTREAM_PROXY` / `upstream_proxy`.
- `llmtrim doctor` diagnoses a broken setup and names the fix; `llmtrim uninstall` reverses
  everything `setup` changed.
