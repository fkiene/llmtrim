"""End-to-end tests for the llmtrim Python bindings (the installed wheel).

Run after building + installing the wheel:
    crates/llmtrim-uniffi/scripts/build-wheel.sh
    pip install target/wheels/llmtrim-*.whl
    pytest crates/llmtrim-uniffi/tests/python

Tests use explicit presets so they don't depend on ambient LLMTRIM_* environment config.
"""

import json

import pytest

import llmtrim


def _openai(content="hello world"):
    return json.dumps(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": content}], "max_tokens": 5}
    )


def test_compress_returns_projected_fields():
    out = llmtrim.compress(_openai(), llmtrim.Provider.OPEN_AI, "safe")
    assert out.provider == "openai"
    assert out.model == "gpt-4o"
    assert out.tokenizer_exact is True
    assert out.input_tokens_before > 0
    assert out.request_json  # non-empty compressed body
    json.loads(out.request_json)  # still valid JSON


def test_agent_preset_compresses_a_tool_result():
    diff = "".join(
        f"diff --git a/f{i}.rs b/f{i}.rs\n@@ -1,2 +1,2 @@\n-old {i}\n+new {i}\n" for i in range(40)
    )
    req = json.dumps(
        {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": diff}]}
            ],
            "max_tokens": 1024,
        }
    )
    out = llmtrim.compress(req, llmtrim.Provider.ANTHROPIC, "agent")
    assert out.provider == "anthropic"
    assert out.input_tokens_after < out.input_tokens_before


def test_auto_detect_provider_when_none():
    req = json.dumps({"system": "s", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
    out = llmtrim.compress(req, None, "safe")
    assert out.provider == "anthropic"


def test_unknown_preset_raises():
    with pytest.raises(llmtrim.LlmtrimError.UnknownPreset):
        llmtrim.compress(_openai(), llmtrim.Provider.OPEN_AI, "no-such-preset")


def test_invalid_json_raises():
    with pytest.raises(llmtrim.LlmtrimError.Compress):
        llmtrim.compress("not json", llmtrim.Provider.OPEN_AI, "safe")
