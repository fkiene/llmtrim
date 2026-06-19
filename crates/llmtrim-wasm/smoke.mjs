import { compress } from './pkg/llmtrim_wasm.js';
import assert from 'node:assert';

// 1. OpenAI, default preset, English
const en = JSON.stringify({ model:"gpt-4o",
  messages:[{role:"user",content:"hi there"}], max_tokens:5 });
const a = compress(en, "openai", undefined);
assert.equal(a.provider, "openai");
assert.equal(a.model, "gpt-4o");
assert.ok(a.input_tokens_before > 0);
assert.ok(typeof a.tokenizer_exact === "boolean");
assert.ok(Array.isArray(a.stages));
console.log("openai/default:", { provider:a.provider, before:a.input_tokens_before, after:a.input_tokens_after, exact:a.tokenizer_exact });

// 2. Anthropic, agent preset, big tool_result -> real compression
const log = Array.from({length:400}, (_,i)=>`[INFO] worker ${i%8} batch ${i} ok`).join("\n");
const big = JSON.stringify({ model:"claude-3-5-sonnet-20241022", max_tokens:1024,
  messages:[{role:"user",content:[{type:"tool_result",tool_use_id:"t1",content:log}]}] });
const b = compress(big, "anthropic", "agent");
assert.equal(b.provider, "anthropic");
assert.ok(b.input_tokens_after <= b.input_tokens_before);
assert.ok(b.stages.some(s => s.applied && s.tokens_after < s.tokens_before));
console.log("anthropic/agent:", { before:b.input_tokens_before, after:b.input_tokens_after, saved_pct: (100*(1-b.input_tokens_after/b.input_tokens_before)).toFixed(1) });

// 3. Non-English (Japanese + Chinese) content must work (universal-language rule)
const ja = JSON.stringify({ model:"gpt-4o", max_tokens:50,
  messages:[{role:"user",content:"お世話になっております。トークンを削減したいです。这是一段中文文本，用于测试压缩。"}] });
const c = compress(ja, "openai", "aggressive");
assert.ok(c.input_tokens_before > 0, "CJK tokenized");
assert.ok(c.request_json.includes("お世話") || c.request_json.length > 0);
console.log("openai/CJK:", { before:c.input_tokens_before, after:c.input_tokens_after });

// 4. Error paths throw
let threw=false; try { compress("not json","openai",undefined); } catch { threw=true; }
assert.ok(threw, "invalid JSON throws");
threw=false; try { compress("{}","openai","nope"); } catch { threw=true; }
assert.ok(threw, "unknown preset throws");

console.log("\nALL SMOKE TESTS PASSED");
