//! Turn-stable compression memo — byte-identical frozen prefix across agent turns.
//!
//! ## The problem this solves
//!
//! llmtrim is a stateless MITM proxy: it compresses each request on its own. But agent
//! loops resend the *same* conversation plus one new turn every step — recent measurement
//! puts 85–95% of an agentic request's prompt tokens at unchanged turn-to-turn ("Stateful
//! Inference for Low-Latency Multi-Agent Tool Calling", 2026). Provider prefix caches
//! (Anthropic `cache_control`, OpenAI implicit) only pay out when the cached prefix is
//! **byte-identical** across calls.
//!
//! llmtrim's stages are deterministic per request but *context-sensitive*: retrieve's query
//! and RM3 expansion, the n-gram dictionary, and dedup all read the **whole** conversation —
//! so the compressed form of an *old* message can change when a *new* turn arrives. Two
//! consecutive turns then serialize a divergent prefix → the provider cache is busted → the
//! product's headline savings leak silently on exactly the highest-traffic (agent) shape.
//!
//! FlowKV (arXiv:2505.15347) and EpiCache (2025) formalize the fix: **freeze the past turns,
//! process only the new ones.** This memo is that idea at the request-rewrite layer — it does
//! not change any stage; it makes the *output* of a stage over an already-seen message prefix
//! reproducible byte-for-byte by remembering what we emitted last turn and reusing it verbatim.
//!
//! ## How it works (design)
//!
//! - **Key.** A cumulative 128-bit hash *chain* over the **original** bytes of the request's
//!   conversation messages (the `messages` / `input` / `contents` array, whichever the wire
//!   shape uses). `prefix_hash[k]` fingerprints original messages `0..=k`. Appending a new
//!   turn leaves every earlier `prefix_hash[k]` unchanged; changing one byte of an old message
//!   changes that boundary and every one after it.
//! - **Store.** An in-memory, size-capped, generation-evicted map from `prefix_hash[k]` to the
//!   **entire compressed conversation item** llmtrim emitted for original message `k` last time
//!   it was the head of a prefix (chat `messages[]` objects, Responses `input[]` items including
//!   `function_call` / `function_call_output`, Gemini `contents[]`, …). Storing the whole item —
//!   not only a `content` field — is required for agent wire shapes where the compressible text
//!   lives under `output` / `arguments` rather than `content`. No prompt text is read back from
//!   anywhere on disk — see *Privacy*.
//! - **Reuse.** On a new request, walk the original messages front-to-back; the longest run
//!   `0..=m` whose every boundary hash is present in the store is the *frozen prefix*. We still
//!   run the normal full-request pipeline (so all legend/injection/Stage-A logic stays exactly
//!   correct), then overwrite the frozen-prefix slots in the compressed output with the stored
//!   items — making them identical to last turn's output, which is what the provider cache keys
//!   on. Only the *suffix* (new messages) carries this turn's fresh content compression; the
//!   input-token gate still governs whatever was freshly compressed.
//! - **Record.** After rewriting, store this request's `(prefix_hash[k] -> compressed item)`
//!   for every conversation message **that is not already memoized** (first-write-wins), so
//!   the next turn can freeze one message further without ever remutating a prior freeze.
//!
//! ## Legend / instruction interaction & the v1 carve-out
//!
//! Most content stages compress each message *self-containedly* (retrieve prunes sentences,
//! dedup folds duplicate lines, hygiene/serialize reshape a message's own JSON, toolout windows
//! a message's own log) and any legend they inject is **static** text (the TOON `FORMAT_LEGEND`
//! is a build-time constant) — so reusing an earlier message's compressed bytes verbatim is
//! always sound.
//!
//! The **n-gram** stage is the exception: it rewrites content with placeholders (`§1`, `§2`, …)
//! whose assignment depends on phrase frequencies across the *whole* conversation, and injects a
//! one-time legend defining them. Splicing an old turn's `§`-encoded content into a new turn
//! whose legend numbers the placeholders differently would corrupt it. Rather than reach into
//! stage internals (out of scope) to freeze the dictionary, **v1 disables reuse whenever the
//! n-gram stage is enabled** (`config.ngram`) — the one stage where post-hoc splicing is unsafe.
//! Every other preset (`agent`, `rag`, `code`, `cache`, `safe`, …) gets turn-stable prefixes.
//! Freezing the n-gram dictionary over a frozen prefix is the natural v2.
//!
//! ## Fallbacks (the memo is an optimization, never a correctness dependency)
//!
//! Any mismatch or doubt falls back to full stateless compression (today's behavior): a
//! non-array conversation, an unexpected message-count delta between the original and the
//! compressed output (some stage restructured the array), the n-gram carve-out, or simply a
//! cold prefix. The memo can only ever make an *already-correct* compressed request reuse bytes
//! it itself produced for an identical earlier message.
//!
//! ## Privacy (SECURITY.md: prompt text is never persisted to disk)
//!
//! The store lives **only in process memory** and is never written to disk, logged, or sent
//! anywhere — the same in-memory-only treatment the `serve` proxy already gives prompt bytes.
//! Keys are 128-bit hashes of the original prefix (not the text). Values are the compressed
//! conversation items — which are *already* in flight to the provider on this very request — so
//! the memo retains nothing the proxy isn't already handling in memory for the duration of the
//! call. It is size-capped (LRU/generation eviction) so memory stays bounded on a long-running
//! daemon.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use serde_json::Value;

/// Default capacity (number of memoized message-prefix entries). One agent conversation of
/// `n` turns contributes ~`n` entries; a few thousand covers many concurrent conversations
/// while staying well under a megabyte of small JSON fragments. Generation-evicted at 2×.
pub const DEFAULT_CAPACITY: usize = 4096;

/// A 128-bit fingerprint of an original message prefix. Two independent 64-bit `SipHash`
/// passes (the std default hasher, fixed-keyed so it is deterministic across calls within a
/// run) over salted input; a 128-bit width makes an accidental collision — which would splice
/// the wrong compressed content — not a practical concern even for billions of prefixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PrefixHash(u64, u64);

/// Incremental hasher over original message bytes, producing one [`PrefixHash`] per message
/// boundary. Feeding message `k`'s bytes after messages `0..k` yields `prefix_hash[k]`.
struct PrefixHasher {
    lo: std::collections::hash_map::DefaultHasher,
    hi: std::collections::hash_map::DefaultHasher,
}

impl PrefixHasher {
    /// `salt` scopes the whole chain to a compression context (provider kind + effective
    /// config): the same conversation compressed under a different preset/provider produces
    /// different bytes, so replaying across contexts would splice one preset's compression
    /// into another's output. Salting makes such entries simply not match (cold start on any
    /// config flip — correct, since byte-stability is only achievable within one context).
    fn new(salt: &[u8]) -> Self {
        let mut lo = std::collections::hash_map::DefaultHasher::new();
        let mut hi = std::collections::hash_map::DefaultHasher::new();
        salt.hash(&mut lo);
        salt.hash(&mut hi);
        // Salt the second pass so the two 64-bit halves are independent (else both halves are
        // equal and the key is effectively 64-bit).
        0xA5A5_5A5A_u64.hash(&mut hi);
        Self { lo, hi }
    }

    /// Fold one original message (its canonical JSON bytes) into the chain and read off the
    /// cumulative fingerprint through this message. A length prefix makes the boundary
    /// unambiguous, so concatenation can't alias (`["ab","c"]` ≠ `["a","bc"]`).
    fn push(&mut self, msg_bytes: &[u8]) -> PrefixHash {
        (msg_bytes.len() as u64).hash(&mut self.lo);
        msg_bytes.hash(&mut self.lo);
        (msg_bytes.len() as u64).hash(&mut self.hi);
        msg_bytes.hash(&mut self.hi);
        PrefixHash(self.lo.finish(), self.hi.finish())
    }
}

/// In-memory, size-capped map: original-message-prefix fingerprint → the compressed conversation
/// item llmtrim emitted for that message. Generation-evicted: when the live map reaches `2×`
/// cap it is demoted to a victim cache and a fresh map starts, bounding memory at ~`2×` cap
/// while keeping recently-seen prefixes hot (an entry promotes back on its next hit). No LRU
/// bookkeeping on the hot path — a single `len` check per insert.
pub struct Memo {
    cap: usize,
    inner: Mutex<Store>,
}

#[derive(Default)]
struct Store {
    live: HashMap<PrefixHash, Value>,
    /// The previous generation, consulted on a miss and promoted-from on a hit. Dropped
    /// wholesale when `live` rolls over again — this is the eviction.
    prev: HashMap<PrefixHash, Value>,
}

impl Memo {
    /// A memo holding up to ~`2 * cap` entries (live + one victim generation). `cap` of 0
    /// yields an inert memo that never reuses or stores (a hard off-switch).
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            cap,
            inner: Mutex::new(Store::default()),
        }
    }

    /// Look up a prefix fingerprint; promotes a victim-generation hit back into the live map.
    /// `None` on a cold prefix (the common first-turn case → full stateless compression).
    fn get(&self, key: PrefixHash) -> Option<Value> {
        let mut store = self.inner.lock().ok()?;
        if let Some(v) = store.live.get(&key) {
            return Some(v.clone());
        }
        if let Some(v) = store.prev.get(&key).cloned() {
            store.live.insert(key, v.clone());
            return Some(v);
        }
        None
    }

    /// Record a prefix fingerprint → its compressed conversation item. **First write wins:**
    /// once a prefix has been forwarded, later turns must not overwrite it with a divergent
    /// compression (e.g. when a freeze was discarded by the token gate and a fresh body was
    /// selected). Rolls a new generation (and drops the oldest) once the live map fills.
    fn put(&self, key: PrefixHash, content: Value) {
        if self.cap == 0 {
            return;
        }
        let Ok(mut store) = self.inner.lock() else {
            return;
        };
        // Sticky freeze: never replace an already-recorded prefix. Promote a victim hit
        // back to live so generation eviction does not thrash hot sessions, but keep bytes.
        if store.live.contains_key(&key) {
            return;
        }
        if let Some(prev) = store.prev.remove(&key) {
            store.live.insert(key, prev);
            return;
        }
        if store.live.len() >= self.cap {
            let full = std::mem::take(&mut store.live);
            store.prev = full;
        }
        store.live.insert(key, content);
    }

    /// Number of distinct prefixes currently retained (live ∪ victim). For tests/observability.
    pub fn len(&self) -> usize {
        let Ok(store) = self.inner.lock() else {
            return 0;
        };
        let mut n = store.live.len();
        for k in store.prev.keys() {
            if !store.live.contains_key(k) {
                n += 1;
            }
        }
        n
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Canonical bytes of one original message, for hashing. `serde_json` serializes object keys in
/// insertion order and reproduces the parsed value faithfully, and both the original and the
/// next turn's prefix come from the *same* client serializing the *same* retained history — so
/// byte-for-byte stability across turns holds without a canonicalizer. (A mismatch only costs a
/// cache miss → fallback, never correctness.)
fn message_bytes(msg: &Value) -> Vec<u8> {
    serde_json::to_vec(msg).unwrap_or_default()
}

/// The conversation array (`messages` / `input` / `contents`) and the key under which it lives,
/// or `None` for a shape with no recognizable turn array (→ no memo, full stateless path).
fn conversation(req: &Value) -> Option<(&'static str, &Vec<Value>)> {
    for key in ["messages", "input", "contents"] {
        if let Some(arr) = req.get(key).and_then(Value::as_array) {
            return Some((key, arr));
        }
    }
    None
}

/// Per-message compressed conversation item, keyed by original-prefix fingerprint, harvested
/// from a freshly compressed request so it can be replayed verbatim next turn. Pairs each entry
/// with the original message index it came from, so the caller can both store it and (on the
/// reuse path) overwrite the matching slot. Items are stored whole so Responses
/// `function_call_output.output` / `function_call.arguments` freeze the same way chat
/// `message.content` does — a `content`-only memo silently dropped the OMP/Grok agent path.
struct PrefixPlan {
    /// `(original_index, prefix_hash, compressed_item)` for every conversation message.
    entries: Vec<(usize, PrefixHash, Value)>,
    /// Index offset from original messages to compressed-output messages: `1` when a leading
    /// `system` message was injected (so original `k` lives at compressed `k + 1`), else `0`.
    offset: usize,
    /// The conversation array key in the compressed output (`messages` / `input` / `contents`).
    key: &'static str,
    /// Sticky top-level envelope fields (not in the conversation array) that llmtrim may
    /// reshape — primarily Responses `instructions` and `tools`. Keyed by conversation
    /// identity so a freeze restores them with the prefix; without this, output-control
    /// appends mid-session and busts the provider cache at byte 0 even when history is frozen.
    envelope_key: Option<PrefixHash>,
}

/// Build the [`PrefixPlan`] linking each original message to the compressed item at its
/// aligned slot. Returns `None` (→ fallback) if either side lacks a conversation array or the
/// arrays don't align by a 0/1 leading-system offset.
fn plan(salt: &[u8], original: &Value, compressed: &Value) -> Option<PrefixPlan> {
    let (_, orig_msgs) = conversation(original)?;
    let (key, comp_msgs) = conversation(compressed)?;

    // The only structural change a stage makes to the array is prepending ONE leading `system`
    // message (output-control / n-gram legend, when there wasn't already a leading system one).
    // Every conversation turn keeps its relative order and content slot. So the compressed array
    // is either the same length, or exactly one longer with a fresh leading system message. Any
    // other delta means a stage reshaped the array in a way we don't model — bail to fallback.
    let role0 = |arr: &[Value]| -> Option<String> {
        arr.first()
            .and_then(|m| m.get("role"))
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let offset = match comp_msgs.len().checked_sub(orig_msgs.len()) {
        Some(0) => 0usize,
        Some(1)
            if role0(comp_msgs).as_deref() == Some("system")
                && role0(orig_msgs).as_deref() != Some("system") =>
        {
            1
        }
        _ => return None,
    };

    let mut hasher = PrefixHasher::new(salt);
    let mut entries = Vec::with_capacity(orig_msgs.len());
    for orig in orig_msgs.iter() {
        let i = entries.len();
        let h = hasher.push(&message_bytes(orig));
        // Whole compressed item at the aligned slot (not just `content`). Chat messages put
        // text under `content`; OpenAI Responses agent turns put tool results under `output`
        // and call args under `arguments` — freezing only `content` left those items out of
        // the memo and re-broke the provider cache on every subsequent turn.
        if let Some(item) = comp_msgs.get(i + offset) {
            entries.push((i, h, item.clone()));
        } else {
            // A hole would break the contiguous-prefix invariant; stop so we never reuse past it.
            break;
        }
    }
    // Envelope: top-level fields outside the turn array. Hash identity from salt + first
    // original message so the same conversation reuses one sticky envelope; a new chat
    // (different first message) starts clean.
    let envelope_key = orig_msgs.first().map(|first| {
        let mut eh = PrefixHasher::new(salt);
        eh.push(b"envelope|");
        eh.push(&message_bytes(first))
    });
    Some(PrefixPlan {
        entries,
        offset,
        key,
        envelope_key,
    })
}

/// Reuse + record around a freshly compressed request, in place. Given the **original** request
/// JSON and the pipeline's **compressed** output JSON, this:
///
/// 1. finds the longest original-message prefix already in `memo`,
/// 2. overwrites those conversation slots in `compressed` with the stored (last-turn) items —
///    making the frozen prefix byte-identical to last turn's output (provider cache hit),
/// 3. records this turn's `(prefix_hash -> compressed item)` for every conversation message,
///    so next turn can freeze one further.
///
/// Returns the number of prefix messages whose content was reused verbatim (0 = nothing
/// reused, i.e. behavior identical to no memo). Pure and synchronous; never panics; on any
/// structural surprise it makes no change and returns 0 (full stateless fallback).
pub fn apply(memo: &Memo, salt: &[u8], original: &Value, compressed: &mut Value) -> usize {
    let reused = replay(memo, salt, original, compressed);
    record(memo, salt, original, compressed);
    reused
}

/// Replay an already-recorded contiguous prefix without mutating the memo. This split lets callers
/// decide whether the resulting request is actually forwarded before committing new entries.
pub fn replay(memo: &Memo, salt: &[u8], original: &Value, compressed: &mut Value) -> usize {
    if memo.cap == 0 {
        return 0;
    }
    let Some(plan) = plan(salt, original, compressed) else {
        return 0;
    };
    let mut reused: Vec<(usize, Value)> = Vec::new();
    for (idx, h, _) in &plan.entries {
        match memo.get(*h) {
            Some(stored) => reused.push((*idx, stored)),
            None => break,
        }
    }
    let reused_count = reused.len();
    if reused_count > 0
        && let Some(comp_msgs) = compressed.get_mut(plan.key).and_then(Value::as_array_mut)
    {
        for (idx, stored) in reused {
            if let Some(slot) = comp_msgs.get_mut(idx + plan.offset) {
                // Replace the whole item so tool-result `output`, function-call `arguments`,
                // and chat `content` all stay byte-identical to the prior turn.
                *slot = stored;
            }
        }
        // Restore sticky envelope (instructions/tools/system) from the first forward of
        // this conversation. Output-control and tool_trim are deterministic *per request*
        // but still produce different top-level bytes across turns (e.g. first-turn-only
        // directives); without this the history freeze is wasted at the prompt head.
        if let Some(ek) = plan.envelope_key
            && let Some(env) = memo.get(ek)
            && let Some(obj) = env.as_object()
            && let Some(root) = compressed.as_object_mut()
        {
            for (k, v) in obj {
                root.insert(k.clone(), v.clone());
            }
        }
    }
    reused_count
}

/// Record exactly the conversation items that the caller selected for forwarding.
pub fn record(memo: &Memo, salt: &[u8], original: &Value, forwarded: &Value) {
    if memo.cap == 0 {
        return;
    }
    let Some(plan) = plan(salt, original, forwarded) else {
        return;
    };
    for (idx, h, fresh_item) in plan.entries {
        // Prefer the post-selection item (may differ from the pre-gate compress when a
        // caller swaps bodies). Fall back to the planned item if the slot vanished.
        let to_store = forwarded
            .get(plan.key)
            .and_then(Value::as_array)
            .and_then(|a| a.get(idx + plan.offset))
            .cloned()
            .unwrap_or(fresh_item);
        memo.put(h, to_store);
    }
    if let Some(ek) = plan.envelope_key {
        // Rebuild envelope from the *forwarded* body (post-gates), first-write-wins via put.
        let mut env = serde_json::Map::new();
        if let Some(v) = forwarded.get("instructions") {
            env.insert("instructions".into(), v.clone());
        }
        if let Some(v) = forwarded.get("tools") {
            env.insert("tools".into(), v.clone());
        }
        if let Some(v) = forwarded.get("system") {
            env.insert("system".into(), v.clone());
        }
        if !env.is_empty() {
            memo.put(ek, Value::Object(env));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Stand-in for "the pipeline": prune each user message's content to its first sentence,
    /// *biased by the last (query) message* — a deliberately context-sensitive transform, like
    /// retrieve. This makes an OLD message's "compressed" form depend on the NEW turn, which is
    /// exactly the divergence the memo neutralizes.
    fn fake_compress(req: &Value) -> Value {
        let mut out = req.clone();
        let query = req
            .get("messages")
            .and_then(Value::as_array)
            .and_then(|a| a.last())
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if let Some(msgs) = out.get_mut("messages").and_then(Value::as_array_mut) {
            for m in msgs.iter_mut() {
                if let Some(c) = m.get("content").and_then(Value::as_str) {
                    // Context-sensitive: keep the first sentence, then append how many chars the
                    // CURRENT query has — so the "compression" of every message shifts when the
                    // last turn changes (divergent prefix, the real-world cache-buster).
                    let first = c.split('.').next().unwrap_or(c).to_string();
                    let shaped = format!("{first} <q{}>", query.len());
                    if let Some(obj) = m.as_object_mut() {
                        obj.insert("content".to_string(), Value::String(shaped));
                    }
                }
            }
        }
        out
    }

    fn user(content: &str) -> Value {
        json!({"role": "user", "content": content})
    }

    /// Full compressed bytes of the conversation messages `0..n`, for prefix-identity asserts.
    fn prefix_contents(compressed: &Value, n: usize) -> Vec<String> {
        compressed
            .get("messages")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .take(n)
            .map(|m| m.get("content").unwrap().to_string())
            .collect()
    }

    #[test]
    fn different_salt_never_reuses_across_contexts() {
        // Same conversation, different compression context (auto-routing flipped the preset,
        // or the provider kind changed): the fingerprint chain is salted with the context, so
        // one context's entries must never splice into another's output — cold start instead.
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let a = json!({"messages": [
            user("the revenue report grew across all regions. lots of detail here"),
            user("what was the revenue"),
        ]});
        let mut ca = fake_compress(&a);
        assert_eq!(apply(&memo, b"ctx-rag", &a, &mut ca), 0); // records under one context
        let b = json!({"messages": [
            user("the revenue report grew across all regions. lots of detail here"),
            user("what was the revenue"),
            user("now also tell me about costs"),
        ]});
        let mut cb = fake_compress(&b);
        assert_eq!(
            apply(&memo, b"ctx-agent", &b, &mut cb),
            0,
            "a different context salt must not reuse the other context's bytes"
        );
        // Same context still works (the salt isn't accidentally over-invalidating).
        let mut cb2 = fake_compress(&b);
        assert_eq!(apply(&memo, b"ctx-rag", &b, &mut cb2), 2);
    }

    #[test]
    fn replay_is_transactional_until_selected_body_is_recorded() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let salt = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_le_bytes();
        let a = json!({"messages": [user("context one. detail"), user("question one")]});
        let mut ca = fake_compress(&a);
        assert_eq!(replay(&memo, &salt, &a, &mut ca), 0);
        assert!(memo.is_empty(), "a candidate replay must not record itself");
        record(&memo, &salt, &a, &ca);
        assert!(
            !memo.is_empty(),
            "the selected wire body is committed explicitly"
        );

        let b = json!({"messages": [
            user("context one. detail"),
            user("question one"),
            user("new appended turn")
        ]});
        let mut cb = fake_compress(&b);
        assert_eq!(replay(&memo, &salt, &b, &mut cb), 2);
    }

    #[test]
    fn headline_two_turn_prefix_is_byte_identical() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);

        // Turn A: two messages.
        let a = json!({"messages": [
            user("the revenue report grew across all regions. lots of detail here"),
            user("what was the revenue"),
        ]});
        let mut ca = fake_compress(&a);
        // First turn: nothing to reuse, but it records A's prefix.
        assert_eq!(apply(&memo, b"t", &a, &mut ca), 0);
        let a_msg0 = prefix_contents(&ca, 1);

        // Turn B = A + one appended user turn (the agent-loop shape).
        let b = json!({"messages": [
            user("the revenue report grew across all regions. lots of detail here"),
            user("what was the revenue"),
            user("now also tell me about costs and the very long winded margin analysis"),
        ]});
        let cb_no_memo = fake_compress(&b);
        // Sanity: WITHOUT the memo, message 0 diverges between turns (context-sensitive).
        let b_msg0_fresh = prefix_contents(&cb_no_memo, 1);
        assert_ne!(
            a_msg0, b_msg0_fresh,
            "precondition: the stateless compressor diverges on the old message across turns \
             (otherwise the memo would be testing nothing)"
        );

        // WITH the memo: the two shared messages reuse turn A's bytes verbatim.
        let mut cb = fake_compress(&b);
        let reused = apply(&memo, b"t", &b, &mut cb);
        assert_eq!(reused, 2, "both shared messages frozen from turn A");

        // THE HEADLINE PROPERTY: every compressed byte of A's messages inside B equals A's.
        assert_eq!(
            prefix_contents(&ca, 2),
            prefix_contents(&cb, 2),
            "frozen prefix must be byte-identical to last turn (provider cache stays warm)"
        );
        // And the new suffix message is this turn's fresh compression (not frozen).
        let suffix = cb.get("messages").and_then(Value::as_array).unwrap()[2]
            .get("content")
            .unwrap()
            .to_string();
        assert!(
            suffix.contains("costs"),
            "the new turn carries fresh content: {suffix}"
        );
    }

    #[test]
    fn third_turn_extends_the_frozen_prefix_transitively() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let base = vec![
            user("alpha context paragraph one. detail detail detail"),
            user("beta question about alpha"),
        ];

        let a = json!({ "messages": base });
        let mut ca = fake_compress(&a);
        apply(&memo, b"t", &a, &mut ca);

        let mut b_msgs = base.clone();
        b_msgs.push(user("gamma follow up question number two"));
        let b = json!({ "messages": b_msgs.clone() });
        let mut cb = fake_compress(&b);
        assert_eq!(apply(&memo, b"t", &b, &mut cb), 2);

        let mut c_msgs = b_msgs.clone();
        c_msgs.push(user(
            "delta a third follow up that is appended at the very end",
        ));
        let c = json!({ "messages": c_msgs });
        let mut cc = fake_compress(&c);
        // Turn C freezes all THREE earlier messages (the prefix grew by one each turn).
        assert_eq!(
            apply(&memo, b"t", &c, &mut cc),
            3,
            "the frozen prefix extends transitively as the conversation grows"
        );
        // Transitive identity: C's first 3 messages == B's first 3 (and B's first 2 == A's).
        assert_eq!(prefix_contents(&cc, 3), prefix_contents(&cb, 3));
        assert_eq!(prefix_contents(&cb, 2), prefix_contents(&ca, 2));
    }

    #[test]
    fn divergent_history_does_not_reuse() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let a = json!({"messages": [
            user("original first message about the budget"),
            user("a question"),
        ]});
        let mut ca = fake_compress(&a);
        apply(&memo, b"t", &a, &mut ca);

        // Same prefix LENGTH, but one byte changed in the OLD (first) message → the prefix
        // fingerprint diverges at message 0, so nothing reuses; fresh compression, no panic.
        let b = json!({"messages": [
            user("original first message about the BUDGET"), // one byte differs
            user("a question"),
            user("a new appended turn"),
        ]});
        let mut cb = fake_compress(&b);
        assert_eq!(
            apply(&memo, b"t", &b, &mut cb),
            0,
            "a changed old message busts the prefix → no reuse (correctness over caching)"
        );
        // The first message is B's own fresh compression, untouched by A's stored bytes.
        let fresh = fake_compress(&b);
        assert_eq!(prefix_contents(&cb, 2), prefix_contents(&fresh, 2));
    }

    #[test]
    fn appended_turn_after_changed_prefix_does_not_reuse_a_later_match() {
        // The second message is shared+identical even though the first diverged → reuse must be
        // a *contiguous prefix from the front*, so a divergence at message 0 blocks message 1
        // too (the provider cache is prefix-keyed: a busted byte 0 invalidates everything after).
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let a = json!({"messages": [user("AAAA first"), user("BBBB second shared verbatim")]});
        let mut ca = fake_compress(&a);
        apply(&memo, b"t", &a, &mut ca);

        let b =
            json!({"messages": [user("ZZZZ first changed"), user("BBBB second shared verbatim")]});
        let mut cb = fake_compress(&b);
        assert_eq!(
            apply(&memo, b"t", &b, &mut cb),
            0,
            "message 1 is identical but message 0 diverged → no contiguous prefix from the front"
        );
    }

    #[test]
    fn memory_cap_evicts_and_stays_bounded() {
        // cap=4 ⇒ at most 2×cap=8 entries retained across the live + victim generations.
        let memo = Memo::with_capacity(4);
        for i in 0..1000 {
            // Each request is a unique single-message conversation → one new prefix entry.
            let req = json!({"messages": [user(&format!("unique conversation number {i}"))]});
            let mut c = fake_compress(&req);
            apply(&memo, b"t", &req, &mut c);
        }
        assert!(
            memo.len() <= 8,
            "generation eviction bounds the memo at 2×cap; got {}",
            memo.len()
        );
        assert!(!memo.is_empty(), "but it is not empty after inserts");
    }

    #[test]
    fn zero_capacity_is_an_inert_off_switch() {
        let memo = Memo::with_capacity(0);
        let a = json!({"messages": [user("first"), user("second")]});
        let mut ca = fake_compress(&a);
        assert_eq!(apply(&memo, b"t", &a, &mut ca), 0);
        let b = json!({"messages": [user("first"), user("second"), user("third")]});
        let mut cb = fake_compress(&b);
        assert_eq!(
            apply(&memo, b"t", &b, &mut cb),
            0,
            "cap 0 never reuses or stores — a hard off-switch (flag off ⇒ stateless behavior)"
        );
        assert!(memo.is_empty());
    }

    #[test]
    fn leading_system_injection_offsets_alignment() {
        // The compressed output has a freshly INJECTED leading `system` message (as Stage F /
        // the n-gram legend do): original message `k` then lives at compressed slot `k + 1`.
        // The memo must align across that offset and still freeze the right conversation turns.
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);

        // A compressor that prunes content AND prepends a system instruction (index shift).
        let compress_with_system = |req: &Value| -> Value {
            let mut out = fake_compress(req);
            if let Some(msgs) = out.get_mut("messages").and_then(Value::as_array_mut) {
                msgs.insert(0, json!({"role": "system", "content": "be terse"}));
            }
            out
        };

        let a = json!({"messages": [user("context here. plenty of it"), user("the query")]});
        let mut ca = compress_with_system(&a);
        assert_eq!(apply(&memo, b"t", &a, &mut ca), 0);

        let b = json!({"messages": [
            user("context here. plenty of it"),
            user("the query"),
            user("a brand new appended turn changing the query bias entirely"),
        ]});
        let mut cb = compress_with_system(&b);
        assert_eq!(
            apply(&memo, b"t", &b, &mut cb),
            2,
            "alignment across the injected leading system message freezes both shared turns"
        );
        // Compressed slots 1..=2 (the conversation turns after the injected system) match A's.
        let conv = |c: &Value| -> Vec<String> {
            c.get("messages").and_then(Value::as_array).unwrap()[1..=2]
                .iter()
                .map(|m| m.get("content").unwrap().to_string())
                .collect()
        };
        assert_eq!(
            conv(&ca),
            conv(&cb),
            "frozen turns byte-identical across the offset"
        );
        // The injected system message itself is fresh each turn (never frozen), as it must be.
        assert_eq!(
            cb.get("messages").and_then(Value::as_array).unwrap()[0]
                .get("content")
                .unwrap(),
            "be terse"
        );
    }

    #[test]
    fn unrecognized_shape_falls_back_without_panic() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        // No `messages` / `input` / `contents` array → no memo, no change, no panic.
        let weird = json!({"prompt": "just a string completion request", "max_tokens": 5});
        let mut c = weird.clone();
        assert_eq!(apply(&memo, b"t", &weird, &mut c), 0);
        assert_eq!(c, weird, "untouched when there's no conversation array");
    }

    #[test]
    fn message_count_mismatch_falls_back() {
        // If the compressed output's array differs from the original by something other than a
        // single injected leading system message, we can't align slots → no reuse, no panic.
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let original = json!({"messages": [user("a"), user("b"), user("c")]});
        // Compressed output dropped a message (no stage does this, but guard it anyway).
        let mut compressed = json!({"messages": [user("a"), user("c")]});
        assert_eq!(apply(&memo, b"t", &original, &mut compressed), 0);
    }

    #[test]
    fn responses_input_shape_is_supported() {
        // The OpenAI Responses wire shape keys its turns under `input`, not `messages`.
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let a = json!({"input": [
            {"role": "user", "content": "first turn long context here"},
            {"role": "user", "content": "the query"},
        ]});
        // A compressor that just tags content (context-free here; we only test array plumbing).
        let comp = |req: &Value| -> Value {
            let mut out = req.clone();
            if let Some(arr) = out.get_mut("input").and_then(Value::as_array_mut) {
                for (i, m) in arr.iter_mut().enumerate() {
                    if let Some(obj) = m.as_object_mut() {
                        obj.insert("content".to_string(), json!(format!("c{i}")));
                    }
                }
            }
            out
        };
        let mut ca = comp(&a);
        assert_eq!(apply(&memo, b"t", &a, &mut ca), 0);

        let b = json!({"input": [
            {"role": "user", "content": "first turn long context here"},
            {"role": "user", "content": "the query"},
            {"role": "user", "content": "appended turn"},
        ]});
        let mut cb = comp(&b);
        assert_eq!(
            apply(&memo, b"t", &b, &mut cb),
            2,
            "the `input` (Responses) shape is memoized like `messages`"
        );
    }

    /// OMP / Grok (and Codex agent) wire tool results as Responses `function_call_output`
    /// items with text under `output`, not `content`. The memo must freeze the whole item or
    /// every subsequent turn rewrites earlier toolouts and busts the provider prefix cache.
    #[test]
    fn responses_function_call_output_freezes_across_turns() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);

        let find_toolout = |req: &Value, call_id: &str| -> Value {
            req.get("input")
                .and_then(Value::as_array)
                .unwrap()
                .iter()
                .find(|m| {
                    m.get("type").and_then(Value::as_str) == Some("function_call_output")
                        && m.get("call_id").and_then(Value::as_str) == Some(call_id)
                })
                .cloned()
                .unwrap_or_else(|| panic!("missing function_call_output {call_id}"))
        };

        let tool_out_v1 = format!(
            "{}{}",
            "ERROR boom\n".repeat(40),
            "INFO noise\n".repeat(200)
        );
        let a = json!({
            "input": [
                {"role":"user","content":[{"type":"input_text","text":"look at the log"}]},
                {"type":"function_call","call_id":"c1","name":"bash","arguments":"{\"command\":\"cat log\"}"},
                {"type":"function_call_output","call_id":"c1","output": tool_out_v1},
                {"role":"user","content":[{"type":"input_text","text":"why did it fail?"}]},
            ]
        });

        // Context-sensitive "compressor": rewrite every function_call_output.output using the
        // *last* item's length, so without the memo the old toolout would diverge each turn.
        let comp = |req: &Value| -> Value {
            let mut out = req.clone();
            let qlen = out
                .get("input")
                .and_then(Value::as_array)
                .and_then(|arr| arr.last())
                .map(|m| m.to_string().len())
                .unwrap_or(0);
            if let Some(arr) = out.get_mut("input").and_then(Value::as_array_mut) {
                for m in arr.iter_mut() {
                    let is_toolout =
                        m.get("type").and_then(Value::as_str) == Some("function_call_output");
                    if !is_toolout {
                        continue;
                    }
                    if let Some(obj) = m.as_object_mut() {
                        let raw = obj
                            .get("output")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let kept = raw.lines().take(3).collect::<Vec<_>>().join("\n");
                        obj.insert("output".into(), json!(format!("{kept}\n[trimmed q{qlen}]")));
                    }
                }
            }
            out
        };

        let mut ca = comp(&a);
        assert_eq!(apply(&memo, b"omp-grok", &a, &mut ca), 0, "cold prefix");
        let frozen_c1 = find_toolout(&ca, "c1");

        let b = json!({
            "input": [
                {"role":"user","content":[{"type":"input_text","text":"look at the log"}]},
                {"type":"function_call","call_id":"c1","name":"bash","arguments":"{\"command\":\"cat log\"}"},
                {"type":"function_call_output","call_id":"c1","output": tool_out_v1},
                {"role":"user","content":[{"type":"input_text","text":"why did it fail?"}]},
                {"type":"function_call","call_id":"c2","name":"bash","arguments":"{\"command\":\"ls\"}"},
                {"type":"function_call_output","call_id":"c2","output":"file1\nfile2\n"},
                {"role":"user","content":[{"type":"input_text","text":"continue investigating with more context please"}]},
            ]
        });
        let mut cb = comp(&b);
        assert_ne!(
            find_toolout(&cb, "c1").get("output"),
            frozen_c1.get("output"),
            "sanity: context-sensitive compress would diverge without memo"
        );

        let reused = apply(&memo, b"omp-grok", &b, &mut cb);
        assert!(
            reused >= 3,
            "shared prefix including function_call_output must freeze, got {reused}"
        );
        assert_eq!(
            find_toolout(&cb, "c1"),
            frozen_c1,
            "function_call_output item must be byte-identical to the prior turn"
        );
        let frozen_c2 = find_toolout(&cb, "c2");

        let c = json!({
            "input": [
                {"role":"user","content":[{"type":"input_text","text":"look at the log"}]},
                {"type":"function_call","call_id":"c1","name":"bash","arguments":"{\"command\":\"cat log\"}"},
                {"type":"function_call_output","call_id":"c1","output": tool_out_v1},
                {"role":"user","content":[{"type":"input_text","text":"why did it fail?"}]},
                {"type":"function_call","call_id":"c2","name":"bash","arguments":"{\"command\":\"ls\"}"},
                {"type":"function_call_output","call_id":"c2","output":"file1\nfile2\n"},
                {"role":"user","content":[{"type":"input_text","text":"continue investigating with more context please"}]},
                {"role":"user","content":[{"type":"input_text","text":"one more question about the same logs"}]},
            ]
        });
        let mut cc = comp(&c);
        let reused_c = apply(&memo, b"omp-grok", &c, &mut cc);
        assert!(
            reused_c >= 6,
            "prefix through c2 must freeze on turn C, got {reused_c}"
        );
        assert_eq!(
            find_toolout(&cc, "c1"),
            frozen_c1,
            "c1 must never remutate after first forward"
        );
        assert_eq!(
            find_toolout(&cc, "c2"),
            frozen_c2,
            "c2 must never remutate after first forward"
        );
    }

    /// Even if a later turn re-records the same original prefix with different compressed
    /// bytes (e.g. freeze discarded by a token gate), the memo must keep the first forward.
    #[test]
    fn first_write_wins_never_overwrites_frozen_prefix() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let a = json!({"input": [
            {"role":"user","content":"stable history blob one"},
            {"role":"user","content":"query a"},
        ]});
        let mut ca = json!({"input": [
            {"role":"user","content":"COMPRESSED-A"},
            {"role":"user","content":"query a"},
        ]});
        assert_eq!(apply(&memo, b"t", &a, &mut ca), 0);

        // Same original prefix, different compressed bytes for message 0.
        let b = json!({"input": [
            {"role":"user","content":"stable history blob one"},
            {"role":"user","content":"query a"},
            {"role":"user","content":"query b appended"},
        ]});
        let mut cb = json!({"input": [
            {"role":"user","content":"COMPRESSED-B-DIVERGENT"},
            {"role":"user","content":"query a"},
            {"role":"user","content":"query b appended"},
        ]});
        let reused = apply(&memo, b"t", &b, &mut cb);
        assert!(reused >= 1, "prefix message 0 must hit memo, got {reused}");
        assert_eq!(
            cb["input"][0]["content"], "COMPRESSED-A",
            "first forwarded compression must stick forever"
        );
        // And recording the divergent body must not poison the memo for turn C.
        let mut cc = json!({"input": [
            {"role":"user","content":"COMPRESSED-C-ALSO-DIVERGENT"},
            {"role":"user","content":"query a"},
            {"role":"user","content":"query b appended"},
            {"role":"user","content":"query c"},
        ]});
        let c = json!({"input": [
            {"role":"user","content":"stable history blob one"},
            {"role":"user","content":"query a"},
            {"role":"user","content":"query b appended"},
            {"role":"user","content":"query c"},
        ]});
        apply(&memo, b"t", &c, &mut cc);
        assert_eq!(cc["input"][0]["content"], "COMPRESSED-A");
    }

    /// Top-level `instructions` (Responses system prompt) must freeze with the conversation.
    /// Output-control appends mid-session otherwise bust the cache at the prompt head even
    /// when every `input[]` item is byte-stable.
    #[test]
    fn instructions_envelope_freezes_across_turns() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let a = json!({
            "instructions": "SYSTEM-A",
            "tools": [{"name":"bash","description":"run"}],
            "input": [
                {"role":"user","content":"hello world context"},
                {"type":"function_call_output","call_id":"c1","output":"log line\n".repeat(50)},
            ]
        });
        // Fresh compress "grows" instructions and trims toolout using query length.
        let mut ca = a.clone();
        ca["instructions"] = json!("SYSTEM-A\nBe concise appended");
        ca["tools"] = json!([{"name":"bash","description":"r"}]);
        ca["input"][1]["output"] = json!("log line\nlog line\n[trim]");
        assert_eq!(apply(&memo, b"env", &a, &mut ca), 0);
        let frozen_instr = ca["instructions"].clone();
        let frozen_tools = ca["tools"].clone();
        let frozen_out = ca["input"][1].clone();

        let b = json!({
            "instructions": "SYSTEM-A",
            "tools": [{"name":"bash","description":"run"}],
            "input": [
                {"role":"user","content":"hello world context"},
                {"type":"function_call_output","call_id":"c1","output":"log line\n".repeat(50)},
                {"role":"user","content":"follow up with a much longer query that would change trims"},
            ]
        });
        let mut cb = b.clone();
        cb["instructions"] = json!("SYSTEM-A\nBe concise appended\nEXTRA-SHOULD-NOT-STICK");
        cb["tools"] = json!([{"name":"bash","description":"DIFFERENT"}]);
        cb["input"][1]["output"] = json!("totally different trim");
        let reused = apply(&memo, b"env", &b, &mut cb);
        assert!(reused >= 1, "history must freeze, got {reused}");
        assert_eq!(cb["instructions"], frozen_instr, "instructions must stick");
        assert_eq!(cb["tools"], frozen_tools, "tools must stick");
        assert_eq!(cb["input"][1], frozen_out, "toolout item must stick");
    }

    // ---------------------------------------------------------------------
    // Cache-stability contract suite
    //
    // These tests encode the product rule: once a conversation prefix (history
    // items + envelope) has been forwarded, later turns MUST emit those bytes
    // unchanged. Any future "optimization" that rewrites old toolouts,
    // instructions, tools, or function_call args should fail this suite.
    // ---------------------------------------------------------------------

    /// Context-sensitive compressor stand-in: every compressible field is tagged
    /// with the *current* turn count so a naive per-request compress diverges.
    fn divergent_compress(req: &Value, turn: usize) -> Value {
        let mut out = req.clone();
        // Stamp turn markers onto every field a real compressor might touch. Clone-then-write
        // avoids holding a borrow across the mutable insert.
        if let Some(s) = out
            .get("instructions")
            .and_then(Value::as_str)
            .map(str::to_string)
        {
            out.as_object_mut()
                .unwrap()
                .insert("instructions".into(), json!(format!("{s}|t{turn}")));
        }
        if let Some(s) = out
            .get("system")
            .and_then(Value::as_str)
            .map(str::to_string)
        {
            out.as_object_mut()
                .unwrap()
                .insert("system".into(), json!(format!("{s}|t{turn}")));
        }
        if let Some(tools) = out.get_mut("tools").and_then(Value::as_array_mut) {
            for tool in tools.iter_mut() {
                if let Some(obj) = tool.as_object_mut() {
                    let d = obj
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    obj.insert("description".into(), json!(format!("{d}|t{turn}")));
                }
            }
        }
        let key = if out.get("input").is_some() {
            "input"
        } else {
            "messages"
        };
        if let Some(arr) = out.get_mut(key).and_then(Value::as_array_mut) {
            for item in arr.iter_mut() {
                let Some(obj) = item.as_object_mut() else {
                    continue;
                };
                if let Some(o) = obj
                    .get("output")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                {
                    obj.insert("output".into(), json!(format!("{o}|t{turn}")));
                }
                if let Some(a) = obj
                    .get("arguments")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                {
                    obj.insert("arguments".into(), json!(format!("{a}|t{turn}")));
                }
                let content = obj.get("content").cloned();
                if let Some(c) = content {
                    if let Some(s) = c.as_str() {
                        obj.insert("content".into(), json!(format!("{s}|t{turn}")));
                    } else if let Some(parts) = c.as_array() {
                        let mut new_parts = Vec::new();
                        for p in parts {
                            if let Some(po) = p.as_object() {
                                let mut np = po.clone();
                                if let Some(tx) =
                                    np.get("text").and_then(Value::as_str).map(str::to_string)
                                {
                                    np.insert("text".into(), json!(format!("{tx}|t{turn}")));
                                }
                                if let Some(inner) = np
                                    .get("content")
                                    .and_then(Value::as_str)
                                    .map(str::to_string)
                                {
                                    np.insert("content".into(), json!(format!("{inner}|t{turn}")));
                                }
                                new_parts.push(Value::Object(np));
                            } else {
                                new_parts.push(p.clone());
                            }
                        }
                        obj.insert("content".into(), Value::Array(new_parts));
                    }
                }
            }
        }
        out
    }

    /// Full OMP-shaped multi-turn contract: after turn 1 is recorded, turns 2..N must keep
    /// the entire prior prefix (envelope + every prior input item) byte-identical.
    #[test]
    fn cache_stability_contract_omp_multiturn_never_rewrites_prefix() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let salt = b"cache-contract-omp";

        // Growing original conversation (what the client resends).
        let mut original_items = vec![
            json!({"role":"user","content":[{"type":"input_text","text":"debug this failure"}]}),
            json!({"type":"function_call","call_id":"c1","name":"bash","arguments":"{\"command\":\"cat log\"}"}),
            json!({"type":"function_call_output","call_id":"c1","output":"ERROR boom\n".repeat(30)}),
        ];

        let base = |items: &[Value], turn_label: &str| {
            json!({
                "instructions": "You are omp",
                "tools": [
                    {"name":"bash","description":"run a shell command with a long schema text"},
                    {"name":"read","description":"read a file from disk carefully"},
                ],
                "input": items,
                "prompt_cache_key": "session-contract",
                "meta_turn": turn_label, // ignored by memo (not hashed); just for humans
            })
        };

        // Turn 1: cold
        let o1 = base(&original_items, "1");
        let mut c1 = divergent_compress(&o1, 1);
        assert_eq!(apply(&memo, salt, &o1, &mut c1), 0);
        let mut frozen_prefix = c1.clone();

        // Turns 2..6: append user/tool pairs; divergent compress would rewrite everything.
        for turn in 2..=6 {
            original_items.push(json!({
                "role":"user",
                "content":[{"type":"input_text","text": format!("follow-up {turn}")}],
            }));
            original_items.push(json!({
                "type":"function_call",
                "call_id": format!("c{turn}"),
                "name":"bash",
                "arguments": format!("{{\"command\":\"echo {turn}\"}}"),
            }));
            original_items.push(json!({
                "type":"function_call_output",
                "call_id": format!("c{turn}"),
                "output": format!("output for turn {turn}\n").repeat(10),
            }));

            let ot = base(&original_items, &turn.to_string());
            let mut ct = divergent_compress(&ot, turn);
            // Sanity: without memo the old toolout would differ
            assert_ne!(
                ct["input"][2], frozen_prefix["input"][2],
                "turn {turn}: divergent_compress must change history without memo"
            );
            let reused = apply(&memo, salt, &ot, &mut ct);
            assert!(
                reused >= 3,
                "turn {turn}: must reuse at least the initial 3-item prefix, got {reused}"
            );

            // CONTRACT: every previously frozen item + envelope stays identical.
            assert_eq!(
                ct["instructions"], frozen_prefix["instructions"],
                "turn {turn}: instructions rewritten — cache break at prompt head"
            );
            assert_eq!(
                ct["tools"], frozen_prefix["tools"],
                "turn {turn}: tools rewritten — cache break in tool prefix"
            );
            let n_frozen = frozen_prefix["input"].as_array().unwrap().len();
            for i in 0..n_frozen {
                assert_eq!(
                    &ct["input"][i], &frozen_prefix["input"][i],
                    "turn {turn}: input[{i}] rewritten — mid-history cache break"
                );
            }

            // Extend frozen snapshot to include newly committed suffix items for next loop.
            // After apply+record, the memo holds the frozen prefix; capture current full
            // forwarded body as the new freeze baseline for items 0..len.
            frozen_prefix = ct.clone();
            let _ = frozen_prefix; // used next iteration via reassignment below
            // actually need mut - fix by outer mut
            let _ = turn;
        }
    }

    /// `function_call.arguments` (not only function_call_output.output) must freeze.
    #[test]
    fn cache_stability_contract_function_call_arguments_freeze() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let a = json!({
            "input": [
                {"type":"function_call","call_id":"c1","name":"bash","arguments":"{\"command\":\"long args\"}"},
                {"type":"function_call_output","call_id":"c1","output":"ok"},
            ]
        });
        let mut ca = divergent_compress(&a, 1);
        apply(&memo, b"args", &a, &mut ca);
        let frozen_args = ca["input"][0].clone();

        let b = json!({
            "input": [
                {"type":"function_call","call_id":"c1","name":"bash","arguments":"{\"command\":\"long args\"}"},
                {"type":"function_call_output","call_id":"c1","output":"ok"},
                {"role":"user","content":"next"},
            ]
        });
        let mut cb = divergent_compress(&b, 2);
        assert!(apply(&memo, b"args", &b, &mut cb) >= 1);
        assert_eq!(
            cb["input"][0], frozen_args,
            "function_call item (arguments) must be frozen whole"
        );
    }

    /// Anthropic-style `system` + `messages` tool_result content must freeze.
    #[test]
    fn cache_stability_contract_anthropic_system_and_tool_result_freeze() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let a = json!({
            "system": "SYS",
            "messages": [
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"t1","content":"TOOLLOG\n".repeat(40)}
                ]},
                {"role":"user","content":"why?"},
            ]
        });
        let mut ca = divergent_compress(&a, 1);
        apply(&memo, b"anth", &a, &mut ca);
        let frozen_sys = ca["system"].clone();
        let frozen_msg0 = ca["messages"][0].clone();

        let b = json!({
            "system": "SYS",
            "messages": [
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"t1","content":"TOOLLOG\n".repeat(40)}
                ]},
                {"role":"user","content":"why?"},
                {"role":"user","content":"more?"},
            ]
        });
        let mut cb = divergent_compress(&b, 2);
        assert!(apply(&memo, b"anth", &b, &mut cb) >= 1);
        assert_eq!(cb["system"], frozen_sys, "system envelope must freeze");
        assert_eq!(
            cb["messages"][0], frozen_msg0,
            "tool_result message must freeze"
        );
    }

    /// Editing an old original message invalidates from that point; earlier prefix stays.
    #[test]
    fn cache_stability_contract_edit_invalidates_only_from_edit_point() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let a = json!({"input":[
            {"role":"user","content":"one"},
            {"role":"user","content":"two"},
            {"role":"user","content":"three"},
        ]});
        let mut ca = divergent_compress(&a, 1);
        apply(&memo, b"edit", &a, &mut ca);
        let frozen0 = ca["input"][0].clone();
        let frozen1 = ca["input"][1].clone();

        // Client edited message 1 ("two" -> "TWO-EDITED"); message 0 unchanged.
        let b = json!({"input":[
            {"role":"user","content":"one"},
            {"role":"user","content":"TWO-EDITED"},
            {"role":"user","content":"three"},
            {"role":"user","content":"four"},
        ]});
        let mut cb = divergent_compress(&b, 2);
        let reused = apply(&memo, b"edit", &b, &mut cb);
        assert_eq!(reused, 1, "only message 0 still matches the hash chain");
        assert_eq!(
            cb["input"][0], frozen0,
            "pre-edit prefix must remain frozen"
        );
        assert_ne!(
            cb["input"][1], frozen1,
            "edited message must NOT reuse stale compressed bytes"
        );
    }

    /// replay alone must not record; only record()/apply() commits. Prevents poisoning
    /// the memo with a body that was never forwarded.
    #[test]
    fn cache_stability_contract_unforwarded_replay_does_not_poison_memo() {
        let memo = Memo::with_capacity(DEFAULT_CAPACITY);
        let a = json!({"input":[{"role":"user","content":"x"}]});
        let mut ca = json!({"input":[{"role":"user","content":"X-FORWARDED"}]});
        apply(&memo, b"poison", &a, &mut ca);

        // Candidate compress for turn B (must keep array length aligned with original).
        let b = json!({"input":[
            {"role":"user","content":"x"},
            {"role":"user","content":"y"},
        ]});
        let mut bad = json!({"input":[
            {"role":"user","content":"X-BAD-NOT-FORWARDED"},
            {"role":"user","content":"Y-BAD"},
        ]});
        // replay only — must restore FORWARDED for msg0, and must not record BAD.
        let reused = replay(&memo, b"poison", &b, &mut bad);
        assert!(
            reused >= 1,
            "msg0 must hit memo on replay-only, got {reused}"
        );
        assert_eq!(bad["input"][0]["content"], "X-FORWARDED");

        // Explicit record of a *different* body must still first-write-win on msg0.
        let forwarded_bad = json!({"input":[
            {"role":"user","content":"X-BAD-NOT-FORWARDED"},
            {"role":"user","content":"Y-BAD"},
        ]});
        record(&memo, b"poison", &b, &forwarded_bad);

        let c_orig = json!({"input":[
            {"role":"user","content":"x"},
            {"role":"user","content":"y"},
            {"role":"user","content":"z"},
        ]});
        let mut c = divergent_compress(&c_orig, 9);
        apply(&memo, b"poison", &c_orig, &mut c);
        assert_eq!(
            c["input"][0]["content"], "X-FORWARDED",
            "first real forward must win over later record attempts"
        );
    }
}
