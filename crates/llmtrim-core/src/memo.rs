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
//!   **compressed `content` value** llmtrim emitted for original message `k` last time it was
//!   the head of a prefix. No prompt text is read back from anywhere on disk — see *Privacy*.
//! - **Reuse.** On a new request, walk the original messages front-to-back; the longest run
//!   `0..=m` whose every boundary hash is present in the store is the *frozen prefix*. We still
//!   run the normal full-request pipeline (so all legend/injection/Stage-A logic stays exactly
//!   correct), then overwrite the frozen-prefix messages' `content` in the compressed output
//!   with the stored bytes — making them identical to last turn's output, which is what the
//!   provider cache keys on. Only the *suffix* (new messages) carries this turn's fresh
//!   content compression; the input-token gate still governs whatever was freshly compressed.
//! - **Record.** After rewriting, store this request's `(prefix_hash[k] -> compressed content)`
//!   for every conversation message, so the next turn can freeze one message further.
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
//! `content` fragments — which are *already* in flight to the provider on this very request — so
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

/// In-memory, size-capped map: original-message-prefix fingerprint → the compressed `content`
/// value llmtrim emitted for that message. Generation-evicted: when the live map reaches `2×`
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

    /// Record a prefix fingerprint → its compressed `content`. Rolls a new generation (and
    /// drops the oldest) once the live map fills, keeping memory bounded.
    fn put(&self, key: PrefixHash, content: Value) {
        if self.cap == 0 {
            return;
        }
        let Ok(mut store) = self.inner.lock() else {
            return;
        };
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

/// Per-message compressed `content`, keyed by original-prefix fingerprint, harvested from a
/// freshly compressed request so it can be replayed verbatim next turn. Pairs each entry with
/// the original message index it came from, so the caller can both store it and (on the
/// reuse path) overwrite the matching slot.
struct PrefixPlan {
    /// `(original_index, prefix_hash, compressed_content)` for every conversation message.
    entries: Vec<(usize, PrefixHash, Value)>,
    /// Index offset from original messages to compressed-output messages: `1` when a leading
    /// `system` message was injected (so original `k` lives at compressed `k + 1`), else `0`.
    offset: usize,
    /// The conversation array key in the compressed output (`messages` / `input` / `contents`).
    key: &'static str,
}

/// Build the [`PrefixPlan`] linking each original message to the compressed `content` at its
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
        // The compressed content at the aligned slot. If the slot or its `content` is missing
        // (shouldn't happen given the length check, but never index blindly), skip this entry —
        // it just won't be memoized/reused, no panic.
        if let Some(content) = comp_msgs.get(i + offset).and_then(|m| m.get("content")) {
            entries.push((i, h, content.clone()));
        } else {
            // A hole would break the contiguous-prefix invariant; stop so we never reuse past it.
            break;
        }
    }
    Some(PrefixPlan {
        entries,
        offset,
        key,
    })
}

/// Reuse + record around a freshly compressed request, in place. Given the **original** request
/// JSON and the pipeline's **compressed** output JSON, this:
///
/// 1. finds the longest original-message prefix already in `memo`,
/// 2. overwrites those messages' `content` in `compressed` with the stored (last-turn) bytes —
///    making the frozen prefix byte-identical to last turn's output (provider cache hit),
/// 3. records this turn's `(prefix_hash -> compressed content)` for every conversation message,
///    so next turn can freeze one further.
///
/// Returns the number of prefix messages whose content was reused verbatim (0 = nothing
/// reused, i.e. behavior identical to no memo). Pure and synchronous; never panics; on any
/// structural surprise it makes no change and returns 0 (full stateless fallback).
pub fn apply(memo: &Memo, salt: &[u8], original: &Value, compressed: &mut Value) -> usize {
    if memo.cap == 0 {
        return 0;
    }
    let Some(plan) = plan(salt, original, compressed) else {
        return 0;
    };
    let offset = plan.offset;

    // Longest already-seen prefix: stored values to splice in, in original order. We stop at the
    // first cold boundary — reuse must be a contiguous prefix (the suffix is this turn's new
    // work), matching the provider cache's prefix semantics.
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
            if let Some(slot) = comp_msgs.get_mut(idx + offset)
                && let Some(obj) = slot.as_object_mut()
            {
                obj.insert("content".to_string(), stored);
            }
        }
    }

    // Record THIS turn for next time — including the messages we just reused (so the entry stays
    // hot and a longer prefix can freeze next turn). We re-read the (now possibly rewritten)
    // compressed content so what we store is exactly what we emit on the wire.
    for (idx, h, fresh_content) in plan.entries {
        let to_store = compressed
            .get(plan.key)
            .and_then(Value::as_array)
            .and_then(|a| a.get(idx + offset))
            .and_then(|m| m.get("content"))
            .cloned()
            .unwrap_or(fresh_content);
        memo.put(h, to_store);
    }

    reused_count
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
}
