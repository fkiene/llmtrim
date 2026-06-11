//! Provider adapters: map the neutral pipeline onto each provider's wire shape.
//!
//! The [`Provider`] trait is intentionally object-safe (no generic methods) so the
//! pipeline can hold a `Box<dyn Provider>` chosen at runtime from `--provider` or
//! [`detect`]. Each adapter knows only the structural differences the stages care
//! about: where text content lives, and the field names for output controls.

use serde_json::Value;

use crate::ir::{ProviderKind, Request};

mod anthropic;
mod google;
mod openai;

pub use anthropic::AnthropicProvider;
pub use google::GoogleProvider;
pub use openai::OpenAiProvider;

/// Normalized conversational role of the turn a content pointer belongs to. Lets
/// role-aware stages (retrieve) work across every wire shape instead of hard-coding
/// `/messages/{i}`. `None` from [`Provider::role_at`] means top-level system text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// Map a raw provider role string to the neutral role. Unknown → `User` (the
    /// conservative "compressible context" bucket).
    pub(crate) fn from_str(s: &str) -> Role {
        match s {
            "system" | "developer" => Role::System,
            "assistant" | "model" => Role::Assistant,
            "tool" | "function" => Role::Tool,
            _ => Role::User,
        }
    }
}

/// The conversational turn index a content pointer addresses — the `i` in
/// `messages[i]` / `input[i]` / `contents[i]` — or `None` for top-level text
/// (`/system`, `/instructions`, `/systemInstruction/...`). Wire-shape agnostic.
pub fn turn_index(pointer: &str) -> Option<usize> {
    let rest = pointer
        .strip_prefix("/messages/")
        .or_else(|| pointer.strip_prefix("/input/"))
        .or_else(|| pointer.strip_prefix("/contents/"))?;
    rest.split('/').next()?.parse().ok()
}

/// Append pointers to every JSON string leaf under `value`, rooted at `prefix`
/// (RFC 6901-escaped). Used for free-form object payloads — tool-call arguments,
/// `tool_use.input`, Gemini `functionResponse.response` — where the model-readable
/// text lives in arbitrary string leaves rather than a known field.
pub(crate) fn string_leaf_pointers(value: &Value, prefix: &str, out: &mut Vec<String>) {
    match value {
        Value::String(_) => out.push(prefix.to_string()),
        Value::Array(a) => {
            for (i, v) in a.iter().enumerate() {
                string_leaf_pointers(v, &format!("{prefix}/{i}"), out);
            }
        }
        Value::Object(m) => {
            for (k, v) in m {
                let ek = k.replace('~', "~0").replace('/', "~1");
                string_leaf_pointers(v, &format!("{prefix}/{ek}"), out);
            }
        }
        _ => {}
    }
}

/// Provider-specific structural accessors used by the stages.
pub trait Provider {
    fn kind(&self) -> ProviderKind;

    /// JSON pointers to every text segment in the request (Stage D scan targets).
    /// Each pointer addresses a JSON string.
    fn content_text_pointers(&self, req: &Request) -> Vec<String>;

    /// The conversational role of the turn a content pointer belongs to, or `None`
    /// for top-level system text (no enclosing turn — always pinned). Wire-shape
    /// agnostic seam for role-aware stages; default resolves `/messages/{i}/role`.
    fn role_at(&self, req: &Request, pointer: &str) -> Option<Role> {
        let i = turn_index(pointer)?;
        let role = req
            .raw()
            .pointer(&format!("/messages/{i}/role"))
            .and_then(Value::as_str)?;
        Some(Role::from_str(role))
    }

    /// Set the maximum output tokens using the provider's field name.
    fn set_max_tokens(&self, req: &mut Request, max_tokens: u64);

    /// Current output-token cap, if set.
    fn max_tokens(&self, req: &Request) -> Option<u64>;

    /// Append a stop sequence using the provider's field name.
    fn add_stop_sequence(&self, req: &mut Request, stop: &str);

    /// Prepend a system instruction (provider-specific location).
    fn add_system_instruction(&self, req: &mut Request, text: &str);

    /// Bind server-side structured output to a JSON schema (Stage F, JSON-only).
    fn bind_structured_output(&self, req: &mut Request, name: &str, schema: Value);

    /// Mark the invariant prefix (system, tool schemas) with provider cache
    /// breakpoints, up to `max`. No-op where the provider caches automatically
    /// (OpenAI). Lossless — adds caching hints, never changes content.
    fn set_cache_breakpoints(&self, req: &mut Request, max: usize);

    /// Pin the provider's automatic prefix cache to a tenant-stable identity via a
    /// stable cache key (OpenAI `prompt_cache_key`), so similar prefixes route to the
    /// same cache node instead of colliding org-wide. Only set if absent. No-op where
    /// the provider has no such key (Anthropic / Google use explicit breakpoints).
    fn set_prompt_cache_key(&self, req: &mut Request, key: &str);

    /// `(name, description)` for each tool, in array order (empty if no tools).
    /// Abstracts the OpenAI `function.{name,description}` vs Anthropic top-level
    /// `{name,description}` shapes (Stage G).
    fn tool_descriptors(&self, req: &Request) -> Vec<(String, String)>;

    /// Retain only the tools whose `keep` flag is true (positional). Stage G.
    fn retain_tools(&self, req: &mut Request, keep: &[bool]);

    /// Truncate each tool description to at most `max_chars`. Stage G.
    fn truncate_tool_descriptions(&self, req: &mut Request, max_chars: usize);

    /// Extract the model's answer text from a response body (None if the shape is
    /// unexpected). Used by rehydration and the live quality `Model`.
    fn answer_text(&self, response: &Value) -> Option<String>;

    /// Set the image detail tier on image content blocks (Stage H). No-op where the
    /// provider has no per-image tier (Anthropic).
    fn set_image_detail(&self, req: &mut Request, tier: &str);

    /// Downscale embedded base64 images to this provider's effective resolution cap
    /// (quality-neutral).
    fn downscale_images(&self, req: &mut Request);
}

/// JSON pointer to a content block's text, when it is a `{"type":"text","text":"…"}`
/// block (`prefix` is the block's own address, e.g. `/messages/0/content/2`). The
/// single text-block predicate, shared by both providers' pointer scans.
pub(crate) fn text_block_ptr(block: &Value, prefix: &str) -> Option<String> {
    let is_text = block.get("type").and_then(Value::as_str) == Some("text")
        && block.get("text").is_some_and(Value::is_string);
    is_text.then(|| format!("{prefix}/text"))
}

/// Append pointers to every text segment under a `messages` array: string content
/// directly, or the text blocks of array content. The shared message walk for
/// `content_text_pointers` (both wire formats share the `messages` shape).
pub(crate) fn message_text_pointers(messages: &Value, out: &mut Vec<String>) {
    let Some(messages) = messages.as_array() else {
        return;
    };
    for (i, msg) in messages.iter().enumerate() {
        match msg.get("content") {
            Some(Value::String(_)) => out.push(format!("/messages/{i}/content")),
            Some(Value::Array(blocks)) => {
                for (j, block) in blocks.iter().enumerate() {
                    let prefix = format!("/messages/{i}/content/{j}");
                    if let Some(p) = text_block_ptr(block, &prefix) {
                        out.push(p);
                        continue;
                    }
                    match block.get("type").and_then(Value::as_str) {
                        // Tool results carry the bulk of agent context (file reads, command
                        // output). Their content is a string or an array of text blocks.
                        Some("tool_result") => match block.get("content") {
                            Some(Value::String(_)) => out.push(format!("{prefix}/content")),
                            Some(Value::Array(inner)) => {
                                for (k, ib) in inner.iter().enumerate() {
                                    if let Some(p) =
                                        text_block_ptr(ib, &format!("{prefix}/content/{k}"))
                                    {
                                        out.push(p);
                                    }
                                }
                            }
                            _ => {}
                        },
                        // Anthropic `tool_use` echoes the assistant's call arguments — for
                        // Write/Edit tools the whole file lives in `input` (resent every turn).
                        Some("tool_use") => {
                            if let Some(input) = block.get("input") {
                                string_leaf_pointers(input, &format!("{prefix}/input"), out);
                            }
                        }
                        // Anthropic text `document` blocks: plain-text data we can compress.
                        Some("document") => {
                            let textual = block
                                .pointer("/source/media_type")
                                .and_then(Value::as_str)
                                .is_none_or(|m| m.starts_with("text/"));
                            if textual
                                && block.pointer("/source/data").is_some_and(Value::is_string)
                            {
                                out.push(format!("{prefix}/source/data"));
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        // OpenAI assistant history: `tool_calls[].function.arguments` is a JSON-in-a-string
        // (file writes, patches), model-readable and resent every turn.
        if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
            for (j, call) in calls.iter().enumerate() {
                if call
                    .pointer("/function/arguments")
                    .is_some_and(Value::is_string)
                {
                    out.push(format!("/messages/{i}/tool_calls/{j}/function/arguments"));
                }
            }
        }
    }
}

/// Apply `f` to every content block of every array-content message, mutating each in
/// place. The shared messages→content→blocks traversal for the per-block image transforms.
pub(crate) fn for_each_content_block(req: &mut Request, mut f: impl FnMut(&mut Value)) {
    let Some(messages) = req
        .raw_mut()
        .get_mut("messages")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    for m in messages.iter_mut() {
        let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for b in blocks.iter_mut() {
            f(b);
        }
    }
}

/// Drop tools where `keep[i]` is false (shared by the adapters — tools is a
/// top-level array in both wire formats).
pub(crate) fn retain_tools_array(req: &mut Request, keep: &[bool]) {
    if let Some(Value::Array(tools)) = req.raw_mut().get_mut("tools") {
        let mut idx = 0usize;
        tools.retain(|_| {
            let k = keep.get(idx).copied().unwrap_or(true);
            idx += 1;
            k
        });
    }
}

/// Truncate `s` to at most `max` chars, appending `…` when shortened.
///
/// Boundary-aware: fills the budget with whole sentences (Unicode sentence
/// segmentation, so any script works), falling back to whole lines, then whole
/// words — never cutting mid-word. Slight undershoot of the budget is expected.
pub(crate) fn truncate_chars(s: &mut String, max: usize) {
    use unicode_segmentation::UnicodeSegmentation;

    if s.chars().count() <= max {
        return;
    }
    let keep_bytes = fit_units(s.split_sentence_bounds(), max)
        .or_else(|| fit_units(s.split_inclusive('\n'), max))
        .or_else(|| fit_units(s.split_word_bounds(), max));
    match keep_bytes {
        Some(n) => {
            s.truncate(n);
            s.truncate(s.trim_end().len());
        }
        // Degenerate case: the very first word exceeds the budget. Hard char
        // cut rather than emptying the description entirely.
        None => *s = s.chars().take(max).collect(),
    }
    s.push('…');
}

/// Byte length of the longest prefix of contiguous `units` whose total char
/// count fits in `max`. `None` when not even the first unit fits.
fn fit_units<'a>(units: impl Iterator<Item = &'a str>, max: usize) -> Option<usize> {
    let mut chars = 0;
    let mut bytes = 0;
    for u in units {
        let c = u.chars().count();
        if chars + c > max {
            break;
        }
        chars += c;
        bytes += u.len();
    }
    (bytes > 0).then_some(bytes)
}

/// Construct the adapter for a known provider kind.
pub fn for_kind(kind: ProviderKind) -> Box<dyn Provider> {
    match kind {
        ProviderKind::OpenAi => Box::new(OpenAiProvider),
        ProviderKind::Anthropic => Box::new(AnthropicProvider),
        ProviderKind::Google => Box::new(GoogleProvider),
    }
}

/// Heuristically detect the provider from a parsed request body. Static, no model.
/// Returns `None` when the shape is ambiguous — the caller should then require an
/// explicit `--provider`.
pub fn detect(body: &Value) -> Option<ProviderKind> {
    let obj = body.as_object()?;

    // Gemini-only top-level fields: messages live under `contents`, the system prompt
    // under `systemInstruction`, output controls under `generationConfig`.
    if obj.contains_key("contents")
        || obj.contains_key("systemInstruction")
        || obj.contains_key("system_instruction")
        || obj.contains_key("generationConfig")
        || obj.contains_key("generation_config")
    {
        return Some(ProviderKind::Google);
    }

    // Anthropic-only top-level fields.
    if obj.contains_key("system")
        || obj.contains_key("stop_sequences")
        || obj.contains_key("anthropic_version")
    {
        return Some(ProviderKind::Anthropic);
    }

    // OpenAI Responses API: `input` replaces `messages`, with `instructions` or
    // `max_output_tokens` alongside. No other provider uses this top-level shape.
    if obj.contains_key("input")
        && (obj.contains_key("instructions") || obj.contains_key("max_output_tokens"))
    {
        return Some(ProviderKind::OpenAi);
    }

    // OpenAI-only top-level fields.
    if obj.contains_key("max_completion_tokens") || obj.contains_key("response_format") {
        return Some(ProviderKind::OpenAi);
    }

    // A `system`-role message is OpenAI-shaped (Anthropic carries system top-level).
    if let Some(messages) = obj.get("messages").and_then(Value::as_array)
        && messages
            .iter()
            .any(|m| m.get("role").and_then(Value::as_str) == Some("system"))
    {
        return Some(ProviderKind::OpenAi);
    }

    None
}

/// Append a stop sequence to `key`, promoting a bare string to an array as needed.
pub(crate) fn append_stop(root: &mut Value, key: &str, stop: &str) {
    let Some(obj) = root.as_object_mut() else {
        return;
    };
    match obj.get_mut(key) {
        Some(Value::Array(arr)) => arr.push(Value::String(stop.to_string())),
        Some(slot @ Value::String(_)) => {
            let prev = slot.as_str().unwrap_or_default().to_string();
            *slot = Value::Array(vec![Value::String(prev), Value::String(stop.to_string())]);
        }
        _ => {
            obj.insert(
                key.to_string(),
                Value::Array(vec![Value::String(stop.to_string())]),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::truncate_chars;

    fn trunc(s: &str, max: usize) -> String {
        let mut s = s.to_string();
        truncate_chars(&mut s, max);
        s
    }

    #[test]
    fn short_input_untouched() {
        assert_eq!(trunc("Short description.", 300), "Short description.");
    }

    #[test]
    fn cuts_at_sentence_boundary() {
        let input =
            "First sentence here. Second sentence is longer. Third one overflows the budget.";
        // Budget admits the first two sentences but not the third.
        assert_eq!(
            trunc(input, 50),
            "First sentence here. Second sentence is longer.…"
        );
    }

    #[test]
    fn falls_back_to_line_boundary() {
        // One run-on "sentence" spread over lines: sentence segmentation can't
        // fit a unit, line fallback can.
        let input = "alpha beta gamma\ndelta epsilon zeta\neta theta iota kappa lambda";
        assert_eq!(trunc(input, 40), "alpha beta gamma\ndelta epsilon zeta…");
    }

    #[test]
    fn single_long_sentence_falls_back_to_words() {
        let input = "one two three four five six seven eight nine ten eleven twelve";
        let out = trunc(input, 30);
        assert!(out.ends_with('…'), "{out}");
        let body = out.trim_end_matches('…');
        // Never mid-word: the kept prefix must end on a word from the input.
        assert!(input.starts_with(body));
        assert!(body.split_whitespace().all(|w| input.contains(w)));
        assert_eq!(body, "one two three four five six");
    }

    #[test]
    fn no_mid_word_cut() {
        let out = trunc("Avoid cutting important words in the middle always", 20);
        assert_eq!(out, "Avoid cutting…");
    }

    #[test]
    fn japanese_sentences() {
        let input = "これは最初の文です。これは二番目の文です。これは三番目のとても長い文です。";
        // 10 + 11 = 21 chars for the first two sentences; third doesn't fit.
        assert_eq!(
            trunc(input, 25),
            "これは最初の文です。これは二番目の文です。…"
        );
    }

    #[test]
    fn degenerate_giant_word_hard_cuts() {
        let input = "a".repeat(50);
        let out = trunc(&input, 10);
        assert_eq!(out, format!("{}…", "a".repeat(10)));
    }
}
