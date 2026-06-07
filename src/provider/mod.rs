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

/// Provider-specific structural accessors used by the stages.
pub trait Provider {
    fn kind(&self) -> ProviderKind;

    /// JSON pointers to every text segment in the request (Stage D scan targets).
    /// Each pointer addresses a JSON string.
    fn content_text_pointers(&self, req: &Request) -> Vec<String>;

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
                    } else if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                        // Tool results carry the bulk of agent context (file reads, command
                        // output). Their content is a string or an array of text blocks.
                        match block.get("content") {
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
                        }
                    }
                }
            }
            _ => {}
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
pub(crate) fn truncate_chars(s: &mut String, max: usize) {
    if s.chars().count() > max {
        let truncated: String = s.chars().take(max).collect();
        *s = format!("{truncated}…");
    }
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
        || obj.contains_key("generationConfig")
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
