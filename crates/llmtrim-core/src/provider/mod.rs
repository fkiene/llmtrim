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
/// Boundary-aware and salience-aware: the first sentence (the tool's one-line
/// identity) is always kept, then the remaining budget is filled with whole
/// sentences preferring those dense in code-like identifiers or enumeration
/// members (a language-neutral lexical signal — plain prose ranks lowest).
/// Original sentence order is preserved; skipped runs are elided with a
/// single " … " marker. Falls back to whole lines, then whole words, when the
/// first sentence alone exceeds the budget — never cutting mid-word. Slight
/// undershoot of the budget is expected.
pub(crate) fn truncate_chars(s: &mut String, max: usize) {
    use unicode_segmentation::UnicodeSegmentation;

    if s.chars().count() <= max {
        return;
    }
    if let Some(out) = select_salient_sentences(s, max) {
        *s = out;
        return;
    }
    let keep_bytes =
        fit_units(s.split_inclusive('\n'), max).or_else(|| fit_units(s.split_word_bounds(), max));
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

/// Marker spliced between kept sentences that are not adjacent in the source.
const ELISION: &str = " … ";

/// A selectable unit of the source text: a sentence span, or a structurally
/// detected list block (optional intro line ending with `:` plus consecutive
/// marker lines with identifier-shaped heads) carrying the data needed to
/// emit a compacted form when the full block does not fit.
struct Unit<'a> {
    start: usize, // byte offsets into the source
    end: usize,
    list: Option<ListInfo<'a>>,
    /// Code/example block — fenced (``` or ~~~) or an unfenced indented run —
    /// atomic (never sentence-split) and ranked below every prose/list unit
    /// regardless of identifier density.
    fence: bool,
    /// For a list-item unit (marker line + its continuation body): byte
    /// offset where the body begins. The unit is scored on the marker line
    /// alone, and when the whole item does not fit the marker line is
    /// emitted by itself — the body (payload/example) elides first.
    body_start: Option<usize>,
}

struct ListInfo<'a> {
    intro: Option<&'a str>,
    heads: Vec<&'a str>,
}

/// How a unit is emitted: dropped, verbatim, or as a compacted list.
enum Keep {
    No,
    Full,
    Compact(String),
}

/// Salience-aware unit selection: keep the first sentence, then fill the
/// remaining budget with the highest-scoring units (ties broken by source
/// order), emitted in original order with `ELISION` over skipped runs.
/// List units that do not fit whole fall back to their compacted form
/// (intro + item heads) rather than being dropped.
/// `None` when the first unit alone does not fit (caller falls back).
fn select_salient_sentences(s: &str, max: usize) -> Option<String> {
    let units = segment_units(s);
    let chars: Vec<usize> = units
        .iter()
        .map(|u| s[u.start..u.end].trim_end().chars().count())
        .collect();
    if units.is_empty() || chars[0] > max {
        return None;
    }

    // Rank candidates (all but the mandatory first) by identifier density.
    // Lists are scored on their compactable core (intro + heads).
    let scores: Vec<f64> = units
        .iter()
        .map(|u| {
            if u.fence {
                // Code examples lose to all prose/list units: their identifier
                // density is high by construction but they carry usage samples,
                // not API constraints.
                return -1.0;
            }
            match (&u.list, u.body_start) {
                (Some(l), _) => identifier_density(&compact_core(l)),
                // Item units score on their marker line only: the marker is
                // the contract, the body is payload.
                (None, Some(b)) => marker_score(&s[u.start..b]),
                (None, None) => identifier_density(&s[u.start..u.end]),
            }
        })
        .collect();
    let mut ranked: Vec<usize> = (1..units.len()).collect();
    ranked.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let elision_chars = ELISION.chars().count();
    let mut keep: Vec<Keep> = units.iter().map(|_| Keep::No).collect();
    keep[0] = Keep::Full;
    let mut used = chars[0];
    for i in ranked {
        // Pessimistic cost: the unit plus one elision marker for the gap it
        // may open. Adjacent picks undershoot, which is fine.
        if used + chars[i] + elision_chars <= max {
            used += chars[i] + elision_chars;
            keep[i] = Keep::Full;
        } else if let Some(list) = &units[i].list {
            // Compaction tier: the whole block does not fit, but the intro
            // plus item heads (or a prefix of them ending ", …") might.
            let budget = max.saturating_sub(used + elision_chars);
            if let Some(text) = compact_list(list, budget) {
                used += text.chars().count() + elision_chars;
                keep[i] = Keep::Compact(text);
            }
        } else if let Some(b) = units[i].body_start {
            // Item fallback: the whole item does not fit — emit its marker
            // line alone with an elision marker; the body elides first. A
            // marker line itself over budget is trimmed at sentence bounds.
            use unicode_segmentation::UnicodeSegmentation;
            let marker = s[units[i].start..b].trim_end();
            let budget = max.saturating_sub(used + elision_chars + 1);
            let text = if marker.chars().count() <= budget {
                Some(marker)
            } else {
                fit_units(marker.split_sentence_bounds(), budget).map(|n| marker[..n].trim_end())
            };
            if let Some(t) = text {
                used += t.chars().count() + 1 + elision_chars;
                keep[i] = Keep::Compact(format!("{t}…"));
            }
        }
    }

    // Rebuild from contiguous runs of the original text so adjacent verbatim
    // units keep their exact source bytes (no separator is invented between
    // them). Compacted lists are synthesized text, joined with a space when
    // adjacent to the previous kept unit and ELISION otherwise.
    let mut out = String::new();
    let mut run_start: Option<usize> = None;
    let mut last_kept: Option<usize> = None;
    for (i, u) in units.iter().enumerate() {
        match &keep[i] {
            Keep::No => {
                if let Some(start) = run_start.take() {
                    out.push_str(s[start..u.start].trim_end());
                }
            }
            Keep::Full => {
                if run_start.is_none() {
                    if !out.is_empty() {
                        out.push_str(separator(last_kept, i));
                    }
                    run_start = Some(u.start);
                }
                last_kept = Some(i);
            }
            Keep::Compact(text) => {
                if let Some(start) = run_start.take() {
                    out.push_str(s[start..u.start].trim_end());
                }
                if !out.is_empty() {
                    out.push_str(separator(last_kept, i));
                }
                out.push_str(text);
                last_kept = Some(i);
            }
        }
    }
    if let Some(start) = run_start.take()
        && let Some(u) = units.last()
    {
        out.push_str(s[start..u.end].trim_end());
    }
    // Trailing ellipsis only when tail content was dropped — interior
    // elisions are already marked, and a partial compact ends with "…".
    match &keep[units.len() - 1] {
        Keep::No => out.push('…'),
        Keep::Compact(t) if !t.ends_with('…') => out.push('…'),
        _ => {}
    }
    Some(out)
}

/// Space when unit `i` immediately follows the previously kept unit (nothing
/// was dropped between them), ELISION otherwise.
fn separator(last_kept: Option<usize>, i: usize) -> &'static str {
    if last_kept.is_some_and(|p| p + 1 == i) {
        " "
    } else {
        ELISION
    }
}

/// Split the source into selectable units: fenced code blocks (each one
/// atomic unit), list blocks, and sentence spans — all detected structurally
/// by line shape, never by content language.
fn segment_units(s: &str) -> Vec<Unit<'_>> {
    let mut units = Vec::new();
    let mut pos = 0usize;
    // Base indentation of the description: indented-example detection is
    // relative to it, so uniformly indented texts are not all "examples".
    let base = s
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(indent_width)
        .min()
        .unwrap_or(0);
    for (start, end) in fence_regions(s) {
        segment_prose(s, pos, start, base, &mut units);
        units.push(Unit {
            start,
            end,
            list: None,
            fence: true,
            body_start: None,
        });
        pos = end;
    }
    segment_prose(s, pos, s.len(), base, &mut units);
    units
}

/// Visual indentation width of a line's leading whitespace (tab = 4).
fn indent_width(line: &str) -> usize {
    line.chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .map(|c| if c == '\t' { 4 } else { 1 })
        .sum()
}

/// Index of the last line of an unfenced example block starting at `lines[i]`:
/// a run of two or more lines indented ≥2 columns beyond `base` (blank lines
/// tolerated inside) none of which is a compactable list item. `None` when
/// `lines[i]` does not open such a run.
fn example_run(lines: &[(usize, &str)], i: usize, base: usize) -> Option<usize> {
    let deep =
        |l: &str| !l.trim().is_empty() && indent_width(l) >= base + 2 && item_head(l).is_none();
    if !deep(lines[i].1) {
        return None;
    }
    let mut last = i;
    let mut j = i + 1;
    while j < lines.len() {
        let l = lines[j].1;
        if l.trim().is_empty() {
            j += 1;
        } else if deep(l) {
            last = j;
            j += 1;
        } else {
            break;
        }
    }
    (last > i).then_some(last)
}

/// How an open example region will be closed: a fence line repeating the
/// same char, or the matching closing tag line.
enum BlockClose {
    Fence(char),
    Tag(String),
}

/// Byte ranges of delimited example blocks. Two structural delimiters:
/// a line starting (after indentation) with three or more ``` ` ``` or `~`
/// opens a fence, closed by the next line starting with the same char; and
/// a line that is solely an XML-ish tag (`<example>`, `<good-example>`, …)
/// opens a tag block, closed by the matching `</…>` line. An unclosed
/// delimiter extends to the end.
fn fence_regions(s: &str) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut open: Option<(usize, BlockClose)> = None;
    let mut off = 0usize;
    for line in s.split_inclusive('\n') {
        let t = line.trim();
        let fence_char = t
            .chars()
            .next()
            .filter(|&c| (c == '`' || c == '~') && t.chars().take_while(|&x| x == c).count() >= 3);
        match &open {
            Some((start, close)) => {
                let closed = match close {
                    BlockClose::Fence(c) => fence_char == Some(*c),
                    BlockClose::Tag(name) => t == format!("</{name}>"),
                };
                if closed {
                    regions.push((*start, off + line.len()));
                    open = None;
                }
            }
            None => {
                if let Some(c) = fence_char {
                    open = Some((off, BlockClose::Fence(c)));
                } else if let Some(name) = lone_open_tag(t) {
                    open = Some((off, BlockClose::Tag(name.to_string())));
                }
            }
        }
        off += line.len();
    }
    if let Some((start, _)) = open {
        regions.push((start, s.len()));
    }
    regions
}

/// Tag name when the trimmed line is exactly an opening tag `<name>` made of
/// identifier chars (letters, digits, `-`, `_`). `None` otherwise.
fn lone_open_tag(t: &str) -> Option<&str> {
    let name = t.strip_prefix('<')?.strip_suffix('>')?;
    (!name.is_empty()
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_'))
    .then_some(name)
}

/// Segment `s[start..end]` (a fence-free span) into list blocks interleaved
/// with sentence spans.
fn segment_prose<'a>(s: &'a str, start: usize, end: usize, base: usize, units: &mut Vec<Unit<'a>>) {
    let mut lines: Vec<(usize, &str)> = Vec::new();
    let mut off = start;
    for l in s[start..end].split_inclusive('\n') {
        lines.push((off, l));
        off += l.len();
    }
    let mut plain_start = start;
    let mut i = 0usize;
    while i < lines.len() {
        // A block starts at an item line, or at an intro line ending with ':'
        // immediately followed by an item line.
        // The intro is only the *last sentence* of its line — earlier
        // sentences sharing the line stay ordinary prose.
        let (intro, intro_start, first_item) = if item_head(lines[i].1).is_some() {
            (None, 0, i)
        } else if lines[i].1.trim_end().ends_with(':')
            && i + 1 < lines.len()
            && item_head(lines[i + 1].1).is_some()
        {
            let rel = last_sentence_offset(lines[i].1);
            (Some(lines[i].1[rel..].trim()), rel, i + 1)
        } else if let Some(last) = example_run(&lines, i, base) {
            // Unfenced example block: indented code/sample lines are one
            // atomic, lowest-tier unit — same treatment as a fence.
            let block_start = lines[i].0;
            let block_end = lines[last].0 + lines[last].1.len();
            push_sentences(s, plain_start, block_start, units);
            units.push(Unit {
                start: block_start,
                end: block_end,
                list: None,
                fence: true,
                body_start: None,
            });
            plain_start = block_end;
            i = last + 1;
            continue;
        } else if marker_body(lines[i].1).is_some() {
            // Item unit: a marker line whose head is not code-like (no
            // compactable list possible). Its body — following non-marker
            // lines at ANY indentation, up to the next marker line, blank
            // line or block end — belongs to the item, never to prose.
            let mut j = i + 1;
            while j < lines.len()
                && !lines[j].1.trim().is_empty()
                && marker_body(lines[j].1).is_none()
            {
                j += 1;
            }
            push_item_unit(s, lines[i].0, &lines[i..j], &mut plain_start, units);
            i = j;
            continue;
        } else {
            i += 1;
            continue;
        };
        let mut j = first_item;
        let mut items = 0usize;
        let mut heads = Vec::new();
        while j < lines.len()
            && let Some(h) = item_head(lines[j].1)
        {
            items += 1;
            // Dedup repeated identical heads — the compacted form must not
            // emit "x, x, x" when several items share a head.
            if !heads.contains(&h) {
                heads.push(h);
            }
            j += 1;
            // Non-marker lines after a bullet — at any indentation — are the
            // item's body (continuations), never independent units.
            while j < lines.len()
                && !lines[j].1.trim().is_empty()
                && marker_body(lines[j].1).is_none()
            {
                j += 1;
            }
        }
        if items < 2 {
            // A lone item is not a compactable list: keep it as a single
            // item unit (marker line + body) so its body cannot leak.
            push_item_unit(
                s,
                lines[first_item].0,
                &lines[first_item..j],
                &mut plain_start,
                units,
            );
            i = j;
            continue;
        }
        let block_start = if intro.is_some() {
            lines[i].0 + intro_start
        } else {
            lines[first_item].0
        };
        let block_end = lines[j - 1].0 + lines[j - 1].1.len();
        push_sentences(s, plain_start, block_start, units);
        units.push(Unit {
            start: block_start,
            end: block_end,
            list: Some(ListInfo { intro, heads }),
            fence: false,
            body_start: None,
        });
        plain_start = block_end;
        i = j;
    }
    push_sentences(s, plain_start, end, units);
}

/// Append an item unit — marker line plus its continuation body — covering
/// `lines` (non-empty; first line is the marker), flushing pending prose
/// from `plain_start` first.
fn push_item_unit<'a>(
    s: &'a str,
    start: usize,
    lines: &[(usize, &str)],
    plain_start: &mut usize,
    units: &mut Vec<Unit<'a>>,
) {
    let (last_off, last_line) = lines[lines.len() - 1];
    let end = last_off + last_line.len();
    let marker_end = lines[0].0 + lines[0].1.len();
    push_sentences(s, *plain_start, start, units);
    units.push(Unit {
        start,
        end,
        list: None,
        fence: false,
        // Body-less items use `end`: the whole unit is the marker line.
        body_start: Some(marker_end.min(end)),
    });
    *plain_start = end;
}

/// Byte offset of the last non-whitespace sentence within `line`.
fn last_sentence_offset(line: &str) -> usize {
    use unicode_segmentation::UnicodeSegmentation;
    let mut off = 0usize;
    let mut last = 0usize;
    for sent in line.split_sentence_bounds() {
        if !sent.trim().is_empty() {
            last = off;
        }
        off += sent.len();
    }
    last
}

/// Append the sentence spans of `s[start..end]` as verbatim units.
fn push_sentences<'a>(s: &'a str, start: usize, end: usize, units: &mut Vec<Unit<'a>>) {
    use unicode_segmentation::UnicodeSegmentation;
    let mut off = start;
    for sent in s[start..end].split_sentence_bounds() {
        if !sent.trim().is_empty() {
            units.push(Unit {
                start: off,
                end: off + sent.len(),
                list: None,
                fence: false,
                body_start: None,
            });
        }
        off += sent.len();
    }
}

/// Head of a list item line: the leading identifier-shaped token(s) of a
/// bullet/dash/numbered line, up to the first `:`, `(` or `—`. `None` when
/// the line is not item-shaped or its head is not code-like.
fn item_head(line: &str) -> Option<&str> {
    let body = marker_body(line)?;
    let end = body.find([':', '(', '—']).unwrap_or(body.len());
    let head = body[..end].trim();
    if head.is_empty() || !head.split_whitespace().all(is_code_like) {
        return None;
    }
    Some(head)
}

/// Salience of an item unit's marker line. Three structural tiers:
/// call-signature-shaped markers (`- agent(prompt: …`, `- budget: {…`) are
/// API/example payload and rank with fences; markers carrying an all-caps
/// admonition label (an explicit author salience signal: `IMPORTANT:`,
/// `WARNING:`, `NOTE:` — an all-caps token ending in `:`) rank above every
/// plain unit; the rest score by identifier density like ordinary sentences.
fn marker_score(marker: &str) -> f64 {
    let body = marker_body(marker).unwrap_or(marker);
    if body
        .split_whitespace()
        .take(2)
        .any(|t| t.contains(['(', '{']))
    {
        return -1.0;
    }
    let emphasis = body.split_whitespace().any(|t| {
        t.strip_suffix(':')
            .is_some_and(|w| w.chars().count() >= 2 && w.chars().all(char::is_uppercase))
    });
    identifier_density(marker) + if emphasis { 1.0 } else { 0.0 }
}

/// Trimmed body of a bullet/dash/numbered marker line, regardless of whether
/// its head is code-like. `None` when the line is not marker-shaped.
/// Whitespace is required after the marker so prose like "--flag" or "3.14"
/// never reads as an item.
fn marker_body(line: &str) -> Option<&str> {
    let t = line.trim_start();
    let body = if let Some(r) = t.strip_prefix(['-', '*', '•', '–']) {
        r
    } else {
        let digits = t.chars().take_while(char::is_ascii_digit).count();
        if digits == 0 {
            return None;
        }
        t[digits..].strip_prefix(['.', ')'])?
    };
    body.starts_with([' ', '\t']).then(|| body.trim())
}

/// Compacted core of a list: intro (if any) + item heads joined by ", ".
fn compact_core(list: &ListInfo) -> String {
    let mut out = list.intro.map(|i| format!("{i} ")).unwrap_or_default();
    out.push_str(&list.heads.join(", "));
    out
}

/// Compacted list fitting `budget` chars: the full core if it fits, else as
/// many whole heads as fit ending with ", …". Heads are never cut inside.
/// `None` when not even one head fits.
fn compact_list(list: &ListInfo, budget: usize) -> Option<String> {
    let full = compact_core(list);
    if full.chars().count() <= budget {
        return Some(full);
    }
    let mut out = list.intro.map(|i| format!("{i} ")).unwrap_or_default();
    let tail_chars = ", …".chars().count();
    let mut used = out.chars().count();
    let mut kept = 0usize;
    for h in &list.heads {
        let add = h.chars().count() + if kept > 0 { 2 } else { 0 };
        if used + add + tail_chars > budget {
            break;
        }
        if kept > 0 {
            out.push_str(", ");
        }
        out.push_str(h);
        used += add;
        kept += 1;
    }
    if kept == 0 {
        return None;
    }
    out.push_str(", …");
    Some(out)
}

/// Fraction of whitespace-separated tokens that look code-like: backticked
/// spans, underscores, `::`, digits, mixed case after the first char,
/// hyphenated compounds, or comma-separated enumeration members. Purely
/// lexical — no language- or tool-specific lists.
fn identifier_density(sentence: &str) -> f64 {
    let mut total = 0usize;
    let mut hits = 0usize;
    for tok in sentence.split_whitespace() {
        total += 1;
        if is_code_like(tok) {
            hits += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

fn is_code_like(tok: &str) -> bool {
    if tok.contains('`') || tok.contains('_') || tok.contains("::") {
        return true;
    }
    // Comma-separated enumeration member ("foo, bar, baz" — each but the last
    // ends with a comma).
    let body = tok
        .strip_suffix(',')
        .map(|b| (b, true))
        .unwrap_or((tok, false));
    let (word, is_member) = body;
    if is_member && word.chars().any(char::is_alphanumeric) {
        return true;
    }
    if word.chars().any(|c| c.is_ascii_digit()) {
        return true;
    }
    // Mixed case beyond an ordinary capitalized word: an uppercase letter
    // after the first char alongside lowercase (camelCase, PascalCase).
    let has_lower = word.chars().any(char::is_lowercase);
    let late_upper = word.chars().skip(1).any(char::is_uppercase);
    if has_lower && late_upper {
        return true;
    }
    // Hyphenated compound with alphanumerics on both sides (kebab-case).
    word.match_indices('-').any(|(i, _)| {
        word[..i]
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric)
            && word[i + 1..]
                .chars()
                .next()
                .is_some_and(char::is_alphanumeric)
    })
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
    fn identifier_sentence_survives_mid_text() {
        let input = "Launches a specialized agent to handle the task. \
            The agent runs in its own context and reports back when finished. \
            Valid types: general-purpose, code-reviewer, test-runner. \
            Results may take a while to arrive depending on the task.";
        // Budget can't hold everything: the enumeration sentence must win over
        // the prose sentences around it, with elision markers in between.
        let out = trunc(input, 120);
        assert!(out.starts_with("Launches a specialized agent to handle the task."));
        assert!(
            out.contains("Valid types: general-purpose, code-reviewer, test-runner."),
            "{out}"
        );
        assert!(out.contains(" … "), "{out}");
        assert!(!out.contains("reports back"), "{out}");
        assert!(out.chars().count() <= 121, "{out}"); // budget + trailing …
    }

    #[test]
    fn elision_marker_not_duplicated_when_tail_kept() {
        let input = "Tool identity sentence here. Some filler prose in the middle of it. \
            Use `run_command` with `--flag` and `path/to_file`.";
        let out = trunc(input, 105);
        // Identifier-heavy tail kept, prose middle elided; no trailing … since
        // the true ending is present and the gap is already marked.
        assert!(out.contains("`run_command`"), "{out}");
        assert!(out.contains(" … "), "{out}");
        assert!(!out.ends_with('…'), "{out}");
    }

    #[test]
    fn late_list_compacts_to_heads() {
        // Shaped like the real Agent-tool case: long description whose only
        // enumeration of valid argument values is a late bullet list with
        // 150+ char items — none fits whole in the leftover budget.
        let heads = [
            "general-purpose",
            "rtk-testing-specialist",
            "code-reviewer",
            "test-runner",
            "doc-writer",
            "perf-auditor",
            "release-manager",
            "security-scanner",
            "data-migrator",
            "ui-builder",
            "api-designer",
            "log-analyzer",
            "infra-planner",
        ];
        let mut input =
            String::from("Launch a new agent to handle complex, multi-step tasks autonomously. ");
        for _ in 0..6 {
            input.push_str(
                "Each agent runs independently and reports back a single result \
                 when it completes the work it was given. ",
            );
        }
        input.push_str("Available agent types:\n");
        for h in &heads {
            input.push_str(&format!(
                "- {h}: Use this agent when the task requires that speciality, \
                 with full access to the relevant context and the usual set of \
                 capabilities for reading, editing and executing project files \
                 (Tools: Read, Edit, Bash)\n"
            ));
        }
        let out = trunc(&input, 300);
        assert!(out.chars().count() <= 301, "{out}"); // budget + trailing …
        assert!(
            out.starts_with("Launch a new agent to handle complex, multi-step tasks autonomously."),
            "{out}"
        );
        assert!(out.contains("Available agent types:"), "{out}");
        let present = heads.iter().filter(|h| out.contains(*h)).count();
        // All heads, or most heads with the ", …" continuation tail.
        assert!(
            present == heads.len() || (present >= heads.len() / 2 && out.contains(", …")),
            "only {present}/13 heads in: {out}"
        );
        // Heads are never cut inside: every kept head appears verbatim, and
        // item bodies are gone.
        assert!(!out.contains("Use this agent when"), "{out}");
    }

    #[test]
    fn fenced_code_loses_to_constraint_prose() {
        // Identity sentence + constraint prose + long identifier-dense code
        // example: with a tight budget the constraints must win and no
        // fragment of the fence may leak into the output.
        let mut input = String::from(
            "Runs project workflows and reports their status. \
             Long-running flows are queued and surface progress events. \
             You must pass `workflow_id` and `timeout_ms`; paths are resolved \
             relative to the repo root.\n\nExample:\n```js\n",
        );
        for i in 0..20 {
            input.push_str(&format!(
                "const flaky_{i} = await runWorkflow({{ name: 'find-flaky-tests', \
                 retries: {i} }});\nlog(`${{bugs.length}}/10 found`);\n"
            ));
        }
        input.push_str("```\n");
        let out = trunc(&input, 300);
        assert!(
            out.starts_with("Runs project workflows and reports their status."),
            "{out}"
        );
        assert!(out.contains("`workflow_id`"), "{out}");
        assert!(!out.contains("find-flaky-tests"), "{out}");
        assert!(!out.contains("runWorkflow"), "{out}");
        assert!(!out.contains("```"), "{out}");
    }

    #[test]
    fn unclosed_fence_is_atomic_to_end() {
        // A trailing unclosed fence extends to the end of the text: nothing
        // inside it may be selected as a sentence fragment.
        let input = "Tool identity sentence here. \
            Use `apply_patch` only on tracked files.\n\
            ```\ngit commit -m \"msg\"\n\nCo-Authored-By: Bot <bot@example.com>\n";
        let out = trunc(input, 90);
        assert!(out.contains("`apply_patch`"), "{out}");
        assert!(!out.contains("Co-Authored-By"), "{out}");
        assert!(!out.contains("git commit"), "{out}");
    }

    #[test]
    fn unfenced_indented_example_loses_to_constraint_prose() {
        // Same shape as the fenced case but with no fences at all: the code
        // example is only signalled by indentation. It must stay atomic and
        // lowest-tier — no fragment may win the budget over constraints.
        let mut input = String::from(
            "Runs project workflows and reports their status. \
             You must pass `workflow_id` and `timeout_ms`; paths are resolved \
             relative to the repo root.\n\nExample:\n",
        );
        for i in 0..20 {
            input.push_str(&format!(
                "  const flaky_{i} = await runWorkflow({{ name: 'find-flaky-tests' }})\n    \
                 log(`${{bugs.length}}/10 found`)\n"
            ));
        }
        let out = trunc(&input, 200);
        assert!(out.contains("`workflow_id`"), "{out}");
        assert!(!out.contains("find-flaky-tests"), "{out}");
        assert!(!out.contains("bugs.length"), "{out}");
    }

    #[test]
    fn real_workflow_description_drops_indented_js() {
        let input = include_str!("../../fixtures/tool_desc_workflow.txt");
        let out = trunc(input, 300);
        assert!(
            out.starts_with(
                "Execute a workflow script that orchestrates multiple subagents deterministically."
            ),
            "{out}"
        );
        for frag in ["bugs.length", "find-flaky-tests", "log(`"] {
            assert!(!out.contains(frag), "{frag} leaked: {out}");
        }
        // At least one behavioral constraint sentence beyond the identity.
        assert!(
            out.contains("Required fields: `name`, `description`."),
            "{out}"
        );
    }

    #[test]
    fn real_bash_description_drops_commit_boilerplate() {
        let input = include_str!("../../fixtures/tool_desc_bash.txt");
        let out = trunc(input, 300);
        assert!(
            out.starts_with("Executes a given bash command and returns its output."),
            "{out}"
        );
        for frag in ["Co-Authored-By", "Generated with"] {
            assert!(!out.contains(frag), "{frag} leaked: {out}");
        }
        // Retains at least one behavioral constraint sentence.
        assert!(out.chars().filter(|c| *c == '.').count() >= 2, "{out}");
    }

    #[test]
    fn real_compact_bash_description_drops_column_zero_continuations() {
        // Continuation lines at the SAME indentation as their bullets
        // (column 0) must stay inside the item unit: boilerplate payload
        // elides while behavioral constraints survive.
        let input = include_str!("../../fixtures/tool_desc_bash_compact.txt");
        let out = trunc(input, 300);
        assert!(
            out.starts_with("Executes a bash command and returns its output."),
            "{out}"
        );
        for frag in ["Co-Authored-By", "Generated with"] {
            assert!(!out.contains(frag), "{frag} leaked: {out}");
        }
        assert!(out.contains("IMPORTANT: Avoid using this tool"), "{out}");
    }

    #[test]
    fn salient_marker_line_kept_without_body() {
        // An item is scored on its marker line: a salient marker survives
        // even when the budget cannot hold its body, which elides first.
        let input = "Tool identity sentence here. Some plain filler prose sits in the middle.\n\
            - IMPORTANT: pass `--safe-mode` and `--max-depth=3` to `scan_tree` like:\n\
            scan_tree --safe-mode --max-depth=3 ./src ./tests ./benches ./fixtures\n";
        let out = trunc(input, 110);
        assert!(out.contains("`--safe-mode`"), "{out}");
        assert!(!out.contains("./benches"), "{out}");
    }

    #[test]
    fn indented_bullet_list_still_compacts() {
        // The list-compaction tier must keep claiming indented bullet runs
        // with identifier-shaped heads — they are catalogs, not examples.
        let mut input = String::from("Launch a new agent for the task. Available agent types:\n");
        for h in [
            "general-purpose",
            "code-reviewer",
            "test-runner",
            "doc-writer",
        ] {
            input.push_str(&format!(
                "  - {h}: use this agent whenever the task requires that \
                 speciality, with a deliberately long body so the whole block \
                 cannot fit in the leftover budget at all\n"
            ));
        }
        let out = trunc(&input, 140);
        assert!(out.contains("Available agent types:"), "{out}");
        assert!(out.contains("general-purpose"), "{out}");
        assert!(out.contains("code-reviewer"), "{out}");
        assert!(!out.contains("use this agent whenever"), "{out}");
    }

    #[test]
    fn list_item_continuation_lines_stay_in_body() {
        // Deeper-indented non-bullet lines after a bullet belong to the item
        // body: they must never surface as independent high-scoring units.
        let input = "Tool identity sentence here. Always verify paths before writing.\n\
            Steps:\n\
            - commit-step: create the commit ending with:\n   \
            Co-Authored-By: Bot <bot@example.com>\n   \
            🤖 Generated with [Tool](https://example.com)\n\
            - verify-step: run status to confirm success\n";
        let out = trunc(input, 110);
        assert!(!out.contains("Co-Authored-By"), "{out}");
        assert!(!out.contains("Generated with"), "{out}");
    }

    #[test]
    fn repeated_list_heads_deduped() {
        let mut input = String::from("Pick an agent for the task. Agent types:\n");
        for _ in 0..3 {
            input.push_str(
                "- feature-dev: guided feature development with codebase \
                 understanding, architecture focus and a very long body that \
                 prevents the block from fitting whole in the leftover budget\n",
            );
        }
        input.push_str(
            "- code-reviewer: reviews diffs for correctness bugs and reuse \
             cleanups at the requested effort level, also deliberately long\n",
        );
        let out = trunc(&input, 110);
        assert_eq!(out.matches("feature-dev").count(), 1, "{out}");
        assert!(out.contains("code-reviewer"), "{out}");
    }

    #[test]
    fn prose_only_unchanged_by_list_tier() {
        // No list shape anywhere: behavior identical to plain sentence
        // selection (same as before the compaction tier existed).
        let input =
            "First sentence here. Second sentence is longer. Third one overflows the budget.";
        assert_eq!(
            trunc(input, 50),
            "First sentence here. Second sentence is longer.…"
        );
    }

    #[test]
    fn degenerate_giant_word_hard_cuts() {
        let input = "a".repeat(50);
        let out = trunc(&input, 10);
        assert_eq!(out, format!("{}…", "a".repeat(10)));
    }
}
