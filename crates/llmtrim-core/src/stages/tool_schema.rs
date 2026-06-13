//! Stage G — JSON-Schema minification operators for tool/function parameter schemas.
//!
//! Tool schemas are resent on every request and are frequently the largest fixed input
//! cost in an agent loop. TSCG ("Deterministic Tool-Schema Compilation for Agentic LLM
//! Deployments", 2026; "Tool-Schema Compression Enables Agentic RAG Under Constrained
//! Context Budgets", arXiv:2605.26165, 2026) shows a deterministic operator pipeline cuts
//! tool-schema tokens 44–57% with accuracy *improving* on constrained models. Their full
//! pipeline compiles a schema down to typed prose embedded in the prompt.
//!
//! llmtrim forwards **native function-calling** requests, so `tools[].function.parameters`
//! (and Anthropic `input_schema`) MUST stay valid JSON Schema the provider accepts — we
//! cannot replace a schema with prose. This module implements the **API-safe subset**:
//! minify the schema *in place*, dropping only semantic no-ops and advisory text, never
//! restructuring it into something the provider would reject.
//!
//! Operators (all semantics-preserving for native function-calling):
//!   - drop annotation no-ops: `$schema`, `title`, `examples`, `$comment`, empty
//!     `description` strings (advisory metadata the model doesn't dispatch on);
//!   - collapse single-element type arrays (`["string"]` → `"string"`);
//!   - deduplicate identical property descriptions repeated verbatim across properties
//!     (keep the first, drop exact later dups — the same advisory-text trade
//!     `tool_trim_desc` already makes, and gate-protected);
//!   - trim each per-property `description` with the same length cap Stage G applies to
//!     top-level tool descriptions ([`crate::provider::truncate_chars`]).
//!
//! Explicitly **kept** because they are semantic, not annotation: `additionalProperties:
//! false` (strict-mode contract), `default` (provider may inject it), `required`, `enum`,
//! `type`, `properties`, `format`, numeric/length bounds.
//!
//! Determinism: serde_json is built with `preserve_order`, so object key order is stable;
//! every operator either removes keys or rewrites a value, never re-inserts in a new order,
//! so the serialized bytes are identical across runs.

use serde_json::{Map, Value};

/// Annotation keys with no effect on native function-calling dispatch — pure documentation
/// or tooling metadata. Dropped wherever they appear in the schema tree. `title` is dropped
/// at the root *and* per-property (MCP servers emit a `title` on every property by default).
/// Note `description` is handled separately (kept, but trimmed/deduplicated) — it is the one
/// advisory field the model can actually use, so we shrink rather than drop it.
const NOOP_KEYS: [&str; 4] = ["$schema", "title", "examples", "$comment"];

/// Minify one tool parameter schema in place with the API-safe operator set. `max_desc_chars`
/// is the per-property description cap (shared with Stage G's top-level trim). Returns nothing —
/// mutates `schema`. Safe on any `Value`: non-object schemas (a bare `true`/`false`, or the
/// empty `{}` many tools use) are left untouched.
pub(crate) fn minify_schema(schema: &mut Value, max_desc_chars: usize) {
    minify_value(schema, max_desc_chars);
}

/// Recursively minify a schema node. Drives the operators top-down, recursing into the
/// containers JSON Schema uses to nest subschemas (`properties`, `items`, `$defs`,
/// `definitions`, and the combinators `allOf`/`anyOf`/`oneOf`).
fn minify_value(node: &mut Value, max_desc_chars: usize) {
    match node {
        Value::Object(obj) => minify_object(obj, max_desc_chars),
        Value::Array(arr) => {
            for v in arr.iter_mut() {
                minify_value(v, max_desc_chars);
            }
        }
        _ => {}
    }
}

/// Minify a schema object: drop no-op keys, collapse single-type arrays, trim its own
/// `description`, then recurse into nested subschemas — deduplicating identical property
/// descriptions among direct children of a `properties` map.
fn minify_object(obj: &mut Map<String, Value>, max_desc_chars: usize) {
    // 1. Drop annotation no-op keys (documentation/tooling metadata, never dispatched on).
    for k in NOOP_KEYS {
        obj.remove(k);
    }

    // 2. Drop an empty `description` (advises nothing); otherwise trim it to the cap, reusing
    //    Stage G's top-level-description policy so per-property text obeys the same budget.
    if let Some(Value::String(d)) = obj.get_mut("description") {
        if d.trim().is_empty() {
            obj.remove("description");
        } else {
            crate::provider::truncate_chars(d, max_desc_chars);
        }
    }

    // 3. Collapse a single-element `type` array to the scalar (`["string"]` → `"string"`).
    //    A multi-element union (`["string","null"]`) is semantic and left as-is.
    if let Some(Value::Array(types)) = obj.get("type")
        && types.len() == 1
        && types[0].is_string()
    {
        let only = types[0].clone();
        obj.insert("type".to_string(), only);
    }

    // 4. Recurse into nested subschemas. `properties` (and `patternProperties`) is special:
    //    its direct children are sibling property schemas, so dedup their descriptions there.
    for key in ["properties", "patternProperties", "$defs", "definitions"] {
        if let Some(Value::Object(props)) = obj.get_mut(key) {
            dedup_descriptions(props);
            for v in props.values_mut() {
                minify_value(v, max_desc_chars);
            }
        }
    }
    // `items` is a subschema or a list of subschemas; combinators hold subschema arrays.
    for key in [
        "items",
        "additionalItems",
        "contains",
        "not",
        "propertyNames",
    ] {
        if let Some(v) = obj.get_mut(key) {
            minify_value(v, max_desc_chars);
        }
    }
    for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
        if let Some(Value::Array(arr)) = obj.get_mut(key) {
            for v in arr.iter_mut() {
                minify_value(v, max_desc_chars);
            }
        }
    }
    // `additionalProperties` may itself be a subschema object (not just the strict-mode bool,
    // which we keep verbatim). Recurse only into the object form.
    if let Some(ap @ Value::Object(_)) = obj.get_mut("additionalProperties") {
        minify_value(ap, max_desc_chars);
    }
}

/// Drop exact-duplicate `description` strings among the direct children of a `properties`
/// map: keep the first occurrence (in the map's preserved order), remove the identical
/// `description` key from each later property carrying the same verbatim text. Targets the
/// common MCP shape where dozens of parameters repeat one boilerplate sentence. Trimming
/// happens afterward on the survivors, so the cap still applies. Pre-trim comparison is on
/// the original text, which is correct: equal originals trim to equal results anyway.
fn dedup_descriptions(props: &mut Map<String, Value>) {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for prop in props.values_mut() {
        let Some(obj) = prop.as_object_mut() else {
            continue;
        };
        let dup = match obj.get("description").and_then(Value::as_str) {
            Some(d) if !d.trim().is_empty() => !seen.insert(d.to_string()),
            _ => false,
        };
        if dup {
            obj.remove("description");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Words approximate tokens closely enough for a reduction assertion (the same proxy
    /// `tests/common` uses); the real tokenizer only counts fewer for the dropped JSON
    /// punctuation, so this under-states the win if anything.
    fn count_tokens(v: &Value) -> usize {
        serde_json::to_string(v).unwrap().split_whitespace().count()
    }

    #[test]
    fn drops_annotation_noops_keeps_semantic() {
        let mut s = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "QueryParams",
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "q": {"type": "string", "title": "Query", "description": "the search text"},
                "limit": {"type": "integer", "default": 10, "examples": [5, 20]}
            },
            "required": ["q"]
        });
        minify_schema(&mut s, 300);
        // No-ops gone.
        assert!(s.get("$schema").is_none());
        assert!(s.get("title").is_none());
        assert!(s.pointer("/properties/q/title").is_none());
        assert!(s.pointer("/properties/limit/examples").is_none());
        // Semantics kept verbatim.
        assert_eq!(s.get("additionalProperties"), Some(&json!(false)));
        assert_eq!(s.pointer("/properties/limit/default"), Some(&json!(10)));
        assert_eq!(s.get("required"), Some(&json!(["q"])));
        assert_eq!(
            s.pointer("/properties/q/description")
                .and_then(Value::as_str),
            Some("the search text")
        );
    }

    #[test]
    fn collapses_single_element_type_array() {
        let mut s = json!({"type": ["string"]});
        minify_schema(&mut s, 300);
        assert_eq!(s.get("type"), Some(&json!("string")));
        // A real union is preserved.
        let mut u = json!({"type": ["string", "null"]});
        minify_schema(&mut u, 300);
        assert_eq!(u.get("type"), Some(&json!(["string", "null"])));
    }

    #[test]
    fn dedups_repeated_property_descriptions() {
        let boiler = "Set to true to enable this option. Defaults to false when omitted.";
        let mut s = json!({
            "type": "object",
            "properties": {
                "a": {"type": "boolean", "description": boiler},
                "b": {"type": "boolean", "description": boiler},
                "c": {"type": "boolean", "description": boiler},
                "d": {"type": "boolean", "description": "A genuinely distinct note."}
            }
        });
        minify_schema(&mut s, 300);
        // First survives, later exact dups dropped, distinct one kept.
        assert_eq!(
            s.pointer("/properties/a/description")
                .and_then(Value::as_str),
            Some(boiler)
        );
        assert!(s.pointer("/properties/b/description").is_none());
        assert!(s.pointer("/properties/c/description").is_none());
        assert_eq!(
            s.pointer("/properties/d/description")
                .and_then(Value::as_str),
            Some("A genuinely distinct note.")
        );
    }

    #[test]
    fn trims_per_property_description_to_cap() {
        let long = "x".repeat(400);
        let mut s = json!({
            "type": "object",
            "properties": {"p": {"type": "string", "description": long}}
        });
        minify_schema(&mut s, 50);
        let d = s
            .pointer("/properties/p/description")
            .and_then(Value::as_str)
            .unwrap();
        assert!(d.chars().count() <= 51, "trimmed to cap + ellipsis");
    }

    #[test]
    fn drops_empty_description() {
        let mut s = json!({"type": "object", "description": "   ",
            "properties": {"p": {"type": "string", "description": ""}}});
        minify_schema(&mut s, 300);
        assert!(s.get("description").is_none());
        assert!(s.pointer("/properties/p/description").is_none());
    }

    #[test]
    fn recurses_into_nested_subschemas() {
        let mut s = json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "title": "Item",
                        "properties": {"id": {"type": ["integer"], "$comment": "drop me"}}}
                }
            }
        });
        minify_schema(&mut s, 300);
        assert!(s.pointer("/properties/items/items/title").is_none());
        assert!(
            s.pointer("/properties/items/items/properties/id/$comment")
                .is_none()
        );
        assert_eq!(
            s.pointer("/properties/items/items/properties/id/type"),
            Some(&json!("integer")),
            "single-type array collapsed deep in the tree"
        );
    }

    #[test]
    fn strict_mode_schema_survives_untouched_in_semantics() {
        // additionalProperties:false + defaults: a strict-mode schema must round-trip with
        // every semantic field byte-identical (only annotation/advisory noise removed).
        let strict = json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "mode": {"type": "string", "enum": ["fast", "slow"], "default": "fast"},
                "retries": {"type": "integer", "default": 3, "minimum": 0, "maximum": 9}
            },
            "required": ["mode"]
        });
        let mut s = strict.clone();
        minify_schema(&mut s, 300);
        // Nothing annotation-only was present, so the schema is unchanged.
        assert_eq!(s, strict, "strict schema is semantics-stable");
    }

    #[test]
    fn deterministic_byte_stable_across_runs() {
        let make = || {
            json!({
                "$schema": "x", "title": "T", "type": "object",
                "properties": {
                    "a": {"type": ["string"], "description": "same", "title": "A"},
                    "b": {"type": "string", "description": "same"}
                }
            })
        };
        let mut a = make();
        let mut b = make();
        minify_schema(&mut a, 300);
        minify_schema(&mut b, 300);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
            "minification is byte-stable"
        );
    }

    #[test]
    fn round_trips_as_valid_json() {
        let mut s = json!({
            "$schema": "x", "type": "object",
            "properties": {"q": {"type": ["string"], "description": "text"}},
            "required": ["q"], "additionalProperties": false
        });
        minify_schema(&mut s, 300);
        let bytes = serde_json::to_string(&s).unwrap();
        let reparsed: Value = serde_json::from_str(&bytes).unwrap();
        assert_eq!(reparsed, s, "minified schema is valid, stable JSON");
    }

    #[test]
    fn token_reduction_on_verbose_fixture() {
        // A verbose MCP-style schema: $schema, per-property titles, repeated boilerplate
        // descriptions, single-element type arrays. Minify must cut ≥30% of tokens.
        let boiler = "Optional. Provide a value to override the server default for this field.";
        let before = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "ListResourcesArguments",
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "cursor":   {"type": ["string"],  "title": "Cursor",   "description": boiler},
                "limit":    {"type": ["integer"], "title": "Limit",    "description": boiler, "default": 50},
                "filter":   {"type": ["string"],  "title": "Filter",   "description": boiler},
                "order":    {"type": ["string"],  "title": "Order",    "description": boiler, "enum": ["asc", "desc"]},
                "verbose":  {"type": ["boolean"], "title": "Verbose",  "description": boiler, "default": false},
                "fields":   {"type": ["string"],  "title": "Fields",   "description": boiler}
            },
            "required": ["cursor"]
        });
        let mut after = before.clone();
        minify_schema(&mut after, 300);
        let (b, a) = (count_tokens(&before), count_tokens(&after));
        let saved = 100.0 - (a as f64 / b as f64 * 100.0);
        assert!(
            saved >= 30.0,
            "expected ≥30% schema token reduction, got {saved:.1}%"
        );
        // Structural validity + semantics intact after the cut.
        assert_eq!(after.get("additionalProperties"), Some(&json!(false)));
        assert_eq!(after.get("required"), Some(&json!(["cursor"])));
        assert_eq!(after.pointer("/properties/limit/default"), Some(&json!(50)));
        assert_eq!(
            after.pointer("/properties/order/enum"),
            Some(&json!(["asc", "desc"]))
        );
        assert_eq!(
            after
                .pointer("/properties/cursor/type")
                .and_then(Value::as_str),
            Some("string"),
            "type arrays collapsed"
        );
    }
}
