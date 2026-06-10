//! Stage D — columnar (TOON) serialization of uniform flat record arrays.
//!
//! Per spec §4 / *Notation Matters*: apply columnar encoding ONLY to flat, uniform
//! arrays of records (array of objects, all-scalar values, identical key sets) —
//! keep JSON for nested data and as the source of truth. When at least one segment
//! is encoded, inject the format legend once so the model can read TOON. The token
//! gate reverts the whole stage if the legend cost outweighs the savings.
//!
//! Two shapes are handled: content that is *itself* a uniform array (emitted as raw
//! TOON), and (when `nested`) uniform arrays nested inside a content JSON object —
//! each is replaced in place by a TOON string value, keeping the rest as JSON.
//!
//! This is input-side: the model *reads* TOON and replies normally, so no
//! rehydration entry is recorded. The `from_toon` decoder exists for the lossless
//! round-trip property tests (and output-side columnar in a later phase).

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde_json::{Map, Value};

use crate::config::FORMAT_LEGEND;
use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::Provider;

pub struct SerializeStage {
    pub min_rows: usize,
    pub nested: bool,
    /// Encode a top-level uniform flat array as CSV instead of TOON. CSV drops TOON's
    /// per-row indentation + array header, so it can win on large flat tables; the
    /// gate picks the smaller by reverting if it isn't. Nested arrays still use TOON.
    pub csv: bool,
    /// Flatten records whose nested objects are themselves uniform into dotted-key
    /// columns (`meta.region`), so a once-nested array becomes columnar-encodable.
    /// Information-preserving (no value dropped) but structurally reshaped to dotted
    /// keys — the model reads it, but it is not byte-reversible to the nested form, so
    /// opt-in (like `normalize_unicode`).
    pub flatten: bool,
    /// Partition a *heterogeneous* array of records (differing key sets) into uniform
    /// groups by shape, each emitted as its own TOON table. Regroups rows (order kept
    /// within a group); opt-in.
    pub buckets: bool,
}

impl Transform for SerializeStage {
    fn name(&self) -> &str {
        "serialize-toon"
    }

    fn gate_kind(&self) -> GateKind {
        GateKind::InputTokens
    }

    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        _plan: &mut Vec<PlanEntry>,
    ) -> Result<()> {
        let (mut toon_used, mut csv_used) = (false, false);
        for ptr in crate::cache_zone::compressible_pointers(req, provider) {
            let Some(s) = req.get_str(&ptr).map(str::to_string) else {
                continue;
            };
            let Ok(mut value) = serde_json::from_str::<Value>(&s) else {
                continue;
            };
            // Flatten nested-uniform records to dotted columns first; if it yields a
            // uniform flat array the existing columnar path below encodes it.
            if self.flatten
                && let Some(flat) = flatten_array(&value, self.min_rows)
            {
                value = flat;
            }
            let new_content = if is_uniform_flat_array(&value, self.min_rows) {
                // Whole content is the array. With CSV enabled, FORMAT-ROUTE: encode
                // both TOON and CSV and keep the smaller (char length proxies tokens for
                // structured data; the InputTokens gate still reverts the whole stage if
                // neither beats JSON). Otherwise TOON only.
                let toon = to_toon(&value).context("TOON encode failed")?;
                if self.csv {
                    let csv = to_csv(&value);
                    if csv.len() <= toon.len() {
                        csv_used = true;
                        Some(csv)
                    } else {
                        toon_used = true;
                        Some(toon)
                    }
                } else {
                    toon_used = true;
                    Some(toon)
                }
            } else if is_columnar_array(&value, self.min_rows) {
                // Near-uniform records (a few rows carry extra/missing keys): one CSV
                // table with a *union* header, missing cells left empty. Lossless —
                // nothing dropped — and the big win on real arrays that aren't strictly
                // identical (the case `is_uniform_flat_array` rejects).
                csv_used = true;
                Some(to_csv(&value))
            } else if let Some(blocks) = self
                .buckets
                .then(|| bucket_encode(&value, self.min_rows))
                .flatten()
            {
                // Heterogeneous array → one TOON table per record shape (read via the
                // same TOON legend; multiple tables are already valid TOON).
                toon_used = true;
                Some(blocks)
            } else if self.nested && encode_in_place(&mut value, self.min_rows) {
                // Uniform arrays nested in JSON → TOON string values; rest stays JSON.
                toon_used = true;
                Some(serde_json::to_string(&value).context("reserialize after TOON failed")?)
            } else {
                None
            };
            if let Some(content) = new_content {
                req.set(&ptr, Value::String(content));
            }
        }
        // Inject the format legend(s) once, after encoding, so message indices used
        // above are not shifted mid-loop. The gate measures this added cost.
        if toon_used {
            provider.add_system_instruction(req, FORMAT_LEGEND);
        }
        if csv_used {
            provider.add_system_instruction(req, CSV_LEGEND);
        }
        Ok(())
    }
}

/// True iff `v` is an array of >= `min_rows` objects whose values are all scalars
/// and whose key sets are identical — the case columnar notation actually wins on.
fn is_uniform_flat_array(v: &Value, min_rows: usize) -> bool {
    let Some(arr) = v.as_array() else {
        return false;
    };
    // Need ≥1 row to read a schema from, and columnar never wins on a single row; a
    // user-set `min_rows` of 0 would otherwise index `arr[0]` on an empty array (panic).
    if arr.len() < min_rows.max(1) {
        return false;
    }
    let Some(first) = arr.first() else {
        return false;
    };
    let first_keys = match first.as_object() {
        Some(o) if o.values().all(|x| !x.is_array() && !x.is_object()) => sorted_keys(o),
        _ => return false,
    };
    for item in &arr[1..] {
        let Some(o) = item.as_object() else {
            return false;
        };
        if o.values().any(|x| x.is_array() || x.is_object()) {
            return false;
        }
        if sorted_keys(o) != first_keys {
            return false;
        }
    }
    true
}

/// If `v` is an array of ≥ `min_rows` records whose nested objects flatten to one
/// consistent set of dotted-key scalar columns, return the flattened array (which the
/// caller feeds to the columnar path). `None` when nothing is nested (already flat — no
/// gain), a value is an array, or the rows don't flatten to a uniform shape.
fn flatten_array(v: &Value, min_rows: usize) -> Option<Value> {
    let arr = v.as_array()?;
    if arr.len() < min_rows {
        return None;
    }
    let has_nested = arr.iter().any(|it| {
        it.as_object()
            .is_some_and(|o| o.values().any(Value::is_object))
    });
    if !has_nested {
        return None; // nothing to flatten — leave it to the plain uniform check
    }
    let mut rows = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = item.as_object()?;
        let mut flat = Map::new();
        if !flatten_into(obj, "", &mut flat) {
            return None;
        }
        rows.push(Value::Object(flat));
    }
    let flattened = Value::Array(rows);
    is_uniform_flat_array(&flattened, min_rows).then_some(flattened)
}

/// Flatten `obj` into `out` with dotted keys. Returns `false` (caller bails) if a value
/// is an array — which has no scalar column form — or a dotted key would collide with an
/// existing one (which would silently drop data).
fn flatten_into(obj: &Map<String, Value>, prefix: &str, out: &mut Map<String, Value>) -> bool {
    for (k, v) in obj {
        let key = if prefix.is_empty() {
            k.clone()
        } else {
            format!("{prefix}.{k}")
        };
        match v {
            Value::Object(inner) => {
                if !flatten_into(inner, &key, out) {
                    return false;
                }
            }
            Value::Array(_) => return false,
            scalar => {
                if out.insert(key, scalar.clone()).is_some() {
                    return false;
                }
            }
        }
    }
    true
}

/// Partition a heterogeneous array of flat scalar records into uniform groups by their
/// sorted key signature, emitting each group as its own TOON table (joined by blank
/// lines). Returns `None` unless the array partitions cleanly into ≥2 groups each of
/// ≥ `min_rows` — otherwise the uniform or nested paths fit better. Rows are regrouped
/// (kept in order within a group); no record is dropped.
fn bucket_encode(v: &Value, min_rows: usize) -> Option<String> {
    let arr = v.as_array()?;
    if arr.len() < 2 * min_rows {
        return None;
    }
    let mut groups: Vec<(Vec<&str>, Vec<Value>)> = Vec::new();
    for item in arr {
        let obj = item.as_object()?;
        if obj.values().any(|x| x.is_array() || x.is_object()) {
            return None; // not flat — buckets only handle scalar records
        }
        let sig = sorted_keys(obj);
        match groups.iter_mut().find(|(s, _)| *s == sig) {
            Some((_, rows)) => rows.push(item.clone()),
            None => groups.push((sig, vec![item.clone()])),
        }
    }
    if groups.len() < 2 || groups.iter().any(|(_, rows)| rows.len() < min_rows) {
        return None;
    }
    let blocks = groups
        .into_iter()
        .map(|(_, rows)| to_toon(&Value::Array(rows)))
        .collect::<Result<Vec<_>>>()
        .ok()?;
    Some(blocks.join("\n\n"))
}

/// Recursively replace every uniform-flat array inside `value` with its TOON string
/// encoding (a string the legend tells the model to read as TOON). Returns whether
/// any array was encoded.
fn encode_in_place(value: &mut Value, min_rows: usize) -> bool {
    if is_uniform_flat_array(value, min_rows) {
        if let Ok(toon) = to_toon(value) {
            *value = Value::String(toon);
            return true;
        }
        return false;
    }
    match value {
        Value::Object(map) => {
            let mut any = false;
            for v in map.values_mut() {
                if encode_in_place(v, min_rows) {
                    any = true;
                }
            }
            any
        }
        Value::Array(arr) => {
            let mut any = false;
            for v in arr.iter_mut() {
                if encode_in_place(v, min_rows) {
                    any = true;
                }
            }
            any
        }
        _ => false,
    }
}

fn sorted_keys(obj: &serde_json::Map<String, Value>) -> Vec<&str> {
    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort_unstable();
    keys
}

fn to_toon(v: &Value) -> Result<String> {
    toon_format::encode_default(v).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Decode TOON back to JSON. Used by the lossless round-trip property tests and by
/// rehydration when output-side columnar is requested in a later phase.
pub fn from_toon(s: &str) -> Result<Value> {
    toon_format::decode_default(s).map_err(|e| anyhow::anyhow!("{e}"))
}

const CSV_LEGEND: &str = include_str!("../../prompts/csv_legend.txt");

/// Encode a flat record array as RFC 4180 CSV with a **union header** (every key seen
/// across the array, frequency-ordered); a row missing a key gets an empty cell, so a few
/// anomaly rows with extra keys don't break it and nothing is dropped. The model reads the
/// table, so no rehydration is recorded.
fn to_csv(v: &Value) -> String {
    let Some(arr) = v.as_array() else {
        return String::new();
    };
    if arr.is_empty() {
        return String::new();
    }
    let keys = union_keys(arr);
    if keys.is_empty() {
        return String::new();
    }
    let mut wtr = csv::WriterBuilder::new()
        .terminator(csv::Terminator::Any(b'\n'))
        .from_writer(Vec::new());
    if wtr.write_record(&keys).is_err() {
        return String::new();
    }
    for row in arr {
        let obj = row.as_object();
        let cells: Vec<String> = keys
            .iter()
            .map(|k| {
                obj.and_then(|o| o.get(*k))
                    .map(scalar_str)
                    .unwrap_or_default()
            })
            .collect();
        if wtr.write_record(&cells).is_err() {
            return String::new();
        }
    }
    match wtr.into_inner() {
        Ok(bytes) => String::from_utf8_lossy(&bytes)
            .trim_end_matches('\n')
            .to_string(),
        Err(_) => String::new(),
    }
}

fn scalar_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Distinct keys across every record, ordered by descending frequency then name — the
/// union schema for the sparse CSV (so an anomaly row's extra keys aren't dropped).
fn union_keys(arr: &[Value]) -> Vec<&str> {
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for item in arr {
        if let Some(obj) = item.as_object() {
            for k in obj.keys() {
                *freq.entry(k.as_str()).or_default() += 1;
            }
        }
    }
    let mut keys: Vec<(&str, usize)> = freq.into_iter().collect();
    keys.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    keys.into_iter().map(|(k, _)| k).collect()
}

/// True iff `v` is a near-uniform flat record array: ≥ `min_rows` all-scalar objects
/// sharing a dominant schema — a "core" of keys (each in ≥80% of rows) covering ≥50% of
/// all distinct keys. This is the case [`is_uniform_flat_array`] rejects (a few rows
/// differ) but a union-header CSV still compresses losslessly.
fn is_columnar_array(v: &Value, min_rows: usize) -> bool {
    let Some(arr) = v.as_array() else {
        return false;
    };
    let n = arr.len();
    if n < min_rows {
        return false;
    }
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for item in arr {
        let Some(obj) = item.as_object() else {
            return false;
        };
        for (k, val) in obj {
            if val.is_array() || val.is_object() {
                return false;
            }
            *freq.entry(k.as_str()).or_default() += 1;
        }
    }
    let total_keys = freq.len();
    total_keys > 0 && freq.values().filter(|&&c| c * 100 >= n * 80).count() * 2 >= total_keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ProviderKind;
    use crate::pipeline;
    use crate::provider::OpenAiProvider;
    use crate::tokenizer::counter_for;
    use serde_json::json;

    fn records(n: usize) -> Value {
        let mut a = Vec::new();
        for i in 0..n {
            let role = if i % 2 == 0 { "admin" } else { "user" };
            a.push(json!({"id": i, "name": format!("user{i}"), "role": role, "active": true}));
        }
        Value::Array(a)
    }

    fn serialize_stage() -> Box<dyn Transform> {
        Box::new(SerializeStage {
            min_rows: 2,
            nested: true,
            csv: false,
            flatten: false,
            buckets: false,
        })
    }

    #[test]
    fn detects_uniform_flat_array() {
        assert!(is_uniform_flat_array(&records(3), 2));
        assert!(!is_uniform_flat_array(&records(1), 2), "below min_rows");
        assert!(
            !is_uniform_flat_array(&json!([{"a":1},{"a":[1,2]}]), 2),
            "nested value"
        );
        assert!(
            !is_uniform_flat_array(&json!([{"a":1},{"b":2}]), 2),
            "non-uniform keys"
        );
        assert!(!is_uniform_flat_array(&json!({"a":1}), 2), "not an array");
    }

    #[test]
    fn empty_array_with_min_rows_zero_does_not_panic() {
        // `min_rows` is user-settable; 0 on an empty array must not index `arr[0]` (panic).
        assert!(
            !is_uniform_flat_array(&json!([]), 0),
            "empty array is never columnar"
        );
        // A `min_rows` of 0 is floored to 1: an empty array still fails, no panic.
        assert!(!is_uniform_flat_array(&Value::Array(vec![]), 0));
    }

    #[test]
    fn toon_round_trip_is_lossless() {
        let v = records(10);
        let toon = to_toon(&v).unwrap();
        let back = from_toon(&toon).unwrap();
        assert_eq!(back, v, "TOON must round-trip losslessly");
    }

    #[test]
    fn toon_output_format_snapshot() {
        // Locks the exact TOON output shape so a codec change is caught (the legend
        // we ship to the model must keep matching this format).
        let v = json!([{"city": "paris", "pop": 2}, {"city": "lyon", "pop": 1}]);
        let toon = to_toon(&v).unwrap();
        assert_eq!(toon, "[2]{city,pop}:\n  paris,2\n  lyon,1");
    }

    #[test]
    fn serialize_stage_reduces_tokens_and_round_trips() {
        let arr = records(25);
        let content = serde_json::to_string(&arr).unwrap();
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}],"max_tokens":200});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages = vec![serialize_stage()];

        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(
            out.stages[0].applied,
            "should apply on a 25-row uniform array (savings > legend)"
        );
        assert!(
            out.input_tokens_after < out.input_tokens_before,
            "net token win including the legend"
        );
        // Legend inserted at messages[0]; original user content is now at [1] as TOON.
        let encoded = req.get_str("/messages/1/content").unwrap();
        let back = from_toon(encoded).unwrap();
        assert_eq!(
            back, arr,
            "encoded content decodes back to the original array"
        );
    }

    #[test]
    fn nested_array_encodes_in_place_lossless() {
        // Nested TOON is stored as a JSON-escaped string, so its per-row efficiency
        // is lower than raw top-level TOON; the array must be large enough to beat
        // the legend cost (break-even ~20 rows here).
        let arr = records(40);
        let wrapper = json!({"results": arr.clone(), "total": 40, "page": 1});
        let content = serde_json::to_string(&wrapper).unwrap();
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}],"max_tokens":200});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages = vec![serialize_stage()];

        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied, "nested uniform array should encode");

        // content now at messages[1] (legend at 0); the wrapper stays JSON, the
        // `results` field became a TOON string.
        let encoded = req.get_str("/messages/1/content").unwrap();
        let v: Value = serde_json::from_str(encoded).expect("wrapper is still valid JSON");
        assert_eq!(
            v.get("total"),
            Some(&json!(40)),
            "non-array fields preserved"
        );
        assert_eq!(v.get("page"), Some(&json!(1)));
        let results = v
            .get("results")
            .and_then(Value::as_str)
            .expect("results is now a TOON string");
        assert_eq!(from_toon(results).unwrap(), arr, "nested array round-trips");
    }

    #[test]
    fn nested_disabled_leaves_wrapper_json() {
        let wrapper = json!({"results": records(8), "total": 8});
        let content = serde_json::to_string(&wrapper).unwrap();
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(SerializeStage {
            min_rows: 2,
            nested: false,
            csv: false,
            flatten: false,
            buckets: false,
        })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(!out.stages[0].applied, "nested disabled => no encoding");
    }

    #[test]
    fn csv_encoding_for_flat_array() {
        let arr = records(25);
        let content = serde_json::to_string(&arr).unwrap();
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}],"max_tokens":200});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(SerializeStage {
            min_rows: 2,
            nested: true,
            csv: true,
            flatten: false,
            buckets: false,
        })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(
            out.stages[0].applied,
            "CSV cuts tokens on a 25-row flat array"
        );
        // CSV legend inserted at messages[0]; encoded content now at [1].
        let encoded = req.get_str("/messages/1/content").unwrap();
        assert!(
            encoded.starts_with("active,id,name,role"),
            "CSV header row first"
        );
        assert!(!encoded.contains('{'), "no JSON braces remain");
    }

    #[test]
    fn format_routing_keeps_smaller_encoding() {
        let arr = records(25);
        let smaller = to_toon(&arr).unwrap().len().min(to_csv(&arr).len());
        let content = serde_json::to_string(&arr).unwrap();
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}],"max_tokens":200});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(SerializeStage {
            min_rows: 2,
            nested: true,
            csv: true,
            flatten: false,
            buckets: false,
        })];
        pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        let encoded = req.get_str("/messages/1/content").unwrap();
        assert_eq!(
            encoded.len(),
            smaller,
            "format-routing keeps the smaller of TOON/CSV"
        );
    }

    /// Records with a uniform nested `meta` object.
    fn nested_records(n: usize) -> Value {
        let mut a = Vec::new();
        for i in 0..n {
            let region = if i % 2 == 0 { "eu" } else { "us" };
            a.push(json!({"id": i, "meta": {"region": region, "tier": i % 3}}));
        }
        Value::Array(a)
    }

    /// A heterogeneous array: two record shapes, `n` of each.
    fn mixed_records(n: usize) -> Value {
        let mut a = Vec::new();
        for i in 0..n {
            a.push(json!({"id": i, "name": format!("u{i}"), "active": true}));
        }
        for i in 0..n {
            a.push(json!({"id": i, "code": 500, "msg": format!("err{i}")}));
        }
        Value::Array(a)
    }

    #[test]
    fn flatten_dots_nested_uniform_records() {
        let flat = flatten_array(&nested_records(4), 2).expect("nested-uniform flattens");
        assert!(
            is_uniform_flat_array(&flat, 2),
            "flattened result is columnar-ready"
        );
        let first = flat.as_array().unwrap()[0].as_object().unwrap();
        assert!(first.contains_key("meta.region"), "dotted column present");
        assert!(first.contains_key("meta.tier"));
        assert!(!first.contains_key("meta"), "nested key replaced");
    }

    #[test]
    fn flatten_declines_when_nothing_nested_or_value_is_array() {
        assert!(
            flatten_array(&records(3), 2).is_none(),
            "already flat → no gain"
        );
        let with_array = json!([{"id": 1, "tags": [1, 2]}, {"id": 2, "tags": [3]}]);
        assert!(
            flatten_array(&with_array, 2).is_none(),
            "array value can't be a column"
        );
    }

    #[test]
    fn flatten_stage_encodes_nested_records() {
        // Enough rows that the columnar savings clear the one-time TOON legend cost.
        let content = serde_json::to_string(&nested_records(40)).unwrap();
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}],"max_tokens":200});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(SerializeStage {
            min_rows: 2,
            nested: false,
            csv: false,
            flatten: true,
            buckets: false,
        })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied, "flatten + TOON beats nested JSON");
        assert!(out.input_tokens_after < out.input_tokens_before);
        let encoded = req.get_str("/messages/1/content").unwrap();
        assert!(
            encoded.contains("meta.region"),
            "TOON header carries dotted columns"
        );
    }

    #[test]
    fn buckets_partition_heterogeneous_records() {
        let blocks = bucket_encode(&mixed_records(3), 2).expect("two shapes partition");
        // Two TOON tables, one per shape, separated by a blank line.
        assert_eq!(blocks.matches("]{").count(), 2, "one TOON header per shape");
        assert!(blocks.contains("name"), "first shape's columns present");
        assert!(blocks.contains("msg"), "second shape's columns present");
    }

    #[test]
    fn buckets_decline_on_uniform_array() {
        // A single shape is the uniform path's job, not buckets.
        assert!(bucket_encode(&records(6), 2).is_none());
    }

    #[test]
    fn near_uniform_array_is_columnar_and_unions_keys() {
        let mut a: Vec<Value> = (0..9).map(|i| json!({"id": i, "status": "ok"})).collect();
        a.push(json!({"id": 9, "status": "error", "detail": "boom"}));
        let arr = Value::Array(a);
        assert!(
            !is_uniform_flat_array(&arr, 2),
            "anomaly row breaks strict uniformity"
        );
        assert!(
            is_columnar_array(&arr, 2),
            "but it's columnar (shared core schema)"
        );
        let csv = to_csv(&arr);
        assert!(
            csv.lines().next().unwrap().contains("detail"),
            "union header keeps the extra key: {csv}"
        );
        assert!(csv.contains("boom"), "anomaly value survives (lossless)");
    }

    #[test]
    fn union_csv_stage_compresses_near_uniform_array() {
        let mut a: Vec<Value> = (0..40)
            .map(|i| json!({"ts": i, "level": "INFO", "msg": format!("event {i}")}))
            .collect();
        a[7] = json!({"ts": 7, "level": "ERROR", "msg": "failed", "code": 500});
        let content = serde_json::to_string(&Value::Array(a)).unwrap();
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}],"max_tokens":200});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let out = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[serialize_stage()],
        );
        assert!(out.stages[0].applied, "near-uniform array encoded");
        assert!(out.input_tokens_after < out.input_tokens_before);
        let encoded = req.get_str("/messages/1/content").unwrap();
        assert!(
            encoded.contains("code"),
            "the anomaly's extra column survives (lossless)"
        );
    }

    #[test]
    fn buckets_stage_reduces_tokens_on_mixed_array() {
        // Enough rows per shape that two TOON tables clear the one-time legend cost.
        let content = serde_json::to_string(&mixed_records(20)).unwrap();
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}],"max_tokens":200});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(SerializeStage {
            min_rows: 2,
            nested: false,
            csv: false,
            flatten: false,
            buckets: true,
        })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(
            out.stages[0].applied,
            "bucketed TOON beats heterogeneous JSON"
        );
        assert!(out.input_tokens_after < out.input_tokens_before);
    }
}
