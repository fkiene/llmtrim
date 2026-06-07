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

use anyhow::{Context, Result};
use serde_json::Value;

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
        for ptr in provider.content_text_pointers(req) {
            let Some(s) = req.get_str(&ptr).map(str::to_string) else {
                continue;
            };
            let Ok(mut value) = serde_json::from_str::<Value>(&s) else {
                continue;
            };
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
    if arr.len() < min_rows {
        return false;
    }
    let first_keys = match arr[0].as_object() {
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

/// Encode a uniform flat record array as RFC 4180 CSV (header row + one row per
/// record). Caller guarantees the uniform-flat-array shape. Input-side like TOON:
/// the model reads the table, so no rehydration is recorded.
fn to_csv(v: &Value) -> String {
    let Some(arr) = v.as_array() else {
        return String::new();
    };
    let Some(first) = arr.first().and_then(Value::as_object) else {
        return String::new();
    };
    let keys = sorted_keys(first);
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
        })];
        pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        let encoded = req.get_str("/messages/1/content").unwrap();
        assert_eq!(
            encoded.len(),
            smaller,
            "format-routing keeps the smaller of TOON/CSV"
        );
    }
}
