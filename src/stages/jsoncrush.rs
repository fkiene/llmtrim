//! Lossy down-sampling of large JSON record arrays — keep a representative subset.
//!
//! Serialize (Stage D) re-encodes record arrays losslessly to TOON, but a 10,000-row
//! audit dump is still huge after columnar encoding. This stage *samples* such arrays
//! down to a representative subset before serialize runs: it keeps the first and last
//! rows, every statistical **outlier** (a rare categorical value, or a row carrying an
//! error keyword — the rows that usually matter), and a query-biased sample of the rest
//! up to an adaptive budget, dropping the others. A one-time system note tells the model
//! the arrays were sampled. Lossy, `InputTokens`-gated, `Content`-scoped — and like the
//! other lossy stages it touches only the live (non-cached) zone.
//!
//! Only record arrays (arrays of objects) above the row cap are sampled; smaller arrays
//! and scalar arrays are left for serialize's lossless columnar encoding.

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

use crate::gate::{GateKind, PlanEntry, Scope, Transform};
use crate::ir::Request;
use crate::provider::Provider;
use crate::stages::tools::lex_words;

/// One-time note so the model knows some arrays are representative samples, not complete.
const SAMPLE_NOTE: &str = include_str!("../../prompts/jsoncrush_note.txt");

static ERROR_KEYWORD: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(error|fail(?:ed|ure)?|fatal|panic|exception|denied|invalid|timeout)\b")
        .unwrap()
});

pub struct JsonCrushStage {
    /// Sample record arrays longer than this down to ~this many representative rows.
    pub max_rows: usize,
}

impl Transform for JsonCrushStage {
    fn name(&self) -> &str {
        "json-crush"
    }

    fn gate_kind(&self) -> GateKind {
        GateKind::InputTokens
    }

    fn scope(&self) -> Scope {
        Scope::Content
    }

    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        _plan: &mut Vec<PlanEntry>,
    ) -> Result<()> {
        let pointers = crate::cache_zone::compressible_pointers(req, provider);
        // The "ask" — short segments bias which rows survive.
        let query: HashSet<String> = pointers
            .iter()
            .filter_map(|p| req.get_str(p))
            .filter(|t| t.lines().count() < 4 && t.len() < 600)
            .flat_map(lex_words)
            .collect();

        let mut sampled_any = false;
        for ptr in &pointers {
            let Some(s) = req.get_str(ptr).map(str::to_string) else {
                continue;
            };
            let Ok(mut value) = serde_json::from_str::<Value>(&s) else {
                continue;
            };
            if crush_value(&mut value, self.max_rows, &query) {
                req.set(ptr, Value::String(value.to_string()));
                sampled_any = true;
            }
        }
        if sampled_any {
            provider.add_system_instruction(req, SAMPLE_NOTE);
        }
        Ok(())
    }
}

/// Sample any oversized record array within `value` (the value itself, or arrays nested
/// one level inside an object). Returns whether anything was sampled.
fn crush_value(value: &mut Value, max_rows: usize, query: &HashSet<String>) -> bool {
    if let Some(rows) = crush_array(value, max_rows, query) {
        *value = Value::Array(rows);
        return true;
    }
    // Object with big array fields (e.g. `{"results": [...]}`): sample each in place.
    if let Value::Object(map) = value {
        let mut any = false;
        for field in map.values_mut() {
            if let Some(rows) = crush_array(field, max_rows, query) {
                *field = Value::Array(rows);
                any = true;
            }
        }
        return any;
    }
    false
}

/// Sampled rows for a record array longer than `max_rows`, in original order; `None` if
/// `v` isn't an over-cap array of objects.
fn crush_array(v: &Value, max_rows: usize, query: &HashSet<String>) -> Option<Vec<Value>> {
    let arr = v.as_array()?;
    let n = arr.len();
    if n <= max_rows || !arr.iter().all(Value::is_object) {
        return None;
    }

    // Serialize once per row, reused below for the error scan, rare-value freq map, and
    // query scoring (was `to_string()`d 2–3× per row).
    let serialized: Vec<String> = arr.iter().map(Value::to_string).collect();

    // Fixed sample: keep the first ~60% and last ~20% of the budget, every anomaly, then
    // fill toward the budget by query relevance.
    // A *fixed* keep-count, not information-saturation — so diverse rows still get cut
    // hard (saturation kept ~all of them).
    let k_first = ((max_rows as f64 * 0.6).round() as usize).clamp(1, n);
    let k_last = ((max_rows as f64 * 0.2).round() as usize).clamp(1, (n - k_first).max(1));
    let mut keep = vec![false; n];
    for slot in keep.iter_mut().take(k_first) {
        *slot = true;
    }
    for slot in keep.iter_mut().skip(n - k_last) {
        *slot = true;
    }
    // Outliers (error/rare rows) are bounded by the row budget too — an error-dense array
    // would otherwise "sample" to nearly every row, defeating the cap. Add outliers in
    // order until we hit `max_rows`, so the budget is a hard ceiling.
    let mut count = keep.iter().filter(|&&x| x).count();
    for &i in outlier_rows(arr, &serialized).iter() {
        if count >= max_rows {
            break;
        }
        if !keep[i] {
            keep[i] = true;
            count += 1;
        }
    }

    // Fill the remaining budget by query relevance (ties → original order).
    let scores: Vec<f64> = serialized.iter().map(|r| query_overlap(r, query)).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    for &i in &order {
        if count >= max_rows {
            break;
        }
        if !keep[i] {
            keep[i] = true;
            count += 1;
        }
    }

    // Only report a sample when rows were actually dropped: an all-error array can keep
    // everything, and emitting an unchanged array (plus the "sampled" note) would just add
    // tokens and revert. `None` ⇒ the stage leaves this array (and skips the note).
    if count >= n {
        return None;
    }
    Some(
        arr.iter()
            .zip(&keep)
            .filter(|&(_, &k)| k)
            .map(|(row, _)| row.clone())
            .collect(),
    )
}

/// Rows worth keeping regardless of budget: any row carrying an error keyword, or holding
/// a *rare* value in a categorical field (a value in ≤5% of rows where the field has few
/// distinct values — i.e. a status/level/type, not a unique id). Indices in ascending
/// order, so the caller's budget cap is deterministic. `serialized[i]` is row `i`'s JSON
/// (precomputed once by the caller) — reused for the error scan.
fn outlier_rows(arr: &[Value], serialized: &[String]) -> Vec<usize> {
    let n = arr.len();
    let rare_at = (n / 20).max(1);
    // A field counts as categorical only at low cardinality (a status/level/type), not a
    // spread-out numeric/id field where every value would look "rare".
    let cat_cap = (n / 10).clamp(2, 24);

    // value frequencies per scalar field
    let mut freq: HashMap<&str, HashMap<String, usize>> = HashMap::new();
    for row in arr {
        if let Some(obj) = row.as_object() {
            for (key, val) in obj {
                if is_scalar(val) {
                    *freq
                        .entry(key.as_str())
                        .or_default()
                        .entry(val.to_string())
                        .or_default() += 1;
                }
            }
        }
    }

    let mut out = Vec::new();
    for (i, row) in arr.iter().enumerate() {
        if ERROR_KEYWORD.is_match(&serialized[i]) {
            out.push(i);
            continue;
        }
        let Some(obj) = row.as_object() else { continue };
        for (key, val) in obj {
            if !is_scalar(val) {
                continue;
            }
            if let Some(counts) = freq.get(key.as_str()) {
                let distinct = counts.len();
                if (2..=cat_cap).contains(&distinct)
                    && counts.get(&val.to_string()).copied().unwrap_or(0) <= rare_at
                {
                    out.push(i);
                    break;
                }
            }
        }
    }
    out
}

fn is_scalar(v: &Value) -> bool {
    !v.is_array() && !v.is_object()
}

/// Count of a row's words that appear in the query (0 when no query).
fn query_overlap(row: &str, query: &HashSet<String>) -> f64 {
    if query.is_empty() {
        return 0.0;
    }
    lex_words(row)
        .into_iter()
        .filter(|w| query.contains(w))
        .count() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ProviderKind;
    use crate::pipeline;
    use crate::provider::OpenAiProvider;
    use crate::tokenizer::counter_for;
    use serde_json::json;

    /// `n` log-ish records, all `status:"ok"` except a few rare `status:"error"` rows.
    fn records(n: usize) -> Value {
        let mut a = Vec::new();
        for i in 0..n {
            let status = if i == 7 || i == 900 { "error" } else { "ok" };
            a.push(json!({"id": i, "status": status, "msg": format!("request {i} handled")}));
        }
        Value::Array(a)
    }

    #[test]
    fn samples_big_array_and_keeps_outliers() {
        let arr = records(1000);
        let q = HashSet::new();
        let rows = crush_array(&arr, 50, &q).expect("over-cap array is sampled");
        assert!(rows.len() <= 50, "down to the budget, got {}", rows.len());
        // both rare error rows survive
        let errors = rows.iter().filter(|r| r["status"] == "error").count();
        assert_eq!(errors, 2, "rare error rows are kept as outliers");
        // first and last survive
        assert_eq!(rows.first().unwrap()["id"], 0);
        assert_eq!(rows.last().unwrap()["id"], 999);
    }

    #[test]
    fn outliers_are_capped_to_budget() {
        // An error-dense array: every row is an outlier. The kept count must still respect
        // the row budget instead of "sampling" to nearly the whole array.
        let arr = Value::Array(
            (0..1000)
                .map(|i| json!({"id": i, "status": "error", "msg": format!("fail {i}")}))
                .collect(),
        );
        let q = HashSet::new();
        let rows = crush_array(&arr, 50, &q).expect("over-cap array is sampled");
        assert!(
            rows.len() <= 50,
            "outliers bounded by budget, got {}",
            rows.len()
        );
    }

    #[test]
    fn all_error_array_drops_rows_instead_of_keeping_everything() {
        // The regression: an all-error array used to mark every row an outlier and keep
        // them all, so serialize couldn't shrink it and the stage reverted. Now the kept
        // count is strictly below the row count (rows were actually dropped) and within the
        // budget, so the "sampled" note is honest.
        let n = 1000;
        let arr = Value::Array(
            (0..n)
                .map(|i| json!({"id": i, "status": "error"}))
                .collect(),
        );
        let q = HashSet::new();
        let rows = crush_array(&arr, 50, &q).expect("over-cap array is sampled");
        assert!(rows.len() < n, "rows actually dropped, not all kept");
        assert!(rows.len() <= 50, "within budget");
    }

    #[test]
    fn small_or_scalar_arrays_are_left_alone() {
        let q = HashSet::new();
        assert!(
            crush_array(&records(20), 50, &q).is_none(),
            "below cap → serialize's job"
        );
        assert!(
            crush_array(&json!([1, 2, 3, 4, 5]), 2, &q).is_none(),
            "scalar array → not a record array"
        );
    }

    #[test]
    fn stage_reduces_tokens_on_a_huge_array() {
        let content = serde_json::to_string(&records(1000)).unwrap();
        let body = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": content}], "max_tokens": 100});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(JsonCrushStage { max_rows: 50 })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied, "1000-row array sampled");
        assert!(out.input_tokens_after < out.input_tokens_before);
        // the surviving content still parses and carries the error rows
        let encoded = req.get_str("/messages/1/content").unwrap();
        assert!(
            encoded.contains("\"error\""),
            "error rows survive the sample"
        );
    }

    #[test]
    fn nested_array_field_is_sampled_in_place() {
        let wrapper = json!({"results": records(1000), "total": 1000});
        let mut v = wrapper;
        assert!(
            crush_value(&mut v, 50, &HashSet::new()),
            "nested array sampled"
        );
        assert_eq!(v["total"], 1000, "sibling fields preserved");
        assert!(
            v["results"].as_array().unwrap().len() <= 50,
            "results sampled"
        );
    }
}
