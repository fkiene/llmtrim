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
use crate::select::{self, Item, Weights};
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
        // Original row counts of *bare top-level* sampled arrays, in pointer order. A bare
        // crushed array has no parent object to carry N as a sibling field, so N is surfaced
        // in the note text (the channel the model already reads). Nested arrays instead get a
        // `_sampled_from_<field>` sibling on their parent object and don't contribute here.
        let mut bare_counts: Vec<usize> = Vec::new();
        for ptr in &pointers {
            let Some(s) = req.get_str(ptr).map(str::to_string) else {
                continue;
            };
            let Ok(mut value) = serde_json::from_str::<Value>(&s) else {
                continue;
            };
            if let Some(bare_n) = crush_value(&mut value, self.max_rows, &query) {
                req.set(ptr, Value::String(value.to_string()));
                sampled_any = true;
                bare_counts.extend(bare_n);
            }
        }
        if sampled_any {
            provider.add_system_instruction(req, &sample_note(&bare_counts));
        }
        Ok(())
    }
}

/// The one-time sample note, with the original row counts of any bare top-level sampled
/// arrays appended so count/aggregate queries can still recover N. Empty `bare_counts`
/// (only nested arrays were sampled, or none had a recoverable count) yields the static note
/// unchanged.
///
/// One shared note serves the whole request, so it can only state the bare counts as a set,
/// not attribute a specific N to a specific array when several bare arrays are sampled across
/// different messages. That ambiguity is acceptable for the simple/common single-array case;
/// nested arrays avoid it entirely via their per-field `_sampled_from_<field>` sibling.
fn sample_note(bare_counts: &[usize]) -> String {
    if bare_counts.is_empty() {
        return SAMPLE_NOTE.to_string();
    }
    let counts = bare_counts
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let plural = if bare_counts.len() == 1 {
        "array's original row count was"
    } else {
        "arrays' original row counts were"
    };
    format!("{} The sampled {plural}: {counts}.", SAMPLE_NOTE.trim_end())
}

/// Sample any oversized record array within `value` (the value itself, or arrays nested
/// one level inside an object).
///
/// Returns `None` if nothing was sampled. Otherwise returns `Some(bare_n)`:
/// - `Some(Some(n))` — `value` itself was a bare top-level array of original length `n`. A
///   bare array has no parent object, so `n` is surfaced in the note text by the caller.
/// - `Some(None)` — only nested array fields were sampled. Each gets a `_sampled_from_<field>`
///   sibling on its parent object here (the parent is already an object, so this adds no array
///   element and changes no type), so the caller needs no count from this value.
fn crush_value(
    value: &mut Value,
    max_rows: usize,
    query: &HashSet<String>,
) -> Option<Option<usize>> {
    if let Some((n, rows)) = crush_array(value, max_rows, query) {
        *value = Value::Array(rows);
        return Some(Some(n));
    }
    // Object with big array fields (e.g. `{"results": [...]}`): sample each in place and leave
    // a `_sampled_from_<field>: n` sibling so count/aggregate queries can still recover N.
    if let Value::Object(map) = value {
        let mut sampled: Vec<(String, usize)> = Vec::new();
        for (key, field) in map.iter_mut() {
            if let Some((n, rows)) = crush_array(field, max_rows, query) {
                *field = Value::Array(rows);
                sampled.push((key.clone(), n));
            }
        }
        if sampled.is_empty() {
            return None;
        }
        for (key, n) in sampled {
            map.insert(format!("_sampled_from_{key}"), Value::from(n));
        }
        return Some(None);
    }
    None
}

/// Sampled rows for a record array longer than `max_rows`, paired with the array's original
/// row count `n` (so callers can surface it for count/aggregate queries); rows are in original
/// order. `None` if `v` isn't an over-cap array of objects, or no rows were dropped.
fn crush_array(v: &Value, max_rows: usize, query: &HashSet<String>) -> Option<(usize, Vec<Value>)> {
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

    // Fill the remaining budget with a query-biased *diverse* sample of the rest:
    // greedy submodular selection (facility-location-style) over each row's value
    // strings, so the kept sample spans distinct rows instead of N near-identical
    // highest-overlap ones. Relevance = query overlap (preserves the query bias),
    // coverage = the row's value bigrams (the diversity term).
    fill_diverse(arr, &serialized, &mut keep, max_rows, query);

    // Only report a sample when rows were actually dropped: an all-error array can keep
    // everything, and emitting an unchanged array (plus the "sampled" note) would just add
    // tokens and revert. `None` ⇒ the stage leaves this array (and skips the note).
    if keep.iter().filter(|&&k| k).count() >= n {
        return None;
    }
    let rows = arr
        .iter()
        .zip(&keep)
        .filter(|&(_, &k)| k)
        .map(|(row, _)| row.clone())
        .collect();
    Some((n, rows))
}

/// Fill the remaining `max_rows` slots in `keep` with a query-biased, diverse sample of the
/// not-yet-kept rows. Greedy submodular selection ([`crate::select`]) over each row's value
/// strings: relevance is the row's query overlap (the existing bias), coverage is its value
/// bigrams (facility-location-style diversity — Lin & Bilmes, ACL 2011; Chen et al., NeurIPS
/// 2018). Each candidate row costs one slot, so the budget is the leftover row count; this
/// preserves the first/last/outlier rows already pinned in `keep`.
fn fill_diverse(
    arr: &[Value],
    serialized: &[String],
    keep: &mut [bool],
    max_rows: usize,
    query: &HashSet<String>,
) {
    let used = keep.iter().filter(|&&k| k).count();
    let remaining = max_rows.saturating_sub(used);
    if remaining == 0 {
        return;
    }
    // Candidate pool = rows not already pinned. Diversity is computed over this pool only
    // (the saturation ceilings come from the candidates), so the fill spans distinct rows.
    let candidates: Vec<usize> = (0..arr.len()).filter(|&i| !keep[i]).collect();
    let items: Vec<Item> = candidates
        .iter()
        .map(|&i| {
            let rel = query_overlap(&serialized[i], query);
            Item::from_text(&row_value_text(&arr[i]), 1, rel)
        })
        .collect();
    for local in select::select(&items, remaining, &Weights::default()) {
        keep[candidates[local]] = true;
    }
}

/// A row's **value** strings joined into one text — the features the diverse sample spans.
/// Only values are used (object keys are shared across rows and carry no row-distinguishing
/// signal); nested objects/arrays are flattened to their scalar leaves. Strings contribute
/// their text, numbers/bools their literal — universal, no language assumptions.
fn row_value_text(row: &Value) -> String {
    let mut out = String::new();
    collect_scalar_values(row, &mut out);
    out
}

/// Append every scalar leaf of `v` (string text or number/bool literal) to `out`, space-
/// separated, recursing into objects (values only) and arrays.
fn collect_scalar_values(v: &Value, out: &mut String) {
    match v {
        Value::String(s) => {
            out.push_str(s);
            out.push(' ');
        }
        Value::Number(_) | Value::Bool(_) => {
            out.push_str(&v.to_string());
            out.push(' ');
        }
        Value::Array(a) => {
            for e in a {
                collect_scalar_values(e, out);
            }
        }
        Value::Object(m) => {
            for val in m.values() {
                collect_scalar_values(val, out);
            }
        }
        Value::Null => {}
    }
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
        let (n, rows) = crush_array(&arr, 50, &q).expect("over-cap array is sampled");
        assert_eq!(n, 1000, "original row count reported");
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
        let (_, rows) = crush_array(&arr, 50, &q).expect("over-cap array is sampled");
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
        let (_, rows) = crush_array(&arr, 50, &q).expect("over-cap array is sampled");
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
        assert_eq!(
            crush_value(&mut v, 50, &HashSet::new()),
            Some(None),
            "nested array sampled (no bare count surfaced in the note)"
        );
        assert_eq!(v["total"], 1000, "sibling fields preserved");
        assert!(
            v["results"].as_array().unwrap().len() <= 50,
            "results sampled"
        );
    }

    #[test]
    fn nested_array_gets_sampled_from_sibling_with_count() {
        // A nested array field leaves an `_sampled_from_<field>: N` sibling on the parent
        // object (which is already an object, so no element is added and no type changes), so
        // count/aggregate queries can recover the original row count.
        let mut v = json!({"results": records(1000)});
        assert_eq!(
            crush_value(&mut v, 50, &HashSet::new()),
            Some(None),
            "nested array sampled"
        );
        assert_eq!(
            v["_sampled_from_results"], 1000,
            "sibling carries the original row count"
        );
        assert!(
            v["results"].as_array().unwrap().len() <= 50,
            "results sampled, still an array of objects"
        );
        assert!(v["results"].is_array(), "field stays an array");
    }

    #[test]
    fn bare_top_level_array_surfaces_count_in_note() {
        // A bare top-level array has no parent object to carry a sibling, so `crush_value`
        // reports the original count for the caller to surface in the note text. The value
        // itself stays a bare array (no wrapper object, no injected sentinel element).
        let mut v = records(1000);
        assert_eq!(
            crush_value(&mut v, 50, &HashSet::new()),
            Some(Some(1000)),
            "bare array reports its original row count"
        );
        assert!(v.is_array(), "stays a bare array");
        assert!(v.as_array().unwrap().len() <= 50, "sampled");

        // The note text then carries that count.
        let note = sample_note(&[1000]);
        assert!(note.contains("1000"), "note surfaces N: {note}");
    }

    #[test]
    fn sample_note_is_static_without_bare_counts() {
        assert_eq!(
            sample_note(&[]),
            SAMPLE_NOTE,
            "no bare counts → unchanged static note"
        );
    }

    #[test]
    fn sample_note_uses_plural_wording_for_multiple_bare_counts() {
        // Two bare arrays in one request: the shared note can only state the counts as a set,
        // not attribute each to its array, so it uses plural wording and comma-joins them.
        let note = sample_note(&[100, 200]);
        assert!(
            note.contains("arrays' original row counts were"),
            "plural wording: {note}"
        );
        assert!(
            note.contains("100, 200"),
            "both counts comma-joined: {note}"
        );
    }

    #[test]
    fn multiple_nested_array_fields_each_get_their_own_sibling() {
        // Two over-cap array fields on one object: each gets its own `_sampled_from_<field>`
        // sibling carrying that field's original count; neither field changes type.
        let mut v = json!({"a": records(1000), "b": records(800)});
        assert_eq!(
            crush_value(&mut v, 50, &HashSet::new()),
            Some(None),
            "only nested fields sampled"
        );
        assert_eq!(v["_sampled_from_a"], 1000, "field a's original count");
        assert_eq!(v["_sampled_from_b"], 800, "field b's original count");
        assert!(
            v["a"].as_array().unwrap().len() <= 50,
            "a sampled, still array"
        );
        assert!(
            v["b"].as_array().unwrap().len() <= 50,
            "b sampled, still array"
        );
    }

    #[test]
    fn mixed_bare_and_nested_arrays_accumulate_independently() {
        // A request with a bare top-level array AND an object holding a nested array: the bare
        // one reports its count for the note, the nested one gets a sibling, and the two paths
        // don't interfere.
        let mut bare = records(1000);
        let mut nested = json!({"rows": records(900)});
        let mut bare_counts: Vec<usize> = Vec::new();
        bare_counts.extend(crush_value(&mut bare, 50, &HashSet::new()).flatten());
        bare_counts.extend(crush_value(&mut nested, 50, &HashSet::new()).flatten());

        assert_eq!(
            bare_counts,
            vec![1000],
            "only the bare array contributes a note count"
        );
        assert!(bare.is_array(), "bare stays a bare array");
        assert_eq!(
            nested["_sampled_from_rows"], 900,
            "nested array gets its sibling"
        );
    }

    #[test]
    fn diverse_fill_prefers_distinct_rows_over_near_duplicate_spam() {
        // The middle of the array (not first/last, no errors/rare values) is a block of
        // identical rows plus a handful of distinct ones. A pure highest-overlap fill would
        // keep interchangeable duplicates; the diverse (facility-location) fill must surface
        // the distinct rows so the sample spans the data.
        let mut a: Vec<Value> = Vec::new();
        // A long head/tail of identical filler so first/last pins land on duplicates.
        for _ in 0..120 {
            a.push(json!({"kind": "x", "msg": "routine heartbeat ping ok steady nominal"}));
        }
        // Five genuinely distinct rows buried in the middle.
        let distinct = [
            "disk volume remount latency spike detected",
            "auth token rotation completed for tenant",
            "cache warm reload finished across shards",
            "queue backlog drained after worker scale",
            "tls handshake renegotiated upstream peer",
        ];
        let pos: Vec<usize> = (0..distinct.len()).map(|k| 40 + k * 3).collect();
        for (k, &p) in pos.iter().enumerate() {
            a[p] = json!({"kind": "x", "msg": distinct[k]});
        }
        let arr = Value::Array(a);

        let (_, rows) = crush_array(&arr, 30, &HashSet::new()).expect("over-cap array is sampled");
        let msgs: HashSet<&str> = rows.iter().filter_map(|r| r["msg"].as_str()).collect();
        let distinct_kept = distinct.iter().filter(|d| msgs.contains(**d)).count();
        assert!(
            distinct_kept >= 3,
            "diverse fill surfaces the distinct rows (kept {distinct_kept}/5): {msgs:?}"
        );
        assert!(rows.len() <= 30, "within budget, got {}", rows.len());
    }

    #[test]
    fn query_bias_survives_diverse_fill() {
        // The diverse fill keeps the relevance (query-overlap) term: a row matching the
        // query must be sampled even though it isn't first/last or an outlier.
        let mut a: Vec<Value> = Vec::new();
        for i in 0..400 {
            a.push(json!({"kind": "x", "msg": format!("routine event number {i}")}));
        }
        // A single needle in the middle that matches the query's distinctive words.
        a[200] = json!({"kind": "x", "msg": "kubernetes pod eviction quota exceeded"});
        let arr = Value::Array(a);
        let query: HashSet<String> = lex_words("kubernetes pod eviction").into_iter().collect();

        let (_, rows) = crush_array(&arr, 30, &query).expect("over-cap array is sampled");
        let kept_needle = rows
            .iter()
            .any(|r| r["msg"].as_str() == Some("kubernetes pod eviction quota exceeded"));
        assert!(
            kept_needle,
            "the query-matching row is kept (relevance term)"
        );
    }

    #[test]
    fn row_value_text_uses_values_not_keys() {
        // Two rows with the SAME keys but different values must produce different feature
        // text (keys carry no row-distinguishing signal).
        let a = row_value_text(&json!({"city": "Paris", "code": 75}));
        let b = row_value_text(&json!({"city": "Tokyo", "code": 13}));
        assert!(
            a.contains("Paris") && a.contains("75"),
            "values present: {a:?}"
        );
        assert!(!a.contains("city"), "keys excluded: {a:?}");
        assert_ne!(a, b, "different values → different feature text");
    }
}
