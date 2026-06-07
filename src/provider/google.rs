//! Google Gemini (Generative Language API) adapter.
//!
//! Gemini's wire shape differs from OpenAI/Anthropic: messages live under `contents[]`
//! as `{role, parts[]}`, text is `parts[].text`, the system prompt is a top-level
//! `systemInstruction`, output controls live under `generationConfig`, tools are
//! `tools[].functionDeclarations[]`, and the model is in the URL path (not the body).

use serde_json::{Value, json};

use super::Provider;
use crate::ir::{ProviderKind, Request};

pub struct GoogleProvider;

impl Provider for GoogleProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Google
    }

    fn content_text_pointers(&self, req: &Request) -> Vec<String> {
        let mut out = Vec::new();
        let root = req.raw();
        // Top-level systemInstruction.parts[].text
        if let Some(parts) = root
            .pointer("/systemInstruction/parts")
            .and_then(Value::as_array)
        {
            for (j, p) in parts.iter().enumerate() {
                if p.get("text").is_some_and(Value::is_string) {
                    out.push(format!("/systemInstruction/parts/{j}/text"));
                }
            }
        }
        // contents[].parts[].text
        if let Some(contents) = root.get("contents").and_then(Value::as_array) {
            for (i, c) in contents.iter().enumerate() {
                let Some(parts) = c.get("parts").and_then(Value::as_array) else {
                    continue;
                };
                for (j, p) in parts.iter().enumerate() {
                    if p.get("text").is_some_and(Value::is_string) {
                        out.push(format!("/contents/{i}/parts/{j}/text"));
                    }
                }
            }
        }
        out
    }

    fn set_max_tokens(&self, req: &mut Request, max_tokens: u64) {
        gen_config_mut(req, |g| {
            g.insert("maxOutputTokens".to_string(), json!(max_tokens));
        });
    }

    fn max_tokens(&self, req: &Request) -> Option<u64> {
        req.raw()
            .pointer("/generationConfig/maxOutputTokens")
            .and_then(Value::as_u64)
    }

    fn add_stop_sequence(&self, req: &mut Request, stop: &str) {
        gen_config_mut(req, |g| match g.get_mut("stopSequences") {
            Some(Value::Array(arr)) => arr.push(json!(stop)),
            _ => {
                g.insert("stopSequences".to_string(), json!([stop]));
            }
        });
    }

    fn add_system_instruction(&self, req: &mut Request, text: &str) {
        let Some(obj) = req.raw_mut().as_object_mut() else {
            return;
        };
        // Prepend our text part, preserving any existing systemInstruction.
        match obj.get_mut("systemInstruction") {
            Some(si) => {
                if let Some(parts) = si.get_mut("parts").and_then(Value::as_array_mut) {
                    parts.insert(0, json!({"text": text}));
                } else {
                    *si = json!({"parts": [{"text": text}]});
                }
            }
            None => {
                obj.insert(
                    "systemInstruction".to_string(),
                    json!({"parts": [{"text": text}]}),
                );
            }
        }
    }

    fn bind_structured_output(&self, req: &mut Request, _name: &str, schema: Value) {
        // Gemini constrains output via generationConfig (no named schema object).
        gen_config_mut(req, |g| {
            g.insert("responseMimeType".to_string(), json!("application/json"));
            g.insert("responseSchema".to_string(), schema.clone());
        });
    }

    fn set_cache_breakpoints(&self, _req: &mut Request, _max: usize) {
        // Gemini caching is a separate `cachedContents` resource, not inline breakpoints.
    }

    fn tool_descriptors(&self, req: &Request) -> Vec<(String, String)> {
        let Some(tools) = req.raw().get("tools").and_then(Value::as_array) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for t in tools {
            let Some(decls) = t.get("functionDeclarations").and_then(Value::as_array) else {
                continue;
            };
            for d in decls {
                let name = d.get("name").and_then(Value::as_str).unwrap_or_default();
                let desc = d
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                out.push((name.to_string(), desc.to_string()));
            }
        }
        out
    }

    fn retain_tools(&self, req: &mut Request, keep: &[bool]) {
        let Some(tools) = req.raw_mut().get_mut("tools").and_then(Value::as_array_mut) else {
            return;
        };
        // `keep` is positional over the flattened functionDeclarations, in the same order
        // as `tool_descriptors`.
        let mut idx = 0usize;
        for t in tools.iter_mut() {
            if let Some(decls) = t
                .get_mut("functionDeclarations")
                .and_then(Value::as_array_mut)
            {
                decls.retain(|_| {
                    let k = keep.get(idx).copied().unwrap_or(true);
                    idx += 1;
                    k
                });
            }
        }
        // Drop tool entries whose declarations were all removed.
        tools.retain(|t| {
            t.get("functionDeclarations")
                .and_then(Value::as_array)
                .is_none_or(|d| !d.is_empty())
        });
    }

    fn truncate_tool_descriptions(&self, req: &mut Request, max_chars: usize) {
        let Some(tools) = req.raw_mut().get_mut("tools").and_then(Value::as_array_mut) else {
            return;
        };
        for t in tools.iter_mut() {
            let Some(decls) = t
                .get_mut("functionDeclarations")
                .and_then(Value::as_array_mut)
            else {
                continue;
            };
            for d in decls.iter_mut() {
                if let Some(Value::String(s)) = d.get_mut("description") {
                    super::truncate_chars(s, max_chars);
                }
            }
        }
    }

    fn answer_text(&self, response: &Value) -> Option<String> {
        let parts = response
            .pointer("/candidates/0/content/parts")?
            .as_array()?;
        let text: String = parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect();
        (!text.is_empty()).then_some(text)
    }

    fn set_image_detail(&self, _req: &mut Request, _tier: &str) {
        // Gemini has no per-image detail tier.
    }

    fn downscale_images(&self, req: &mut Request) {
        // Walk contents[].parts[].inline_data {mime_type, data}. Cap the long edge at
        // Gemini's effective image resolution (quality-neutral — Gemini downsamples to it
        // anyway).
        let Some(contents) = req
            .raw_mut()
            .get_mut("contents")
            .and_then(Value::as_array_mut)
        else {
            return;
        };
        for c in contents.iter_mut() {
            let Some(parts) = c.get_mut("parts").and_then(Value::as_array_mut) else {
                continue;
            };
            for p in parts.iter_mut() {
                if let Some(Value::String(data)) = p.pointer_mut("/inline_data/data")
                    && let Some(new_data) = crate::media::fit_to_cap(data, crate::media::CAP_GOOGLE)
                {
                    *data = new_data;
                }
            }
        }
    }
}

/// Get a mutable `generationConfig` object, creating it if absent, and apply `f`.
fn gen_config_mut(req: &mut Request, f: impl FnOnce(&mut serde_json::Map<String, Value>)) {
    let Some(obj) = req.raw_mut().as_object_mut() else {
        return;
    };
    let gc = obj.entry("generationConfig").or_insert_with(|| json!({}));
    if let Some(g) = gc.as_object_mut() {
        f(g);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(body: &str) -> Request {
        Request::parse(ProviderKind::Google, body).unwrap()
    }

    #[test]
    fn text_pointers_cover_system_and_contents() {
        let r = req(r#"{
            "systemInstruction": {"parts": [{"text": "sys"}]},
            "contents": [
                {"role": "user", "parts": [{"text": "hi"}, {"inline_data": {"mime_type": "image/png", "data": "x"}}]},
                {"role": "model", "parts": [{"text": "yo"}]}
            ]
        }"#);
        let p = GoogleProvider.content_text_pointers(&r);
        assert_eq!(
            p,
            vec![
                "/systemInstruction/parts/0/text",
                "/contents/0/parts/0/text",
                "/contents/1/parts/0/text",
            ]
        );
    }

    #[test]
    fn max_tokens_and_stop_use_generation_config() {
        let mut r = req(r#"{"contents":[]}"#);
        GoogleProvider.set_max_tokens(&mut r, 64);
        assert_eq!(GoogleProvider.max_tokens(&r), Some(64));
        GoogleProvider.add_stop_sequence(&mut r, "END");
        assert_eq!(
            r.raw().pointer("/generationConfig/stopSequences"),
            Some(&json!(["END"]))
        );
    }

    #[test]
    fn system_instruction_prepends_part() {
        let mut r = req(r#"{"systemInstruction":{"parts":[{"text":"old"}]},"contents":[]}"#);
        GoogleProvider.add_system_instruction(&mut r, "new");
        let parts = r
            .raw()
            .pointer("/systemInstruction/parts")
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(parts[0], json!({"text": "new"}));
        assert_eq!(parts[1], json!({"text": "old"}));
    }

    #[test]
    fn tool_descriptors_and_retain_flatten_function_declarations() {
        let mut r = req(r#"{
            "contents": [],
            "tools": [{"functionDeclarations": [
                {"name": "get_weather", "description": "weather"},
                {"name": "run_sql", "description": "sql"}
            ]}]
        }"#);
        let d = GoogleProvider.tool_descriptors(&r);
        assert_eq!(d.len(), 2);
        assert_eq!(d[1].0, "run_sql");
        GoogleProvider.retain_tools(&mut r, &[false, true]);
        let names: Vec<&str> = r
            .raw()
            .pointer("/tools/0/functionDeclarations")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|x| x.get("name").and_then(Value::as_str))
            .collect();
        assert_eq!(names, vec!["run_sql"]);
    }

    #[test]
    fn answer_text_concatenates_candidate_parts() {
        let resp = json!({
            "candidates": [{"content": {"role": "model", "parts": [{"text": "hello "}, {"text": "world"}]}}]
        });
        assert_eq!(
            GoogleProvider.answer_text(&resp),
            Some("hello world".to_string())
        );
    }
}
