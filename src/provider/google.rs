//! Google Gemini (Generative Language API) adapter.
//!
//! Gemini's wire shape differs from OpenAI/Anthropic: messages live under `contents[]`
//! as `{role, parts[]}`, text is `parts[].text`, the system prompt is a top-level
//! `systemInstruction`, output controls live under `generationConfig`, tools are
//! `tools[].functionDeclarations[]`, and the model is in the URL path (not the body).

use serde_json::{Value, json};

use super::{Provider, Role, string_leaf_pointers, turn_index};
use crate::ir::{ProviderKind, Request};

pub struct GoogleProvider;

/// The system-instruction key actually present (Gemini accepts both proto3 JSON
/// casings; SDKs emit `systemInstruction`, hand-rolled clients sometimes snake_case).
fn system_key(root: &Value) -> Option<&'static str> {
    if root.get("systemInstruction").is_some() {
        Some("systemInstruction")
    } else if root.get("system_instruction").is_some() {
        Some("system_instruction")
    } else {
        None
    }
}

impl Provider for GoogleProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Google
    }

    fn content_text_pointers(&self, req: &Request) -> Vec<String> {
        let mut out = Vec::new();
        let root = req.raw();
        // Top-level system instruction parts[].text (either casing).
        if let Some(sk) = system_key(root)
            && let Some(parts) = root
                .pointer(&format!("/{sk}/parts"))
                .and_then(Value::as_array)
        {
            for (j, p) in parts.iter().enumerate() {
                if p.get("text").is_some_and(Value::is_string) {
                    out.push(format!("/{sk}/parts/{j}/text"));
                }
            }
        }
        // contents[].parts[]: text, plus tool-channel payloads (functionResponse from the
        // tool, functionCall replayed by the model) whose text lives in string leaves.
        if let Some(contents) = root.get("contents").and_then(Value::as_array) {
            for (i, c) in contents.iter().enumerate() {
                let Some(parts) = c.get("parts").and_then(Value::as_array) else {
                    continue;
                };
                for (j, p) in parts.iter().enumerate() {
                    let prefix = format!("/contents/{i}/parts/{j}");
                    if p.get("text").is_some_and(Value::is_string) {
                        out.push(format!("{prefix}/text"));
                    }
                    if let Some(resp) = p.pointer("/functionResponse/response") {
                        string_leaf_pointers(
                            resp,
                            &format!("{prefix}/functionResponse/response"),
                            &mut out,
                        );
                    }
                    if let Some(args) = p.pointer("/functionCall/args") {
                        string_leaf_pointers(
                            args,
                            &format!("{prefix}/functionCall/args"),
                            &mut out,
                        );
                    }
                }
            }
        }
        out
    }

    fn role_at(&self, req: &Request, pointer: &str) -> Option<Role> {
        let i = turn_index(pointer)?;
        let role = req
            .raw()
            .pointer(&format!("/contents/{i}/role"))
            .and_then(Value::as_str)?;
        Some(Role::from_str(role))
    }

    fn set_max_tokens(&self, req: &mut Request, max_tokens: u64) {
        gen_config_mut(req, |g| {
            g.insert("maxOutputTokens".to_string(), json!(max_tokens));
        });
    }

    fn max_tokens(&self, req: &Request) -> Option<u64> {
        let root = req.raw();
        root.pointer("/generationConfig/maxOutputTokens")
            .or_else(|| root.pointer("/generation_config/maxOutputTokens"))
            .or_else(|| root.pointer("/generation_config/max_output_tokens"))
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
        // Reuse whichever casing the client sent; default to the SDK camelCase.
        let key = system_key(req.raw()).unwrap_or("systemInstruction");
        let Some(obj) = req.raw_mut().as_object_mut() else {
            return;
        };
        // Append our part after the existing system instruction (don't prepend): a volatile
        // prepend in front of an otherwise-stable systemInstruction defeats Gemini's implicit
        // prefix caching. Appended, the stable prefix is preserved.
        match obj.get_mut(key) {
            Some(si) => {
                if let Some(parts) = si.get_mut("parts").and_then(Value::as_array_mut) {
                    parts.push(json!({"text": text}));
                } else {
                    *si = json!({"parts": [{"text": text}]});
                }
            }
            None => {
                obj.insert(key.to_string(), json!({"parts": [{"text": text}]}));
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

    fn set_prompt_cache_key(&self, _req: &mut Request, _key: &str) {
        // No inline prompt cache key on Gemini (caching is the `cachedContents` resource).
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
                // Gemini accepts both proto3 JSON casings; SDKs emit camelCase `inlineData`.
                let ptr = if p.pointer("/inlineData/data").is_some() {
                    "/inlineData/data"
                } else {
                    "/inline_data/data"
                };
                if let Some(Value::String(data)) = p.pointer_mut(ptr)
                    && let Some(new_data) = crate::media::fit_to_cap(data, crate::media::CAP_GOOGLE)
                {
                    *data = new_data;
                }
            }
        }
    }
}

/// Get a mutable generation-config object, reusing whichever casing the client sent
/// (creating camelCase if absent — inserting a second `generationConfig` beside an
/// existing `generation_config` is a duplicate proto field → Gemini 400), and apply `f`.
fn gen_config_mut(req: &mut Request, f: impl FnOnce(&mut serde_json::Map<String, Value>)) {
    let Some(obj) = req.raw_mut().as_object_mut() else {
        return;
    };
    let key = if obj.contains_key("generation_config") {
        "generation_config"
    } else {
        "generationConfig"
    };
    let gc = obj.entry(key).or_insert_with(|| json!({}));
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
    fn system_instruction_appends_part() {
        // Appended after the existing system instruction so the stable prefix is preserved
        // (Gemini implicit prefix caching).
        let mut r = req(r#"{"systemInstruction":{"parts":[{"text":"old"}]},"contents":[]}"#);
        GoogleProvider.add_system_instruction(&mut r, "new");
        let parts = r
            .raw()
            .pointer("/systemInstruction/parts")
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(parts[0], json!({"text": "old"}));
        assert_eq!(parts[1], json!({"text": "new"}));
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
    fn function_response_payload_is_covered() {
        // Gemini tool output: text lives in functionResponse.response string leaves.
        let r = req(r#"{"contents":[
            {"role":"user","parts":[{"functionResponse":{"name":"read","response":{"result":"BIG FILE BODY"}}}]}
        ]}"#);
        let p = GoogleProvider.content_text_pointers(&r);
        assert!(
            p.contains(&"/contents/0/parts/0/functionResponse/response/result".to_string()),
            "{p:?}"
        );
        assert_eq!(
            GoogleProvider.role_at(&r, "/contents/0/parts/0/functionResponse/response/result"),
            Some(Role::User)
        );
    }

    #[test]
    fn snake_case_system_instruction_is_covered_and_downscale_handles_camel() {
        // Hand-rolled clients send snake_case; SDKs send camelCase inlineData.
        let r = req(r#"{"system_instruction":{"parts":[{"text":"sys"}]},"contents":[]}"#);
        assert_eq!(
            GoogleProvider.content_text_pointers(&r),
            vec!["/system_instruction/parts/0/text"]
        );
        // gen_config reuses snake_case key instead of inserting a duplicate proto field.
        let mut r2 = req(r#"{"generation_config":{"temperature":0},"contents":[]}"#);
        GoogleProvider.set_max_tokens(&mut r2, 8);
        assert!(
            r2.raw().get("generationConfig").is_none(),
            "no duplicate config"
        );
        assert_eq!(GoogleProvider.max_tokens(&r2), Some(8));
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
