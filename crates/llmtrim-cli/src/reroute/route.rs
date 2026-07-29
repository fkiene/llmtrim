//! Trusted request-local routes carried by generated Claude Code agents.
//!
//! The marker is deliberately accepted only as a standalone line in the top-level Anthropic
//! system prompt; user messages and ordinary system prose are never interpreted as routes.

use anyhow::{Result, bail};
use serde_json::Value;

use super::{SubProvider, resolve_model};

/// Exact text of a generated-agent route marker.
pub const MARKER_PREFIX: &str = "<!-- llmtrim-route-v1:";
pub const MARKER_SUFFIX: &str = " -->";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestRoute {
    pub provider: SubProvider,
    /// `Some` means the custom agent selected a concrete upstream model.
    pub model: Option<String>,
}

impl RequestRoute {
    pub fn resolve(
        &self,
        incoming: &str,
        tiers: &std::collections::BTreeMap<String, String>,
    ) -> String {
        self.model
            .clone()
            .unwrap_or_else(|| resolve_model(self.provider, incoming, tiers))
    }
}

/// Remove and return a single trusted route marker line from the top-level Anthropic system
/// prompt. User, assistant, and tool content is never inspected. A marker-shaped line which is not
/// valid is a client error rather than prompt text, making accidental near-markers visible.
pub fn take_marker(body: &mut Value) -> Result<Option<RequestRoute>> {
    let Some(system) = body.get_mut("system") else {
        return Ok(None);
    };
    let mut found = None;
    match system {
        Value::String(text) => take_marker_line(text, &mut found)?,
        Value::Array(blocks) => {
            for block in blocks.iter_mut() {
                let Some(text) = block.get("text").and_then(Value::as_str) else {
                    continue;
                };
                if !text
                    .lines()
                    .any(|line| line.trim().starts_with(MARKER_PREFIX))
                {
                    continue;
                }
                if block.get("type").and_then(Value::as_str) != Some("text") {
                    bail!("llmtrim: request-local route marker must be in a system text block");
                }
                let Some(Value::String(text)) = block.get_mut("text") else {
                    unreachable!("text was checked above");
                };
                take_marker_line(text, &mut found)?;
            }
            blocks.retain(|block| {
                block.get("type").and_then(Value::as_str) != Some("text")
                    || block.get("text").and_then(Value::as_str) != Some("")
            });
        }
        _ => return Ok(None),
    }
    Ok(found)
}

fn take_marker_line(text: &mut String, found: &mut Option<RequestRoute>) -> Result<()> {
    let mut kept = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(MARKER_PREFIX) {
            let route = parse_marker(trimmed)?;
            if found.replace(route).is_some() {
                bail!("llmtrim: duplicate request-local route marker");
            }
        } else {
            kept.push(line);
        }
    }
    if kept.len() != text.lines().count() {
        *text = kept.join("\n");
    }
    Ok(())
}

fn parse_marker(text: &str) -> Result<RequestRoute> {
    let route = text
        .strip_prefix(MARKER_PREFIX)
        .and_then(|s| s.strip_suffix(MARKER_SUFFIX))
        .ok_or_else(|| anyhow::anyhow!("llmtrim: malformed request-local route marker"))?;
    if route.is_empty() || route.contains(char::is_whitespace) || route.matches('/').count() > 1 {
        bail!("llmtrim: malformed request-local route marker");
    }
    let mut pieces = route.split('/');
    let provider = SubProvider::parse(pieces.next().unwrap())
        .filter(|p| p.as_str() == route.split('/').next().unwrap())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "llmtrim: route marker provider must be canonical (codex, kimi, or grok)"
            )
        })?;
    let model = pieces
        .next()
        .map(|m| resolve_explicit_model(provider, m))
        .transpose()?;
    Ok(RequestRoute { provider, model })
}

/// Resolve curated provider-scoped aliases used by the agent installer. Unknown explicit ids never
/// use the ordinary tier fallback.
pub fn resolve_explicit_model(provider: SubProvider, value: &str) -> Result<String> {
    let normalized: String = value
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | ' '))
        .flat_map(char::to_lowercase)
        .collect();
    let model = match provider {
        SubProvider::Codex => match normalized.as_str() {
            "terra" | "gptterra" | "gpt5terra" | "gpt56terra" => "gpt-5.6-terra",
            "luna" | "gptluna" | "gpt5luna" | "gpt56luna" => "gpt-5.6-luna",
            "sol" | "gptsol" | "gpt5sol" | "gpt56sol" => "gpt-5.6-sol",
            "mini" | "gptmini" | "gpt54mini" => "gpt-5.4-mini",
            _ => value,
        },
        SubProvider::Grok => match normalized.as_str() {
            "grok" | "grok45" => "grok-4.5",
            "composer" | "grokcomposer" | "grokcomposer25fast" => "grok-composer-2.5-fast",
            _ => value,
        },
        SubProvider::Kimi => match normalized.as_str() {
            "kimi" | "k2" | "kimik2" | "kimiforcoding" => super::KIMI_MODEL,
            _ => value,
        },
    };
    let accepted = match provider {
        SubProvider::Codex => super::CODEX_MODELS.contains(&model),
        SubProvider::Grok => super::GROK_MODELS.contains(&model),
        SubProvider::Kimi => model == super::KIMI_MODEL,
    };
    if accepted {
        return Ok(model.to_ascii_lowercase());
    }
    let supported = match provider {
        SubProvider::Codex => super::CODEX_MODELS.join(", "),
        SubProvider::Grok => super::GROK_MODELS.join(", "),
        SubProvider::Kimi => super::KIMI_MODEL.to_string(),
    };
    bail!(
        "llmtrim: unknown {provider} route model `{value}`; supported models: {supported}",
        provider = provider.as_str()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn takes_only_standalone_top_level_system_block() {
        let mut body = json!({"system":[{"type":"text","text":"<!-- llmtrim-route-v1:codex/gpt-5.6-terra -->"},{"type":"text","text":"keep"}],"messages":[{"content":"<!-- llmtrim-route-v1:grok -->"} ]});
        assert_eq!(
            take_marker(&mut body).unwrap().unwrap().model.as_deref(),
            Some("gpt-5.6-terra")
        );
        assert_eq!(body["system"][0]["text"], "keep");
        assert_eq!(
            body["messages"][0]["content"],
            "<!-- llmtrim-route-v1:grok -->"
        );
    }
    #[test]
    fn rejects_malformed_and_duplicate_markers() {
        let mut malformed = json!({"system":"<!-- llmtrim-route-v1:chatgpt -->"});
        assert!(take_marker(&mut malformed).is_err());
        let mut duplicate = json!({"system":[{"text":"<!-- llmtrim-route-v1:codex -->"},{"text":"<!-- llmtrim-route-v1:grok -->"}]});
        assert!(take_marker(&mut duplicate).is_err());
    }
    #[test]
    fn extracts_marker_line_from_generated_agent_prompt() {
        let mut body = json!({
            "system": [{
                "type": "text",
                "text": "Agent instructions\n<!-- llmtrim-route-v1:grok -->\nComplete the task."
            }]
        });
        let route = take_marker(&mut body).unwrap().unwrap();
        assert_eq!(route.provider, SubProvider::Grok);
        assert_eq!(
            body["system"][0]["text"],
            "Agent instructions\nComplete the task."
        );
    }

    #[test]
    fn rejects_marker_in_non_text_system_block() {
        let mut body = json!({
            "system": [{
                "type": "tool_use",
                "text": "<!-- llmtrim-route-v1:grok -->"
            }]
        });
        assert!(take_marker(&mut body).is_err());
    }

    #[test]
    fn provider_only_route_preserves_claude_tier_and_override() {
        let route = RequestRoute {
            provider: SubProvider::Codex,
            model: None,
        };
        assert_eq!(
            route.resolve("claude-fable-5", &Default::default()),
            "gpt-5.6-sol"
        );
        let tiers =
            std::collections::BTreeMap::from([("fable".to_string(), "gpt-5.6-terra".to_string())]);
        assert_eq!(route.resolve("claude-fable-5", &tiers), "gpt-5.6-terra");
    }

    #[test]
    fn resolves_terra_aliases_without_fallback() {
        for name in [
            "terra",
            "gpt-terra",
            "gpt5-terra",
            "gpt-5.6-terra",
            "gpt terra",
        ] {
            assert_eq!(
                resolve_explicit_model(SubProvider::Codex, name).unwrap(),
                "gpt-5.6-terra"
            );
        }
        assert!(resolve_explicit_model(SubProvider::Codex, "unknown").is_err());
    }
}
