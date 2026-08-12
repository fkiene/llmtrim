//! Claude Code custom agents for request-local subscription routing.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::reroute::{CODEX_MODELS, GROK_MODELS};

const OWNED_MARKER: &str = "<!-- llmtrim-owned-route-agent-v1 -->";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Status {
    pub installed: usize,
    pub current: usize,
    pub expected: usize,
}

struct Agent {
    name: String,
    description: String,
    route: String,
}

fn claude_dir() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("CLAUDE_CONFIG_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .context("neither HOME nor USERPROFILE is set")?;
    Ok(PathBuf::from(home).join(".claude"))
}

pub fn agents_dir() -> Result<PathBuf> {
    Ok(claude_dir()?.join("agents"))
}

fn machine_component(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

fn display_name(model: &str) -> &str {
    match model {
        "gpt-5.6-terra" => "Terra (GPT Terra)",
        "gpt-5.6-luna" => "Luna (GPT Luna)",
        "gpt-5.6-sol" => "Sol (GPT Sol)",
        "gpt-5.4-mini" => "Mini (GPT Mini)",
        "grok-4.6" => "Grok 4.6",
        "grok-composer-2.5-fast" => "Grok Composer",
        "kimi-for-coding" => "Kimi",
        other => other,
    }
}

fn definitions() -> Vec<Agent> {
    let mut agents = vec![
        Agent {
            name: "llmtrim-codex".into(),
            description: "Use when the user asks for a Codex, ChatGPT, or OpenAI subagent. The provider model follows the current Claude model tier.".into(),
            route: "codex".into(),
        },
        Agent {
            name: "llmtrim-grok".into(),
            description: "Use when the user asks to delegate work to Grok or a Grok subagent. The provider model follows the current Claude model tier.".into(),
            route: "grok".into(),
        },
        Agent {
            name: "llmtrim-kimi".into(),
            description: "Use when the user asks to delegate work to Kimi, Moonshot, or a Kimi subagent.".into(),
            route: "kimi".into(),
        },
    ];
    for model in CODEX_MODELS {
        agents.push(Agent {
            name: format!("llmtrim-codex-{}", machine_component(model)),
            description: format!(
                "Use when the user explicitly asks for {}, {model}, or that Codex model as a subagent.",
                display_name(model)
            ),
            route: format!("codex/{model}"),
        });
    }
    for model in GROK_MODELS {
        agents.push(Agent {
            name: format!("llmtrim-grok-{}", machine_component(model)),
            description: format!(
                "Use when the user explicitly asks for {}, {model}, or that Grok model as a subagent.",
                display_name(model)
            ),
            route: format!("grok/{model}"),
        });
    }
    // The provider-only Kimi agent already resolves to its sole concrete model.
    agents
}

fn render(agent: &Agent) -> String {
    format!(
        "---\nname: {}\ndescription: {}\nmodel: inherit\n---\n\n{}\n<!-- llmtrim-route-v1:{} -->\n\nComplete the delegated task in this thread and return the result to the parent agent.\n",
        agent.name, agent.description, OWNED_MARKER, agent.route
    )
}

fn is_owned(path: &Path) -> bool {
    fs::read_to_string(path)
        .map(|text| text.contains(OWNED_MARKER))
        .unwrap_or(false)
}

fn write_atomic(path: &Path, content: &str) -> Result<()> {
    let tmp = path.with_extension("md.tmp");
    if tmp.exists() && fs::symlink_metadata(&tmp)?.file_type().is_symlink() {
        bail!(
            "refusing symlinked route-agent temporary file: {}",
            tmp.display()
        );
    }
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp)?;
    file.write_all(content.as_bytes())?;
    file.sync_all()?;
    #[cfg(windows)]
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(tmp, path)?;
    Ok(())
}

pub fn install() -> Result<Status> {
    install_at(&agents_dir()?)
}

fn install_at(dir: &Path) -> Result<Status> {
    fs::create_dir_all(dir)?;
    let definitions = definitions();

    // Check every collision before changing anything.
    for agent in &definitions {
        let path = dir.join(format!("{}.md", agent.name));
        if path.exists() && !is_owned(&path) {
            bail!(
                "refusing to overwrite non-llmtrim Claude agent: {}",
                path.display()
            );
        }
    }

    for agent in &definitions {
        let path = dir.join(format!("{}.md", agent.name));
        write_atomic(&path, &render(agent))?;
    }

    // Remove stale files only when both their name and ownership marker identify this integration.
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if name.starts_with("llmtrim-")
            && name.ends_with(".md")
            && is_owned(&path)
            && !definitions
                .iter()
                .any(|agent| name == format!("{}.md", agent.name))
        {
            fs::remove_file(path)?;
        }
    }

    status_at(dir)
}

pub fn uninstall() -> Result<usize> {
    uninstall_at(&agents_dir()?)
}

fn uninstall_at(dir: &Path) -> Result<usize> {
    if !dir.exists() {
        return Ok(0);
    }
    let mut removed = 0;
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        if name.starts_with("llmtrim-") && name.ends_with(".md") && is_owned(&path) {
            fs::remove_file(path)?;
            removed += 1;
        }
    }
    Ok(removed)
}

pub fn status() -> Result<Status> {
    status_at(&agents_dir()?)
}

fn status_at(dir: &Path) -> Result<Status> {
    let definitions = definitions();
    let installed = definitions
        .iter()
        .filter(|agent| {
            let path = dir.join(format!("{}.md", agent.name));
            path.is_file() && is_owned(&path)
        })
        .count();
    let current = definitions
        .iter()
        .filter(|agent| {
            let path = dir.join(format!("{}.md", agent.name));
            fs::read_to_string(path).is_ok_and(|text| text == render(agent))
        })
        .count();
    Ok(Status {
        installed,
        current,
        expected: definitions.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    struct TempAgents(PathBuf);

    impl TempAgents {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                "llmtrim-route-agents-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            )))
        }
    }

    impl Drop for TempAgents {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn install_is_idempotent_and_uninstall_is_owned_only() {
        let temp = TempAgents::new();
        let first = install_at(&temp.0).unwrap();
        assert_eq!(first.installed, first.expected);
        let second = install_at(&temp.0).unwrap();
        assert_eq!(second, first);
        let user = temp.0.join("user.md");
        fs::write(&user, "---\nname: user\n---\n").unwrap();
        assert_eq!(uninstall_at(&temp.0).unwrap(), first.expected);
        assert!(user.exists());
    }

    #[test]
    fn install_refuses_non_owned_collision_before_writing() {
        let temp = TempAgents::new();
        fs::create_dir_all(&temp.0).unwrap();
        fs::write(temp.0.join("llmtrim-codex.md"), "user content").unwrap();
        assert!(install_at(&temp.0).is_err());
        assert!(!temp.0.join("llmtrim-grok.md").exists());
    }

    #[test]
    fn generated_terra_agent_uses_canonical_marker_and_alias_description() {
        let terra = definitions()
            .into_iter()
            .find(|agent| agent.route == "codex/gpt-5.6-terra")
            .unwrap();
        let text = render(&terra);
        assert!(text.contains("<!-- llmtrim-route-v1:codex/gpt-5.6-terra -->"));
        assert!(text.contains("Terra (GPT Terra)"));
        assert!(text.contains("model: inherit"));
    }

    #[test]
    fn sole_kimi_model_is_not_duplicated() {
        let agents = definitions();
        assert_eq!(
            agents.iter().filter(|agent| agent.route == "kimi").count(),
            1
        );
        assert!(
            !agents
                .iter()
                .any(|agent| agent.route == format!("kimi/{}", crate::reroute::KIMI_MODEL))
        );
    }
}
