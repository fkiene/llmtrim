//! Protocol-level test: drive the MCP stdio handler through a real JSON-RPC exchange
//! (`initialize` → `tools/list` → `tools/call`) over an in-memory duplex pipe, the same
//! way an MCP client would over stdio, and assert the responses are well-formed and the
//! tools run.
#![cfg(feature = "mcp")]

use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use serde_json::json;

#[tokio::test]
async fn initialize_list_and_call_over_jsonrpc() {
    // Isolate the ledger so the recording tools never write to the user's real savings DB.
    let db = std::env::temp_dir().join(format!("llmtrim_mcp_proto_{}.db", std::process::id()));

    let (server_transport, client_transport) = tokio::io::duplex(8192);

    // The server side: spawn the real handler the `llmtrim mcp` command serves. We start
    // it through the same public entry the binary uses, over the duplex instead of stdio.
    let server = tokio::spawn(async move {
        let service = llmtrim::mcp::test_server(db)
            .serve(server_transport)
            .await
            .expect("server handshake");
        service.waiting().await.expect("server run");
    });

    // The client side: `()` is a no-op ClientHandler. `serve` performs `initialize` for us.
    let client = ().serve(client_transport).await.expect("client initialize");

    // tools/list advertises exactly the five documented tools, each with an input schema.
    let tools = client.list_all_tools().await.expect("tools/list");
    let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "llmtrim_compress",
            "llmtrim_compress_text",
            "llmtrim_read_file_compressed",
            "llmtrim_read_folder_compressed",
            "llmtrim_stats",
        ]
    );
    assert!(
        tools.iter().all(|t| !t.input_schema.is_empty()),
        "every tool advertises an input schema"
    );
    // The documented input fields show up in the advertised schemas.
    let schema = |name: &str| {
        let t = tools.iter().find(|t| t.name == name).unwrap();
        t.input_schema["properties"].clone()
    };
    assert!(schema("llmtrim_compress")["request"].is_object());
    assert!(schema("llmtrim_compress")["provider"].is_object());
    assert!(schema("llmtrim_compress_text")["text"].is_object());
    assert!(schema("llmtrim_read_file_compressed")["path"].is_object());
    assert!(schema("llmtrim_read_folder_compressed")["path"].is_object());

    // Helper: call a tool over JSON-RPC and parse its single text result as JSON.
    async fn call(
        client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
        name: &'static str,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> serde_json::Value {
        let mut params = CallToolRequestParams::new(name);
        params.arguments = Some(args);
        let result = client.call_tool(params).await.expect("tools/call");
        assert_ne!(result.is_error, Some(true), "{name} must not error");
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map(|t| t.text.clone())
            .expect("text content");
        serde_json::from_str(&text).expect("tool result is JSON")
    }

    // llmtrim_compress: a real-shaped request comes back compressed with token deltas. We
    // don't assert the input shrinks (the server honors ~/.llmtrim config, whose `auto`
    // default may add an output-shaping instruction); the deterministic mapping is unit-tested.
    let request_body = json!({
        "model": "gpt-4o",
        "messages": [
            { "role": "system", "content": "You are a helpful assistant.    " },
            { "role": "user", "content": "Hello    world\n\n\nwith   redundant    whitespace." }
        ]
    })
    .to_string();
    let mut args = serde_json::Map::new();
    args.insert("request".into(), json!(request_body));
    let payload = call(&client, "llmtrim_compress", args).await;
    assert!(payload["request_json"].is_string());
    assert!(payload["input_tokens_before"].as_u64().unwrap() > 0);
    assert!(payload["stages"].as_array().is_some_and(|s| !s.is_empty()));
    assert_eq!(payload["provider"], "openai");

    // llmtrim_compress_text: a blob comes back as shrunk text with blob-level deltas.
    let mut args = serde_json::Map::new();
    args.insert(
        "text".into(),
        json!("repeat me\nrepeat me\ntail words here"),
    );
    args.insert("client".into(), json!("Devin"));
    args.insert("model".into(), json!("Kimi K2.6"));
    let payload = call(&client, "llmtrim_compress_text", args).await;
    assert!(payload["text"].is_string());
    assert!(payload["input_tokens_before"].as_u64().unwrap() > 0);
    assert!(payload["tokens_saved"].as_i64().is_some());

    // llmtrim_stats: returns the ledger snapshot as well-formed JSON.
    let payload = call(&client, "llmtrim_stats", serde_json::Map::new()).await;
    assert!(payload["requests"].as_u64().is_some());
    assert!(payload["by_model"].is_array());
    // MCP label should appear in by_model when client/model were provided
    let by_model = payload["by_model"].as_array().unwrap();
    assert!(
        by_model.iter().any(|m| m["model"]
            .as_str()
            .unwrap()
            .contains("mcp · Devin · Kimi K2.6")),
        "by_model should contain MCP label"
    );

    // llmtrim_read_file_compressed: a temp text file comes back compressed.
    let tmp_dir = std::env::temp_dir().join(format!("llmtrim_mcp_file_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let tmp_file = tmp_dir.join("sample.txt");
    std::fs::write(&tmp_file, "Hello    world\nwith   extra   spaces.\n").unwrap();

    let mut args = serde_json::Map::new();
    args.insert("path".into(), json!(tmp_file.to_str().unwrap()));
    let payload = call(&client, "llmtrim_read_file_compressed", args).await;
    assert!(payload["text"].is_string());
    assert!(payload["input_tokens_before"].as_u64().unwrap() > 0);
    assert!(payload["tokens_saved"].as_i64().is_some());

    // llmtrim_read_file_compressed: rejects a secret file.
    let secret_file = tmp_dir.join(".env");
    std::fs::write(&secret_file, "SECRET=123").unwrap();
    let mut args = serde_json::Map::new();
    args.insert("path".into(), json!(secret_file.to_str().unwrap()));
    let mut params = CallToolRequestParams::new("llmtrim_read_file_compressed");
    params.arguments = Some(args);
    let result = client.call_tool(params).await;
    assert!(result.is_err(), "secret file must error");

    let _ = std::fs::remove_dir_all(&tmp_dir);

    // llmtrim_read_folder_compressed: a temp folder with source files comes back compressed.
    let tmp_folder =
        std::env::temp_dir().join(format!("llmtrim_mcp_folder_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_folder).unwrap();
    std::fs::write(
        tmp_folder.join("main.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();
    std::fs::write(
        tmp_folder.join("util.rs"),
        "fn helper() {\n    let x = 42;\n}\n",
    )
    .unwrap();
    // Create an excluded subdirectory
    let excluded_dir = tmp_folder.join("node_modules");
    std::fs::create_dir_all(&excluded_dir).unwrap();
    std::fs::write(excluded_dir.join("bad.ts"), "export const bad = 1;\n").unwrap();

    let mut args = serde_json::Map::new();
    args.insert("path".into(), json!(tmp_folder.to_str().unwrap()));
    let payload = call(&client, "llmtrim_read_folder_compressed", args).await;
    assert_eq!(
        payload["folder_path"].as_str().unwrap(),
        tmp_folder.to_str().unwrap()
    );
    assert_eq!(payload["files_included"].as_u64().unwrap(), 2);
    // summary must exist and appear before files in serialized order
    let keys: Vec<&str> = payload
        .as_object()
        .unwrap()
        .keys()
        .map(|k| k.as_str())
        .collect();
    let summary_idx = keys
        .iter()
        .position(|&k| k == "summary")
        .expect("summary key missing");
    let files_idx = keys
        .iter()
        .position(|&k| k == "files")
        .expect("files key missing");
    assert!(
        summary_idx < files_idx,
        "summary must appear before files in JSON output"
    );
    // Verify structured summary shape
    let summary = &payload["summary"];
    assert_eq!(
        summary["folder"].as_str().unwrap(),
        tmp_folder.to_str().unwrap()
    );
    assert_eq!(summary["included_count"].as_u64().unwrap(), 2);
    assert!(summary["skipped_count"].as_u64().unwrap() >= 1);
    assert!(summary["total_input_tokens_before"].as_u64().unwrap() > 0);
    assert!(summary["total_tokens_saved"].as_i64().is_some());
    assert!(summary["saved_pct"].as_f64().is_some());
    assert_eq!(
        summary["budgets"]["max_total_input_tokens"]
            .as_u64()
            .unwrap(),
        1_000_000
    );
    assert_eq!(
        summary["budgets"]["max_total_output_tokens"]
            .as_u64()
            .unwrap(),
        100_000
    );
    let top_savings = summary["top_savings"].as_array().unwrap();
    assert!(!top_savings.is_empty());
    assert!(
        top_savings
            .iter()
            .any(|s| s["path"].as_str().unwrap().contains("main.rs")),
        "top_savings should contain main.rs"
    );
    assert!(top_savings[0]["input_tokens_before"].as_u64().unwrap() > 0);
    assert!(top_savings[0]["tokens_saved"].as_i64().is_some());
    assert!(top_savings[0]["saved_pct"].as_f64().is_some());
    // top_savings must be sorted by tokens_saved descending
    for window in top_savings.windows(2) {
        let a = window[0]["tokens_saved"].as_i64().unwrap();
        let b = window[1]["tokens_saved"].as_i64().unwrap();
        assert!(
            a >= b,
            "top_savings must be sorted by tokens_saved descending"
        );
    }
    let top_skipped = summary["top_skipped"].as_array().unwrap();
    assert!(
        top_skipped.iter().any(|s| {
            s["path"].as_str().unwrap().contains("node_modules")
                && s["reason"].as_str().unwrap() == "excluded"
        }),
        "top_skipped should contain excluded node_modules"
    );
    let files = payload["files"].as_array().unwrap();
    assert_eq!(files.len(), 2);
    // main.rs should be first because of higher importance score
    assert!(files[0]["path"].as_str().unwrap().contains("main.rs"));
    assert!(files[0]["text"].as_str().unwrap().contains("{ /* … */ }"));
    assert!(files[1]["path"].as_str().unwrap().contains("util.rs"));
    assert!(payload["total_input_tokens_before"].as_u64().unwrap() > 0);
    assert!(payload["total_tokens_saved"].as_i64().is_some());
    // Budget fields should be present with new defaults
    assert_eq!(
        payload["max_total_input_tokens"].as_u64().unwrap(),
        1_000_000
    );
    assert_eq!(
        payload["max_total_output_tokens"].as_u64().unwrap(),
        100_000
    );
    // The excluded node_modules directory should appear in skipped with reason "excluded"
    let skipped = payload["skipped"].as_array().unwrap();
    assert!(
        skipped.iter().any(|s| {
            s["path"].as_str().unwrap().contains("node_modules")
                && s["reason"].as_str().unwrap() == "excluded"
        }),
        "excluded directory should be in skipped list with reason 'excluded'"
    );
    let included_paths: Vec<&str> = files.iter().map(|f| f["path"].as_str().unwrap()).collect();
    assert!(
        !included_paths.iter().any(|p| p.contains("node_modules")),
        "node_modules should be excluded"
    );

    // llmtrim_read_folder_compressed: respects max_files limit.
    let mut args = serde_json::Map::new();
    args.insert("path".into(), json!(tmp_folder.to_str().unwrap()));
    args.insert("max_files".into(), json!(1));
    let payload = call(&client, "llmtrim_read_folder_compressed", args).await;
    assert_eq!(payload["files_included"].as_u64().unwrap(), 1);
    assert!(payload["files_skipped"].as_u64().unwrap() >= 1);
    let skipped = payload["skipped"].as_array().unwrap();
    assert!(
        skipped
            .iter()
            .any(|s| s["reason"].as_str().unwrap() == "max_files"),
        "skipped should contain max_files reason"
    );

    // llmtrim_read_folder_compressed: rejects a folder containing a secret file (secret is skipped, not errored).
    let secret_dir = tmp_folder.join("secrets");
    std::fs::create_dir_all(&secret_dir).unwrap();
    std::fs::write(secret_dir.join(".env"), "SECRET=123").unwrap();
    let mut args = serde_json::Map::new();
    args.insert("path".into(), json!(tmp_folder.to_str().unwrap()));
    args.insert("exclude_patterns".into(), json!([]));
    let payload = call(&client, "llmtrim_read_folder_compressed", args).await;
    // The secret file should appear in skipped with reason "secret", not cause an error
    let skipped = payload["skipped"].as_array().unwrap();
    assert!(
        skipped.iter().any(|s| {
            s["path"].as_str().unwrap().contains(".env")
                && s["reason"].as_str().unwrap() == "secret"
        }),
        "secret file should be in skipped list with reason 'secret'"
    );

    let _ = std::fs::remove_dir_all(&tmp_folder);

    client.cancel().await.ok();
    server.abort();
}
