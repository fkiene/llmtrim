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
    // Isolate the config too: a fixed built-in preset, so the test never reads (or fails on)
    // the developer's `~/.llmtrim` config file. CI has no such file; a malformed local one
    // used to fail this test while CI stayed green.
    let config = llmtrim_core::config::DenseConfig::preset("auto").expect("built-in preset");

    let server = tokio::spawn(async move {
        let service = llmtrim::mcp::test_server(db, config)
            .serve(server_transport)
            .await
            .expect("server handshake");
        service.waiting().await.expect("server run");
    });

    // The client side: `()` is a no-op ClientHandler. `serve` performs `initialize` for us.
    let client = ().serve(client_transport).await.expect("client initialize");

    // tools/list advertises exactly the three documented tools, each with an input schema.
    let tools = client.list_all_tools().await.expect("tools/list");
    let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        ["llmtrim_compress", "llmtrim_compress_text", "llmtrim_stats"]
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
    // don't assert the input shrinks (the `auto` config may add an output-shaping
    // instruction); the deterministic mapping is unit-tested.
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
    let payload = call(&client, "llmtrim_compress_text", args).await;
    assert!(payload["text"].is_string());
    assert!(payload["input_tokens_before"].as_u64().unwrap() > 0);
    assert!(payload["tokens_saved"].as_i64().is_some());

    // llmtrim_stats: returns the ledger snapshot as well-formed JSON.
    let payload = call(&client, "llmtrim_stats", serde_json::Map::new()).await;
    assert!(payload["requests"].as_u64().is_some());
    assert!(payload["by_model"].is_array());

    client.cancel().await.ok();
    server.abort();
}
