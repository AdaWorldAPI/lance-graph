//! OpenAI-compatible REST server powered by lance-graph-planner.
//!
//! ```bash
//! cargo run --manifest-path crates/lance-graph-planner/Cargo.toml \
//!   --features serve --bin serve --release
//!
//! curl http://localhost:3000/v1/chat/completions \
//!   -H "Content-Type: application/json" \
//!   -d '{"model":"qwen35-opus46","messages":[{"role":"user","content":"Hello"}]}'
//! ```

#[cfg(feature = "serve")]
mod server {
    use axum::{
        extract::State,
        http::StatusCode,
        response::Json,
        routing::{get, post},
        Router,
    };
    use serde_json::{json, Value};
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};

    use lance_graph_planner::cache::candidate_pool::Phase;
    use lance_graph_planner::cache::kv_bundle::HeadPrint;
    use lance_graph_planner::cache::nars_engine::{
        analytical_style, creative_style, empathetic_style, style_score,
        NarsEngine, SpoDistances, SpoHead, MASK_PO, MASK_SO, MASK_SPO,
    };
    use lance_graph_planner::cache::triple_model::TripleModel;
    use lance_graph_planner::strategy::chat_bundle::AutocompleteCache;

    type AppState = std::sync::Arc<Mutex<AutocompleteCache>>;

    fn timestamp() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
    }

    fn message_to_headprint(content: &str) -> HeadPrint {
        // Hash message content into Base17 fingerprint
        let mut dims = [0i16; 17];
        let bytes = content.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            dims[i % 17] = dims[i % 17].wrapping_add(b as i16 * 31);
        }
        // Normalize
        for d in &mut dims {
            *d = (*d % 1000).abs() as i16;
        }
        HeadPrint { dims }
    }

    fn phase_to_str(phase: Phase) -> &'static str {
        match phase {
            Phase::Exposition => "exposition",
            Phase::Durchfuehrung => "development",
            Phase::Contrapunkt => "counterpoint",
            Phase::Bridge => "bridge",
            Phase::Pointe => "resolution",
            Phase::Coda => "coda",
        }
    }

    async fn health() -> &'static str { "ok" }

    async fn list_models() -> Json<Value> {
        Json(json!({
            "object": "list",
            "data": [
                {"id": "qwen35-opus46", "object": "model", "owned_by": "ada", "created": timestamp()},
                {"id": "qwen35-opus45", "object": "model", "owned_by": "ada", "created": timestamp()},
                {"id": "qwen35-9b", "object": "model", "owned_by": "ada", "created": timestamp()},
            ]
        }))
    }

    async fn chat_completions(
        State(state): State<AppState>,
        Json(req): Json<Value>,
    ) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
        let model = req.get("model").and_then(|v| v.as_str()).unwrap_or("qwen35-opus46");
        let messages = req.get("messages").and_then(|v| v.as_array()).cloned().unwrap_or_default();

        if messages.is_empty() {
            return Err((StatusCode::BAD_REQUEST, Json(json!({
                "error": {"message": "messages array is empty", "type": "invalid_request_error"}
            }))));
        }

        let mut cache = state.lock().unwrap();

        // Process each message through the cache
        let mut last_content = String::new();
        let mut cache_hit = false;

        for msg in &messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");

            let fp = message_to_headprint(content);

            match role {
                "user" => {
                    if let Some(spo) = cache.on_user_message(&fp) {
                        // Cache hit — we have a candidate
                        cache_hit = true;
                        last_content = format!(
                            "[Cache HIT] Palette route: S={} P={} O={} | \
                             NARS f={:.3} c={:.3} E={:.3} | \
                             Pearl mask={:03b} | \
                             Phase: {} | \
                             Model: {}",
                            spo.s_idx, spo.p_idx, spo.o_idx,
                            spo.frequency(), spo.confidence(), spo.expectation(),
                            spo.pearl,
                            phase_to_str(cache.phase()),
                            model,
                        );
                    } else {
                        // Cache miss — would normally call LLM
                        let surprise = cache.triple.free_energy(&fp);
                        let alignment = cache.triple.alignment();
                        last_content = format!(
                            "[Cache MISS → LLM fallthrough] \
                             Surprise={:.3} Alignment={:.3} | \
                             Phase: {} | \
                             Pool: {} candidates | \
                             DK: self={:?} user={:?} | \
                             Model: {}",
                            surprise, alignment,
                            phase_to_str(cache.phase()),
                            cache.pool.count(),
                            cache.triple.self_model.dk,
                            cache.triple.user_model.dk,
                            model,
                        );
                    }
                }
                "assistant" => {
                    cache.on_self_output(&fp);
                }
                _ => {} // system, tool — pass through
            }
        }

        let response_id = format!("chatcmpl-ada-{}", timestamp());

        Ok(Json(json!({
            "id": response_id,
            "object": "chat.completion",
            "created": timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": last_content,
                },
                "finish_reason": if cache.should_stop() { "stop" } else { "length" },
            }],
            "usage": {
                "prompt_tokens": messages.len(),
                "completion_tokens": 1,
                "total_tokens": messages.len() + 1,
            },
            "system_fingerprint": format!("palette-{}", phase_to_str(cache.phase())),
        })))
    }

    pub async fn run(port: u16) {
        let cache = AutocompleteCache::new();
        let state: AppState = std::sync::Arc::new(Mutex::new(cache));

        let app = Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state);

        let addr = format!("0.0.0.0:{port}");
        eprintln!("lance-graph-planner serve listening on {addr}");
        eprintln!("  POST /v1/chat/completions  (OpenAI compatible)");
        eprintln!("  GET  /v1/models");
        eprintln!("  GET  /health");
        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    }
}

#[cfg(feature = "serve")]
#[tokio::main]
async fn main() {
    let port: u16 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    server::run(port).await;
}

#[cfg(not(feature = "serve"))]
fn main() {
    eprintln!("Enable the 'serve' feature:");
    eprintln!("  cargo run --manifest-path crates/lance-graph-planner/Cargo.toml --features serve --bin serve");
    std::process::exit(1);
}
