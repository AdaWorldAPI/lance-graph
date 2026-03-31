//! OpenAI-compatible REST server powered by lance-graph-planner.
//!
//! Weight vectors are raw Base17 (34 bytes, ρ=0.993 vs BF16).
//! No palette indirection — direct L1 on 17 dims is sub-microsecond.
//! At scale: store in LanceDB, use RaBitQ index for ANN search.
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
        NarsEngine, SpoDistances, SpoHead, MASK_SPO,
    };
    use lance_graph_planner::cache::triple_model::TripleModel;
    use lance_graph_planner::strategy::chat_bundle::AutocompleteCache;

    /// Raw weight vectors. 34 bytes each. Direct L1 search.
    struct WeightStore {
        /// All weight rows as raw Base17 vectors.
        vectors: Vec<HeadPrint>,
        /// Tensor name per row (provenance).
        names: Vec<String>,
        /// HEEL: element-wise mean of all vectors (the gestalt).
        heel: HeadPrint,
    }

    impl WeightStore {
        fn new() -> Self {
            Self {
                vectors: Vec::new(),
                names: Vec::new(),
                heel: HeadPrint::zero(),
            }
        }

        /// Add vectors from a bgz7 file.
        fn ingest(&mut self, path: &str) {
            match ndarray::hpc::gguf_indexer::read_bgz7_file(path) {
                Ok(tensors) => {
                    for ct in tensors {
                        for row in ct.rows.into_iter().take(10000) {
                            self.vectors.push(row);
                            self.names.push(ct.name.clone());
                        }
                    }
                }
                Err(e) => eprintln!("  SKIP {path}: {e}"),
            }
        }

        /// Compute HEEL after all ingestion.
        fn compute_heel(&mut self) {
            if self.vectors.is_empty() { return; }
            let n = self.vectors.len() as f64;
            let mut sums = [0.0f64; 17];
            for v in &self.vectors {
                for d in 0..17 { sums[d] += v.dims[d] as f64; }
            }
            for d in 0..17 {
                self.heel.dims[d] = (sums[d] / n).round() as i16;
            }
        }

        /// Direct L1 nearest neighbor search. Returns (index, distance, tensor_name).
        fn nearest(&self, query: &HeadPrint, k: usize) -> Vec<(usize, u32, &str)> {
            let mut scored: Vec<(usize, u32)> = self.vectors.iter()
                .enumerate()
                .map(|(i, v)| (i, query.l1(v)))
                .collect();
            scored.sort_unstable_by_key(|&(_, d)| d);
            scored.truncate(k);
            scored.iter()
                .map(|&(i, d)| (i, d, self.names[i].as_str()))
                .collect()
        }

        fn len(&self) -> usize { self.vectors.len() }
    }

    struct ServerState {
        cache: AutocompleteCache,
        weights: WeightStore,
    }

    type AppState = std::sync::Arc<Mutex<ServerState>>;

    fn timestamp() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
    }

    /// Hash message content into Base17 space.
    /// TODO: replace with BGE-M3 embed → golden-step projection for real semantic matching.
    fn message_to_base17(content: &str) -> HeadPrint {
        let mut dims = [0i16; 17];
        let bytes = content.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            dims[i % 17] = dims[i % 17].wrapping_add(b as i16 * 31);
        }
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
                {"id": "qwen35-opus46", "object": "model", "owned_by": "ada", "created": timestamp(),
                 "description": "Qwen3.5-27B + Opus 4.6 reasoning scaffold (174 MB bgz7)"},
                {"id": "qwen35-opus45", "object": "model", "owned_by": "ada", "created": timestamp(),
                 "description": "Qwen3.5-27B + Opus 4.5 behavioral traits (174 MB bgz7)"},
                {"id": "qwen35-9b", "object": "model", "owned_by": "ada", "created": timestamp(),
                 "description": "Qwen3.5-9B distilled, scale-invariant core (80 MB bgz7)"},
                {"id": "reader-lm", "object": "model", "owned_by": "ada", "created": timestamp(),
                 "description": "jinaai/reader-lm-1.5b HTML→Markdown (26 MB bgz7)"},
                {"id": "bge-m3", "object": "model", "owned_by": "ada", "created": timestamp(),
                 "description": "BAAI/bge-m3 multilingual embeddings (7.3 MB bgz7)"},
                {"id": "llama4-scout", "object": "model", "owned_by": "ada", "created": timestamp(),
                 "description": "Llama-4-Scout-17B MoE (37 MB bgz7)"},
                {"id": "openchat-3.5", "object": "model", "owned_by": "ada", "created": timestamp(),
                 "description": "OpenChat 3.5 Mistral-7B (41 MB bgz7)"},
            ]
        }))
    }

    async fn chat_completions(
        State(state): State<AppState>,
        Json(req): Json<Value>,
    ) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
        let model = req.get("model").and_then(|v| v.as_str()).unwrap_or("qwen35-opus46");
        let messages = req.get("messages").and_then(|v| v.as_array()).cloned().unwrap_or_default();

        const VALID_MODELS: &[&str] = &[
            "qwen35-opus46", "qwen35-opus45", "qwen35-9b",
            "reader-lm", "bge-m3", "llama4-scout", "openchat-3.5",
        ];
        if !VALID_MODELS.contains(&model) {
            return Err((StatusCode::NOT_FOUND, Json(json!({
                "error": {
                    "message": format!("Model '{}' not found. Available: {}", model, VALID_MODELS.join(", ")),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }))));
        }

        if messages.is_empty() {
            return Err((StatusCode::BAD_REQUEST, Json(json!({
                "error": {"message": "messages array is empty", "type": "invalid_request_error"}
            }))));
        }

        let mut server = state.lock().unwrap();

        let mut last_content = String::new();

        for msg in &messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");

            let query = message_to_base17(content);

            match role {
                "user" => {
                    if server.weights.len() > 0 {
                        // Direct L1 nearest neighbor on raw Base17 vectors
                        let neighbors = server.weights.nearest(&query, 5);
                        let heel_dist = query.l1(&server.weights.heel);

                        let top: Vec<String> = neighbors.iter()
                            .map(|(i, d, name)| format!("{}:r{}(d={})", name, i, d))
                            .collect();

                        last_content = format!(
                            "[L1 search] heel_dist={} top_5=[{}] | \
                             vectors={} | Phase: {} | Model: {}",
                            heel_dist,
                            top.join(", "),
                            server.weights.len(),
                            phase_to_str(server.cache.phase()),
                            model,
                        );
                    } else {
                        let surprise = server.cache.triple.free_energy(&query);
                        last_content = format!(
                            "[No weights] Surprise={:.3} | Phase: {} | Model: {}",
                            surprise,
                            phase_to_str(server.cache.phase()),
                            model,
                        );
                    }
                }
                "assistant" => {
                    server.cache.on_self_output(&query);
                }
                _ => {}
            }
        }

        Ok(Json(json!({
            "id": format!("chatcmpl-ada-{}", timestamp()),
            "object": "chat.completion",
            "created": timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": last_content },
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": messages.len(),
                "completion_tokens": 1,
                "total_tokens": messages.len() + 1,
            },
            "system_fingerprint": format!("base17-{}", server.weights.len()),
        })))
    }

    async fn embeddings(
        State(state): State<AppState>,
        Json(req): Json<Value>,
    ) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
        let model = req.get("model").and_then(|v| v.as_str()).unwrap_or("bge-m3");
        let input = req.get("input").and_then(|v| v.as_str())
            .or_else(|| req.get("input").and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_str()))
            .unwrap_or("");

        if input.is_empty() {
            return Err((StatusCode::BAD_REQUEST, Json(json!({
                "error": {"message": "input is empty", "type": "invalid_request_error"}
            }))));
        }

        // Base17: 17 dims, f32 for OpenAI compat
        let fp = message_to_base17(input);
        let embedding: Vec<f64> = fp.dims.iter().map(|&d| d as f64).collect();

        Ok(Json(json!({
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": embedding,
            }],
            "model": model,
            "usage": {
                "prompt_tokens": input.split_whitespace().count(),
                "total_tokens": input.split_whitespace().count(),
            }
        })))
    }

    pub async fn run(port: u16) {
        let mut weights = WeightStore::new();

        // Ingest available bgz7 shards
        for path in &[
            "/tmp/qwen35_27b_v2_shard02.bgz7",
            "/tmp/qwen35_27b_base_shard02.bgz7",
        ] {
            if std::fs::metadata(path).is_ok() {
                eprintln!("Ingesting {path}...");
                weights.ingest(path);
            }
        }

        if weights.len() > 0 {
            weights.compute_heel();
            eprintln!("WeightStore: {} vectors, HEEL={:?}", weights.len(), weights.heel.dims);
        } else {
            eprintln!("No bgz7 weights found — running empty");
        }

        let server = ServerState {
            cache: AutocompleteCache::new(),
            weights,
        };

        let state: AppState = std::sync::Arc::new(Mutex::new(server));

        let app = Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/embeddings", post(embeddings))
            .with_state(state);

        let addr = format!("0.0.0.0:{port}");
        eprintln!("Listening on {addr}");
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
