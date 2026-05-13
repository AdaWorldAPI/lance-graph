//! OpenAI-compatible REST server powered by lance-graph-planner.
//!
//! Request flow:
//!   message → extract SPO triplets → triplet_to_headprint → headprint_to_spo
//!   → NarsEngine.score() with SpoDistances + StyleVector → NARS reasoning
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
    use lance_graph_planner::cache::convergence::{headprint_to_spo, triplet_to_headprint};
    use lance_graph_planner::cache::nars_engine::{
        analytical_style, nars_infer, Inference, SpoHead,
    };
    use lance_graph_planner::strategy::chat_bundle::AutocompleteCache;

    struct ServerState {
        cache: AutocompleteCache,
        /// SPO heads from ingested weight tensors (the knowledge base).
        knowledge: Vec<SpoHead>,
        /// Last context SpoHead (for NARS scoring with style vectors).
        context: SpoHead,
    }

    type AppState = std::sync::Arc<Mutex<ServerState>>;

    fn timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Extract SPO triplets from text using verb-pattern matching.
    /// Returns (subject, predicate, object) tuples.
    fn extract_triplets(text: &str) -> Vec<(String, String, String)> {
        let mut triplets = Vec::new();
        // Split on sentence boundaries
        for sentence in text.split(|c| c == '.' || c == '!' || c == '?' || c == '\n') {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            let words: Vec<&str> = sentence.split_whitespace().collect();
            if words.len() < 2 {
                continue;
            }

            // Find verb position by morphological cues or common verb list
            let verb_pos = words.iter().position(|w| {
                let w = w.to_lowercase();
                w.ends_with("ed")
                    || w.ends_with("ing")
                    || w.ends_with("es")
                    || w.ends_with("ize")
                    || w.ends_with("ify")
                    || COMMON_VERBS.contains(&w.as_str())
            });

            if let Some(vp) = verb_pos {
                if vp > 0 && vp < words.len() - 1 {
                    let subject = words[..vp].join(" ");
                    let predicate = words[vp].to_string();
                    let object = words[vp + 1..].join(" ");
                    triplets.push((subject, predicate, object));
                }
            } else if words.len() >= 3 {
                // Fallback: first word = S, second = P, rest = O
                triplets.push((
                    words[0].to_string(),
                    words[1].to_string(),
                    words[2..].join(" "),
                ));
            } else if words.len() == 2 {
                // Intransitive: S P (no object)
                triplets.push((words[0].to_string(), words[1].to_string(), String::new()));
            }
        }
        triplets
    }

    const COMMON_VERBS: &[&str] = &[
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "need",
        "know",
        "think",
        "want",
        "like",
        "use",
        "find",
        "give",
        "tell",
        "say",
        "get",
        "make",
        "go",
        "see",
        "come",
        "take",
        "help",
        "show",
        "try",
        "ask",
        "work",
        "call",
        "keep",
        "let",
        "begin",
        "seem",
        "run",
        "move",
        "live",
        "believe",
        "hold",
        "bring",
        "happen",
        "write",
        "provide",
        "sit",
        "stand",
        "lose",
        "pay",
        "meet",
        "include",
        "continue",
        "set",
        "learn",
        "change",
        "lead",
        "understand",
        "watch",
        "follow",
        "stop",
        "create",
        "speak",
        "read",
        "allow",
        "add",
        "spend",
        "grow",
        "open",
        "walk",
        "win",
        "offer",
        "remember",
        "love",
        "consider",
        "appear",
        "buy",
        "wait",
        "serve",
        "die",
        "send",
        "expect",
        "build",
        "stay",
        "fall",
        "cut",
        "reach",
        "kill",
        "remain",
        "causes",
        "enables",
        "supports",
    ];

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

    async fn health() -> &'static str {
        "ok"
    }

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
        let model = req
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("qwen35-opus46");
        let messages = req
            .get("messages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        const VALID_MODELS: &[&str] = &[
            "qwen35-opus46",
            "qwen35-opus45",
            "qwen35-9b",
            "reader-lm",
            "bge-m3",
            "llama4-scout",
            "openchat-3.5",
        ];
        if !VALID_MODELS.contains(&model) {
            return Err((
                StatusCode::NOT_FOUND,
                Json(json!({
                    "error": {
                        "message": format!("Model '{}' not found. Available: {}", model, VALID_MODELS.join(", ")),
                        "type": "invalid_request_error",
                        "code": "model_not_found"
                    }
                })),
            ));
        }

        if messages.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {"message": "messages array is empty", "type": "invalid_request_error"}
                })),
            ));
        }

        let mut server = state.lock().unwrap();
        let style = analytical_style();

        let mut last_content = String::new();

        for msg in &messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");

            match role {
                "user" => {
                    // 1. Extract SPO triplets from message text
                    let triplets = extract_triplets(content);

                    if triplets.is_empty() {
                        // Can't decompose — use whole message as single SPO
                        let fp = triplet_to_headprint(content, "states", "");
                        let spo = headprint_to_spo(&fp, 0.9, 0.5);
                        let score = server.cache.nars.score(&spo, &server.context, &style);

                        last_content = format!(
                            "[SPO] S={} P={} O={} | score={:.3} E={:.3} | \
                             Phase: {} | Model: {}",
                            spo.s_idx,
                            spo.p_idx,
                            spo.o_idx,
                            score,
                            spo.expectation(),
                            phase_to_str(server.cache.phase()),
                            model,
                        );
                        server.context = spo;
                        continue;
                    }

                    // 2. Process each triplet through the convergence pipeline
                    let mut results = Vec::new();
                    for (s, p, o) in &triplets {
                        let fp = triplet_to_headprint(s, p, o);
                        let spo = headprint_to_spo(&fp, 0.9, 0.7);

                        // 3. Score against context using NARS + style vector
                        let score = server.cache.nars.score(&spo, &server.context, &style);

                        // 4. NARS inference against knowledge base
                        let mut best_inference = None;
                        let mut best_truth_e = 0.0f32;
                        for known in &server.knowledge {
                            // Try deduction: known → spo
                            let truth = nars_infer(known, &spo, Inference::Deduction);
                            let e = truth.expectation();
                            if e > best_truth_e {
                                best_truth_e = e;
                                best_inference =
                                    Some(("deduction", known.s_idx, known.p_idx, known.o_idx, e));
                            }
                            // Try abduction: spo ← known
                            let truth = nars_infer(&spo, known, Inference::Abduction);
                            let e = truth.expectation();
                            if e > best_truth_e {
                                best_truth_e = e;
                                best_inference =
                                    Some(("abduction", known.s_idx, known.p_idx, known.o_idx, e));
                            }
                        }

                        // 5. Update context (the last SPO becomes the new context)
                        server.cache.nars.on_emit(&spo);
                        server.context = spo.clone();

                        let inference_str = match best_inference {
                            Some((rule, s, p, o, e)) => {
                                format!(" | NARS {}→[{},{},{}] E={:.3}", rule, s, p, o, e)
                            }
                            None => String::new(),
                        };

                        results.push(format!(
                            "({} —{}→ {}) S={} P={} O={} score={:.3}{}",
                            s, p, o, spo.s_idx, spo.p_idx, spo.o_idx, score, inference_str,
                        ));
                    }

                    last_content = format!(
                        "[SPO×{}] {} | Phase: {} | knowledge={} | Model: {}",
                        triplets.len(),
                        results.join(" ; "),
                        phase_to_str(server.cache.phase()),
                        server.knowledge.len(),
                        model,
                    );
                }
                "assistant" => {
                    // Extract triplets from assistant response, add to knowledge
                    let triplets = extract_triplets(content);
                    for (s, p, o) in &triplets {
                        let fp = triplet_to_headprint(s, p, o);
                        let spo = headprint_to_spo(&fp, 0.85, 0.8);
                        server.knowledge.push(spo);
                    }
                    let fp = triplet_to_headprint(content, "responds", "");
                    server.cache.on_self_output(&fp);
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
                "finish_reason": if server.cache.should_stop() { "stop" } else { "length" },
            }],
            "usage": {
                "prompt_tokens": messages.len(),
                "completion_tokens": 1,
                "total_tokens": messages.len() + 1,
            },
            "system_fingerprint": format!("spo-{}", phase_to_str(server.cache.phase())),
        })))
    }

    async fn embeddings(
        State(_state): State<AppState>,
        Json(req): Json<Value>,
    ) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
        let model = req
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("bge-m3");
        let input = req
            .get("input")
            .and_then(|v| v.as_str())
            .or_else(|| {
                req.get("input")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_str())
            })
            .unwrap_or("");

        if input.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {"message": "input is empty", "type": "invalid_request_error"}
                })),
            ));
        }

        // SPO-decomposed embedding: extract triplets, encode each, bundle
        let triplets = extract_triplets(input);
        let embedding: Vec<f64> = if !triplets.is_empty() {
            // Average of all triplet HeadPrints
            let mut sums = [0.0f64; 17];
            for (s, p, o) in &triplets {
                let fp = triplet_to_headprint(s, p, o);
                for d in 0..17 {
                    sums[d] += fp.dims[d] as f64;
                }
            }
            let n = triplets.len() as f64;
            sums.iter().map(|s| s / n).collect()
        } else {
            let fp = triplet_to_headprint(input, "states", "");
            fp.dims.iter().map(|&d| d as f64).collect()
        };

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

    /// Load bgz7 weight shards into knowledge base as SPO heads.
    fn ingest_weights(knowledge: &mut Vec<SpoHead>, path: &str) {
        match ndarray::hpc::gguf_indexer::read_bgz7_file(path) {
            Ok(tensors) => {
                for ct in tensors {
                    // Each tensor becomes an SPO: tensor_name → "encodes" → layer
                    let fp = triplet_to_headprint(&ct.name, "encodes", "weights");
                    let spo = headprint_to_spo(&fp, 0.95, 0.99);
                    knowledge.push(spo);

                    // Sample weight rows as additional knowledge
                    for (_r, row) in ct.rows.iter().enumerate().take(100) {
                        let row_spo = headprint_to_spo(row, 0.9, 0.95);
                        knowledge.push(row_spo);
                    }
                }
            }
            Err(e) => eprintln!("  SKIP {path}: {e}"),
        }
    }

    pub async fn run(port: u16) {
        let mut knowledge = Vec::new();

        // Ingest available bgz7 shards into knowledge base
        for path in &[
            "/tmp/qwen35_27b_v2_shard02.bgz7",
            "/tmp/qwen35_27b_base_shard02.bgz7",
        ] {
            if std::fs::metadata(path).is_ok() {
                eprintln!("Ingesting {path}...");
                ingest_weights(&mut knowledge, path);
            }
        }
        eprintln!("Knowledge base: {} SPO heads", knowledge.len());

        let server = ServerState {
            cache: AutocompleteCache::new(),
            knowledge,
            context: SpoHead::zero(),
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
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    server::run(port).await;
}

#[cfg(not(feature = "serve"))]
fn main() {
    eprintln!("Enable the 'serve' feature:");
    eprintln!("  cargo run --manifest-path crates/lance-graph-planner/Cargo.toml --features serve --bin serve");
    std::process::exit(1);
}
