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

    /// Compiled palette pipeline: bgz17 Palette → DistanceMatrix → SimilarityTable.
    /// Built once at startup from bgz7 weight rows. All subsequent lookups are O(1).
    struct PalettePipeline {
        /// 256 archetypal Base17 patterns from weight manifold.
        palette: bgz17::palette::Palette,
        /// 256×256 precomputed L1 distances (128 KB, L1-cache resident).
        distance: bgz17::distance_matrix::DistanceMatrix,
        /// σ-calibrated CDF: raw distance → [0.0, 1.0] similarity.
        similarity: bgz17::similarity::SimilarityTable,
    }

    struct ServerState {
        cache: AutocompleteCache,
        pipeline: Option<PalettePipeline>,
    }

    type AppState = std::sync::Arc<Mutex<ServerState>>;

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

    /// Convert ndarray HeadPrint (Base17) to bgz17 Base17 for palette lookup.
    /// Both types have identical layout: dims: [i16; 17].
    fn headprint_to_bgz17(hp: &HeadPrint) -> bgz17::base17::Base17 {
        bgz17::base17::Base17 { dims: hp.dims }
    }

    /// Score a message against the palette pipeline.
    /// Returns (palette_index, best_match_index, similarity_score).
    fn palette_score(
        pipeline: &PalettePipeline,
        query: &HeadPrint,
        cached_indices: &[u8],
    ) -> (u8, usize, f32) {
        let bgz_query = headprint_to_bgz17(query);
        let q_idx = pipeline.palette.nearest(&bgz_query);

        // Find best match among cached palette indices
        let mut best_sim = 0.0f32;
        let mut best_pos = 0usize;
        for (pos, &c_idx) in cached_indices.iter().enumerate() {
            let dist = pipeline.distance.distance(q_idx, c_idx) as u32;
            let sim = pipeline.similarity.similarity(dist);
            if sim > best_sim {
                best_sim = sim;
                best_pos = pos;
            }
        }
        (q_idx, best_pos, best_sim)
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

        // Validate model name
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

        // Process each message through the cache
        let mut last_content = String::new();
        let mut cache_hit = false;

        for msg in &messages {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");

            let fp = message_to_headprint(content);

            match role {
                "user" => {
                    // Try palette pipeline first (σ-calibrated scoring)
                    if let Some(ref pipeline) = server.pipeline {
                        let (q_idx, best_pos, sim) = palette_score(
                            pipeline,
                            &fp,
                            &server.cache.palette_indices,
                        );

                        if sim > 0.3 {
                            // Palette HIT — σ-calibrated similarity above threshold
                            cache_hit = true;
                            let dist = pipeline.distance.distance(
                                q_idx,
                                server.cache.palette_indices.get(best_pos).copied().unwrap_or(0),
                            );
                            last_content = format!(
                                "[Palette HIT] idx={} match={} dist={} sim={:.4} | \
                                 Phase: {} | \
                                 Palette k={} | \
                                 σ-calibrated | \
                                 Model: {}",
                                q_idx, best_pos, dist, sim,
                                phase_to_str(server.cache.phase()),
                                pipeline.palette.len(),
                                model,
                            );
                        } else {
                            // Palette MISS — similarity too low, fall through
                            let surprise = server.cache.triple.free_energy(&fp);
                            let alignment = server.cache.triple.alignment();
                            last_content = format!(
                                "[Palette MISS → LLM] idx={} best_sim={:.4} | \
                                 Surprise={:.3} Alignment={:.3} | \
                                 Phase: {} | \
                                 Pool: {} candidates | \
                                 Model: {}",
                                q_idx, sim,
                                surprise, alignment,
                                phase_to_str(server.cache.phase()),
                                server.cache.pool.count(),
                                model,
                            );
                        }
                    } else if let Some(spo) = server.cache.on_user_message(&fp) {
                        // Fallback: old cache path (no palette pipeline)
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
                            phase_to_str(server.cache.phase()),
                            model,
                        );
                    } else {
                        // Cache miss — no pipeline, no cache hit
                        let surprise = server.cache.triple.free_energy(&fp);
                        let alignment = server.cache.triple.alignment();
                        last_content = format!(
                            "[Cache MISS → LLM fallthrough] \
                             Surprise={:.3} Alignment={:.3} | \
                             Phase: {} | \
                             Pool: {} candidates | \
                             DK: self={:?} user={:?} | \
                             Model: {}",
                            surprise, alignment,
                            phase_to_str(server.cache.phase()),
                            server.cache.pool.count(),
                            server.cache.triple.self_model.dk,
                            server.cache.triple.user_model.dk,
                            model,
                        );
                    }
                }
                "assistant" => {
                    server.cache.on_self_output(&fp);
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
                "finish_reason": if server.cache.should_stop() { "stop" } else { "length" },
            }],
            "usage": {
                "prompt_tokens": messages.len(),
                "completion_tokens": 1,
                "total_tokens": messages.len() + 1,
            },
            "system_fingerprint": format!("palette-{}", phase_to_str(server.cache.phase())),
        })))
    }

    /// Load Base17 rows from a bgz7 file into HeadPrints.
    /// Delegates to ndarray's canonical bgz7 parser.
    fn load_bgz7(path: &str) -> Vec<(String, Vec<HeadPrint>)> {
        match ndarray::hpc::gguf_indexer::read_bgz7_file(path) {
            Ok(tensors) => tensors
                .into_iter()
                .map(|ct| {
                    // Cap rows at 1000 per tensor to match previous behavior
                    let rows: Vec<HeadPrint> = ct.rows.into_iter().take(1000).collect();
                    (ct.name, rows)
                })
                .collect(),
            Err(e) => {
                eprintln!("  SKIP {path}: {e}");
                Vec::new()
            }
        }
    }

    /// Build the palette pipeline from bgz7 weight rows.
    /// Returns (PalettePipeline, palette_indices) for all collected Base17 rows.
    fn build_palette_pipeline(all_rows: &[HeadPrint]) -> (PalettePipeline, Vec<u8>) {
        // Convert HeadPrint (ndarray Base17) → bgz17 Base17 for palette building
        let bgz_rows: Vec<bgz17::base17::Base17> = all_rows
            .iter()
            .map(|hp| bgz17::base17::Base17 { dims: hp.dims })
            .collect();

        eprintln!("  Building palette from {} weight rows...", bgz_rows.len());
        let palette = bgz17::palette::Palette::build(&bgz_rows, 256, 10);
        eprintln!("  Palette: {} archetypes", palette.len());

        let distance = bgz17::distance_matrix::DistanceMatrix::build(&palette);
        eprintln!("  DistanceMatrix: {} KB", distance.byte_size() / 1024);

        // Collect all pairwise distances for SimilarityTable calibration
        let k = palette.len();
        let mut reservoir: Vec<u32> = Vec::with_capacity(k * (k - 1) / 2);
        for i in 0..k {
            for j in (i + 1)..k {
                reservoir.push(distance.distance(i as u8, j as u8) as u32);
            }
        }
        let similarity = bgz17::similarity::SimilarityTable::from_reservoir(&mut reservoir);
        eprintln!("  SimilarityTable: bucket_width={} max_dist={}",
            similarity.bucket_width(), similarity.max_distance());

        // Assign all weight rows to palette indices
        let indices: Vec<u8> = bgz_rows.iter().map(|r| palette.nearest(r)).collect();
        eprintln!("  Assigned {} rows to palette indices", indices.len());

        (PalettePipeline { palette, distance, similarity }, indices)
    }

    /// Populate attention matrix from bgz7 weight fingerprints.
    fn populate_cache(server: &mut ServerState, v2_path: &str, base_path: &str) {
        eprintln!("Loading Qwen3.5-27B v2 (Opus 4.6) weights...");
        let v2_tensors = load_bgz7(v2_path);
        eprintln!("  {} tensors, {} total rows",
            v2_tensors.len(),
            v2_tensors.iter().map(|(_, r)| r.len()).sum::<usize>());

        eprintln!("Loading Qwen3.5-27B base weights...");
        let base_tensors = load_bgz7(base_path);
        eprintln!("  {} tensors, {} total rows",
            base_tensors.len(),
            base_tensors.iter().map(|(_, r)| r.len()).sum::<usize>());

        // Collect ALL weight rows for palette building
        let mut all_rows: Vec<HeadPrint> = Vec::new();
        for (_, rows) in &v2_tensors {
            all_rows.extend_from_slice(rows);
        }
        for (_, rows) in &base_tensors {
            all_rows.extend_from_slice(rows);
        }

        // Build palette pipeline
        if !all_rows.is_empty() {
            let (pipeline, indices) = build_palette_pipeline(&all_rows);
            server.cache.palette_indices = indices;
            server.pipeline = Some(pipeline);
        }

        // Populate self_model with v2 weights (what Opus 4.6 looks like)
        let cache = &mut server.cache;
        let mut head_count = 0usize;
        for (_name, rows) in &v2_tensors {
            for (r, fp) in rows.iter().enumerate().take(64) {
                let row = head_count % 64;
                let col = r % 64;
                cache.triple.self_model.matrix.set(row, col, fp.clone());
                head_count += 1;
            }
            if head_count >= 4096 { break; }
        }
        eprintln!("  self_model: {} heads populated", head_count.min(4096));

        // Populate user_model with base weights (what the user "knows")
        head_count = 0;
        for (_name, rows) in &base_tensors {
            for (r, fp) in rows.iter().enumerate().take(64) {
                let row = head_count % 64;
                let col = r % 64;
                cache.triple.user_model.matrix.set(row, col, fp.clone());
                head_count += 1;
            }
            if head_count >= 4096 { break; }
        }
        eprintln!("  user_model: {} heads populated", head_count.min(4096));

        // Impact model starts as diff: where self and user diverge
        for row in 0..64 {
            for col in 0..64 {
                let s = cache.triple.self_model.matrix.get(row, col);
                let u = cache.triple.user_model.matrix.get(row, col);
                let dist = s.l1(u);
                if dist > 0 {
                    let mut impact_dims = [0i16; 17];
                    for d in 0..17 {
                        impact_dims[d] = s.dims[d].wrapping_sub(u.dims[d]);
                    }
                    cache.triple.impact_model.matrix.set(row, col, HeadPrint { dims: impact_dims });
                }
            }
        }
        eprintln!("  impact_model: populated from diff");
        eprintln!("  Gestalt L1 (self vs user): {}",
            cache.triple.self_model.matrix.gestalt.l1(&cache.triple.user_model.matrix.gestalt));
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

        let server = state.lock().unwrap();

        // Embed as Base17 fingerprint (17 dims, golden-step folding)
        let fp = message_to_headprint(input);
        let mut embedding: Vec<f64> = fp.dims.iter().map(|d| *d as f64 / 10000.0).collect();

        // If palette pipeline available, append palette index as extra dim
        if let Some(ref pipeline) = server.pipeline {
            let bgz = headprint_to_bgz17(&fp);
            let idx = pipeline.palette.nearest(&bgz);
            embedding.push(idx as f64 / 256.0);
        }

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
        let mut server = ServerState {
            cache: AutocompleteCache::new(),
            pipeline: None,
        };

        // Try to load bgz7 weights from /tmp/ (from indexing session)
        let v2_shard = "/tmp/qwen35_27b_v2_shard02.bgz7";
        let base_shard = "/tmp/qwen35_27b_base_shard02.bgz7";
        if std::fs::metadata(v2_shard).is_ok() && std::fs::metadata(base_shard).is_ok() {
            populate_cache(&mut server, v2_shard, base_shard);
        } else {
            eprintln!("No bgz7 weights found in /tmp/ — running with empty cache");
            eprintln!("  Run indexing first or hydrate --download qwen35-27b-distilled-v2");
        }

        if server.pipeline.is_some() {
            eprintln!("Palette pipeline: ACTIVE (σ-calibrated scoring)");
        } else {
            eprintln!("Palette pipeline: INACTIVE (no weight data)");
        }

        let state: AppState = std::sync::Arc::new(Mutex::new(server));

        let app = Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/embeddings", post(embeddings))
            .with_state(state);

        let addr = format!("0.0.0.0:{port}");
        eprintln!("lance-graph-planner serve listening on {addr}");
        eprintln!("  POST /v1/chat/completions  (OpenAI compatible)");
        eprintln!("  POST /v1/embeddings         (Base17 fingerprints)");
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
