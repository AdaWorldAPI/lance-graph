//! **LAB-ONLY.** Axum REST server for live testing cognitive shader cycles
//! in the Claude Code backend. Not shipped to consumers.
//!
//! Canonical consumer surface is `UnifiedStep` + `OrchestrationBridge` in
//! the library. This server just exposes that bridge over HTTP for test
//! convenience. Per-op endpoints (`/v1/shader/{calibrate,probe,tensors,
//! dispatch,plan}`) are thin adapters that build a `UnifiedStep` and
//! dispatch through the same bridge; the canonical endpoint is
//! `/v1/shader/route`.
//!
//! ```bash
//! cargo run -p cognitive-shader-driver --features serve --bin shader-serve
//!
//! # Dispatch a cycle:
//! curl -X POST http://localhost:3001/v1/shader/dispatch \
//!   -H "Content-Type: application/json" \
//!   -d '{"row_start": 0, "row_end": 100, "style": {"type": "Auto"}}'
//!
//! # Ingest codebook indices:
//! curl -X POST http://localhost:3001/v1/shader/ingest \
//!   -H "Content-Type: application/json" \
//!   -d '{"codebook_indices": [42, 100, 200], "source_ordinal": 1}'
//!
//! # Health + neural-debug diagnostics:
//! curl http://localhost:3001/v1/shader/health
//!
//! # Read qualia (CMYK → RGB):
//! curl http://localhost:3001/v1/shader/qualia/0
//!
//! # List all 12 unified styles:
//! curl http://localhost:3001/v1/shader/styles
//! ```

use std::sync::{Arc, Mutex};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};

use crate::codec_research;
use crate::driver::ShaderDriver;
use crate::engine_bridge::{self, unified_style, UNIFIED_STYLES};
use crate::token_agreement::{ReferenceModel, TokenAgreementHarness};
use crate::wire::{
    WireCalibrateRequest, WireCalibrateResponse, WireCrystal, WireDispatch, WireEncode,
    WireEncodeResponse, WireHealth, WireIngest, WirePlanRequest, WirePlanResponse,
    WireProbeRequest, WireProbeResponse, WireQualia, WireRunbookRequest, WireRunbookResponse,
    WireRunbookStep, WireRunbookStepResult, WireStepResult, WireStyleInfo, WireSweepRequest,
    WireSweepResponse, WireSweepResult, WireTensorsRequest, WireTensorsResponse,
    WireTokenAgreement, WireTokenAgreementResult, WireUnifiedStep,
};
use lance_graph_contract::cam::CodecParams;
use std::path::Path as StdPath;
use lance_graph_contract::cognitive_shader::CognitiveShaderDriver;

struct ServerState {
    driver: ShaderDriver,
    write_cursor: usize,
    /// Planner instance (shared across requests). Only present when the
    /// shader-driver was compiled with `--features with-planner`.
    #[cfg(feature = "with-planner")]
    planner: lance_graph_planner::PlannerAwareness,
}

type AppState = Arc<Mutex<ServerState>>;

pub fn router(driver: ShaderDriver) -> Router {
    let state: AppState = Arc::new(Mutex::new(ServerState {
        driver,
        write_cursor: 0,
        #[cfg(feature = "with-planner")]
        planner: crate::planner_bridge::build_planner(&[]),
    }));

    Router::new()
        .route("/v1/shader/dispatch", post(dispatch_handler))
        .route("/v1/shader/ingest", post(ingest_handler))
        .route("/v1/shader/health", get(health_handler))
        .route("/v1/shader/qualia/{row}", get(qualia_handler))
        .route("/v1/shader/styles", get(styles_handler))
        // Codec research operations — same DTO surface, no separate endpoint.
        // Lets clients encode / measure / probe without recompiling; the
        // codec parameters (num_subspaces, num_centroids, kmeans_iterations,
        // max_rows) are DTO fields, so one running server drives every
        // codec-calibration experiment.
        .route("/v1/shader/tensors", post(tensors_handler))
        .route("/v1/shader/calibrate", post(calibrate_handler))
        .route("/v1/shader/probe", post(probe_handler))
        // D2.3 — I11 cert gate endpoint. Handler routes to
        // TokenAgreementHarness::measure_stub() until D2.2 lands the real
        // decode-and-compare loop. Stub result carries `stub:true` +
        // `backend:"stub"` so clients cannot confuse Phase 0 stub output
        // for a real measurement (anti-#219 defense, type-level).
        .route("/v1/shader/token-agreement", post(token_agreement_handler))
        // D3.1 — codec sweep endpoint (batch mode). Client POSTs a
        // WireSweepRequest containing a cross-product grid; handler
        // enumerates grid, validates each candidate, builds stub results,
        // returns WireSweepResponse. SSE streaming + Lance append land in
        // D3.1b; this batch path stays for clients that want all results
        // in one response without streaming.
        .route("/v1/shader/sweep", post(sweep_handler))
        // Scheduled runbook: one POST runs a list of steps. Test injection
        // lands here — a client script submits its full codec-research
        // protocol as a single DTO, the server executes and returns all
        // results correlated by per-step label.
        .route("/v1/shader/runbook", post(runbook_handler))
        // Planner delegation (Layer 4 per INTEGRATION_PLAN_CS.md). Without
        // `with-planner` the handler returns 503 so the unified endpoint
        // URL shape stays stable whether or not planner is compiled in.
        .route("/v1/shader/plan", post(plan_handler))
        // Generic OrchestrationBridge gateway — route any UnifiedStep by step_type.
        // Composed bridges cover lg.* (planner) + nd.* (codec research).
        .route("/v1/shader/route", post(route_handler))
        // JIT lens encode pipeline — text → DeepNSM → 512-bit VSA → 16Kbit BindSpace row.
        .route("/v1/shader/encode", post(encode_handler))
        .with_state(state)
}

async fn dispatch_handler(
    State(state): State<AppState>,
    Json(wire): Json<WireDispatch>,
) -> Result<Json<WireCrystal>, (StatusCode, Json<Value>)> {
    let internal = wire.to_internal();
    let st = state.lock().map_err(|_| {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "lock poisoned"})))
    })?;
    let crystal = st.driver.dispatch(&internal);
    Ok(Json(WireCrystal::from(&crystal)))
}

async fn ingest_handler(
    State(state): State<AppState>,
    Json(wire): Json<WireIngest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let mut st = state.lock().map_err(|_| {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "lock poisoned"})))
    })?;
    let cursor = st.write_cursor;
    let bs = Arc::get_mut(&mut st.driver.bindspace).ok_or_else(|| {
        (StatusCode::CONFLICT, Json(json!({"error": "bindspace has multiple references"})))
    })?;
    let (start, end) = engine_bridge::ingest_codebook_indices(
        bs,
        &wire.codebook_indices,
        wire.source_ordinal,
        wire.timestamp,
        cursor,
    );
    st.write_cursor = end as usize;
    Ok(Json(json!({
        "ingested": end - start,
        "row_start": start,
        "row_end": end,
        "write_cursor": st.write_cursor,
    })))
}

async fn health_handler(
    State(state): State<AppState>,
) -> Json<WireHealth> {
    let st = state.lock().unwrap();
    Json(WireHealth {
        row_count: st.driver.row_count(),
        byte_footprint: st.driver.byte_footprint(),
        styles: UNIFIED_STYLES.iter().map(|s| WireStyleInfo {
            ordinal: s.ordinal,
            name: s.name.to_string(),
            layer_mask: s.layer_mask,
            density_target: s.density_target,
            resonance_threshold: s.resonance_threshold,
            fan_out: s.fan_out,
        }).collect(),
        neural_debug: None, // populated when neural-debug dep is wired
    })
}

async fn qualia_handler(
    State(state): State<AppState>,
    Path(row): Path<u32>,
) -> Result<Json<WireQualia>, (StatusCode, Json<Value>)> {
    let st = state.lock().map_err(|_| {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "lock poisoned"})))
    })?;
    let bs = st.driver.bindspace();
    if row as usize >= bs.len {
        return Err((StatusCode::NOT_FOUND, Json(json!({"error": "row out of range"}))));
    }
    let (experienced, cd) = engine_bridge::read_qualia_decomposed(bs, row as usize);
    let style_ord = crate::auto_style::style_from_qualia(&experienced);
    let style_name = unified_style(style_ord).name.to_string();
    Ok(Json(WireQualia {
        row,
        experienced: experienced.to_vec(),
        classification_distance: cd,
        style_name,
    }))
}

async fn styles_handler() -> Json<Vec<WireStyleInfo>> {
    Json(UNIFIED_STYLES.iter().map(|s| WireStyleInfo {
        ordinal: s.ordinal,
        name: s.name.to_string(),
        layer_mask: s.layer_mask,
        density_target: s.density_target,
        resonance_threshold: s.resonance_threshold,
        fan_out: s.fan_out,
    }).collect())
}

// ─── Codec research handlers ────────────────────────────────────────────────

async fn tensors_handler(
    Json(req): Json<WireTensorsRequest>,
) -> Result<Json<WireTensorsResponse>, (StatusCode, Json<Value>)> {
    codec_research::list_tensors(&req)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))
}

async fn calibrate_handler(
    Json(req): Json<WireCalibrateRequest>,
) -> Result<Json<WireCalibrateResponse>, (StatusCode, Json<Value>)> {
    codec_research::calibrate_tensor(&req)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))
}

async fn probe_handler(
    Json(req): Json<WireProbeRequest>,
) -> Result<Json<WireProbeResponse>, (StatusCode, Json<Value>)> {
    codec_research::row_count_probe(&req)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))
}

/// D2.3 — `POST /v1/shader/token-agreement` handler.
///
/// Routes `WireTokenAgreement` through the Phase-0-honest stub path:
///
/// 1. Validates `candidate: WireCodecParams → CodecParams` via TryFrom,
///    surfacing typed errors (precision-ladder, overfit guard) as HTTP 400.
/// 2. Loads reference model via `ReferenceModel::load` when `model_path`
///    points to a real directory; otherwise falls back to
///    `ReferenceModel::stub` so tests can drive the handler without a
///    filesystem.
/// 3. Builds `TokenAgreementHarness` + calls `measure_stub()` (D2.1 stub).
/// 4. Returns `WireTokenAgreementResult { stub:true, backend:"stub", … }`.
///
/// Real decode-and-compare lands at D2.2; the Wire surface + routing are
/// frozen now so client integration work can proceed against the stub.
async fn token_agreement_handler(
    Json(req): Json<WireTokenAgreement>,
) -> Result<Json<WireTokenAgreementResult>, (StatusCode, Json<Value>)> {
    // Validate CodecParams at ingress (precision-ladder / overfit guard
    // fire here, not inside the harness).
    let _params: CodecParams = req
        .candidate
        .clone()
        .try_into()
        .map_err(|e: lance_graph_contract::cam::CodecParamsError| {
            (StatusCode::BAD_REQUEST, Json(json!({"error": format!("invalid CodecParams: {e}")})))
        })?;

    // Reference model — real path if it exists, stub otherwise. D2.2
    // replaces with a strict path check once the safetensors loader lands.
    let model_path = StdPath::new(&req.model_path);
    let reference = if model_path.exists() {
        ReferenceModel::load(model_path).map_err(|e| {
            (StatusCode::BAD_REQUEST, Json(json!({"error": format!("model load: {e}")})))
        })?
    } else {
        // Deterministic stub keyed on the path string so repeated calls
        // return the same stub fingerprint (useful for cache/regression
        // tests that POST synthetic model_path values).
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&req.model_path, &mut h);
        ReferenceModel::stub(std::hash::Hasher::finish(&h), 0)
    };

    let harness = TokenAgreementHarness::new(
        reference,
        req.reference,
        req.candidate,
        req.n_tokens,
    );
    harness
        .measure_stub()
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))))
}

/// D3.1 — `POST /v1/shader/sweep` handler (batch mode).
///
/// Enumerates the cross-product grid from `WireSweepRequest`, validates
/// each candidate via TryFrom(CodecParams), computes kernel_signature +
/// backend per point, and returns all results in one `WireSweepResponse`.
///
/// Stub: per-point calibrate/token_agreement are `None`; Phase 3 real
/// handler invokes the actual codec_research + token_agreement harness.
/// SSE streaming variant (D3.1b) replaces the batch return with per-point
/// Server-Sent Events.
async fn sweep_handler(
    Json(req): Json<WireSweepRequest>,
) -> Result<Json<WireSweepResponse>, (StatusCode, Json<Value>)> {
    let start = std::time::Instant::now();

    // P1 — reject oversized grids before materialization. A small JSON
    // payload with moderately-sized axes can explode into a huge Cartesian
    // product; bound it so the endpoint isn't a DoS vector.
    const MAX_GRID_CARDINALITY: usize = 10_000;
    let cardinality = req.grid.cardinality();
    if cardinality > MAX_GRID_CARDINALITY {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!(
                    "sweep grid cardinality {cardinality} exceeds max {MAX_GRID_CARDINALITY}; \
                     reduce axis dimensions"
                )
            })),
        ));
    }

    let candidates = req.grid.enumerate();

    let mut results = Vec::with_capacity(candidates.len());
    for (idx, wire_params) in candidates.into_iter().enumerate() {
        // Validate each grid point at ingress — surface typed errors early.
        let params: CodecParams = wire_params
            .clone()
            .try_into()
            .map_err(|e: lance_graph_contract::cam::CodecParamsError| {
                (StatusCode::BAD_REQUEST, Json(json!({
                    "error": format!("grid point {idx}: invalid CodecParams: {e}")
                })))
            })?;

        results.push(WireSweepResult {
            grid_index: idx as u32,
            candidate: wire_params,
            kernel_hash: params.kernel_signature(),
            calibrate: None,
            token_agreement: None,
            stub: true,
        });
    }

    Ok(Json(WireSweepResponse {
        label: req.label,
        cardinality: cardinality as u32,
        results,
        elapsed_ms: start.elapsed().as_millis() as u64,
        // P2 — do NOT echo req.log_to_lance into the response when no rows
        // were actually written. Clients that treat lance_fragment_path as
        // evidence of successful logging would silently skip retries and
        // lose experiment results. Set to None until the real Lance append
        // writer lands (Phase 3 D3.1b).
        lance_fragment_path: None,
    }))
}

async fn route_handler(
    State(_state): State<AppState>,
    Json(wire): Json<WireUnifiedStep>,
) -> Result<Json<WireStepResult>, (StatusCode, Json<Value>)> {
    use lance_graph_contract::orchestration::{
        OrchestrationBridge, StepStatus, UnifiedStep,
    };

    let mut step = UnifiedStep {
        id: 0,
        step_id: wire.step_id.clone(),
        step_type: wire.step_type.clone(),
        status: StepStatus::Pending,
        thinking: None,
        reasoning: wire.reasoning,
        confidence: None,
        depends_on: vec![],
    };

    // Try codec research bridge first (nd.*)
    let codec_bridge = crate::codec_bridge::CodecResearchBridge;
    let result = codec_bridge.route(&mut step);

    // If codec bridge rejected with DomainUnavailable, try CypherBridge
    // (lg.cypher). This keeps the nd.* hot path unchanged while adding
    // `lg.cypher` routing ahead of the planner fallthrough.
    if matches!(result, Err(lance_graph_contract::orchestration::OrchestrationError::DomainUnavailable(_))) {
        let cypher_bridge = crate::cypher_bridge::CypherBridge;
        let cypher_result = cypher_bridge.route(&mut step);

        // If CypherBridge also rejected with DomainUnavailable, fall
        // through to the planner bridge for the remaining `lg.*` space.
        if matches!(
            cypher_result,
            Err(lance_graph_contract::orchestration::OrchestrationError::DomainUnavailable(_))
        ) {
            #[cfg(feature = "with-planner")]
            {
                let st = _state.lock().map_err(|_| {
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "lock poisoned"})))
                })?;
                let _ = OrchestrationBridge::route(&st.planner, &mut step);
            }
            #[cfg(not(feature = "with-planner"))]
            {
                step.status = StepStatus::Failed;
                step.reasoning = Some("domain unavailable and planner not compiled in".to_string());
            }
        }
    }

    let status_str = match step.status {
        StepStatus::Completed => "completed",
        StepStatus::Failed => "failed",
        StepStatus::Running => "running",
        StepStatus::Pending => "pending",
        StepStatus::Skipped => "skipped",
    };
    Ok(Json(WireStepResult {
        step_id: step.step_id,
        step_type: step.step_type,
        status: status_str.to_string(),
        reasoning: step.reasoning,
        confidence: step.confidence,
    }))
}

async fn plan_handler(
    State(state): State<AppState>,
    Json(req): Json<WirePlanRequest>,
) -> Result<Json<WirePlanResponse>, (StatusCode, Json<Value>)> {
    run_plan(&state, &req)
        .map(Json)
        .map_err(|(code, msg)| (code, Json(json!({"error": msg}))))
}

#[cfg(feature = "with-planner")]
fn run_plan(
    state: &AppState,
    req: &WirePlanRequest,
) -> Result<WirePlanResponse, (StatusCode, String)> {
    let st = state
        .lock()
        .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "lock poisoned".to_string()))?;
    crate::planner_bridge::plan(&st.planner, req)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))
}

#[cfg(not(feature = "with-planner"))]
fn run_plan(
    _state: &AppState,
    _req: &WirePlanRequest,
) -> Result<WirePlanResponse, (StatusCode, String)> {
    Err((
        StatusCode::SERVICE_UNAVAILABLE,
        "planner not compiled in — rebuild with --features with-planner".to_string(),
    ))
}

// ─── Encode handler ─────────────────────────────────────────────────────────

/// `POST /v1/shader/encode` — text → DeepNSM → 512-bit VSA → 16Kbit BindSpace row.
///
/// Pipeline:
/// 1. Split text into words (whitespace + punctuation).
/// 2. Hash each word to a 12-bit vocabulary rank via SplitMix64-style mixing
///    (deterministic; no data files required — DeepNsm's `VsaVec::from_rank`
///    accepts any u16 rank and produces a stable pseudo-random 512-bit vector).
/// 3. XOR-bind each word vector with a position vector so word order matters:
///    `word_fp = VsaVec::from_rank(hash(word)) XOR VsaVec::random(pos * PHI)`.
/// 4. Majority-bundle all word-position vectors → 512-bit sentence fingerprint.
/// 5. Expand 8 × u64 (512-bit) → 256 × u64 (16 Kbit) by tiling: each source
///    u64 occupies a 32-word run in the content plane.
/// 6. Write the content row into BindSpace at write_cursor, advance cursor.
/// 7. Return hex fingerprint + token_count + bits_set + row_written.
///
/// Why hash-based ranks instead of Vocabulary::load?
/// The vocabulary requires CSV data files on disk; the encode endpoint is
/// intended to be stateless and zero-I/O. `VsaVec::from_rank` is pure and
/// deterministic — hashing word strings to u16 rank seeds gives the same
/// VSA vectors on every call without loading any external table. When the
/// data files are available, upgrade to Vocabulary::load + parser::parse for
/// full SPO triple extraction.
async fn encode_handler(
    State(state): State<AppState>,
    Json(req): Json<WireEncode>,
) -> Result<Json<WireEncodeResponse>, (StatusCode, Json<Value>)> {
    use deepnsm::encoder::{bundle, VsaVec, VSA_WORDS};

    // ── 1. Word tokenisation (zero-I/O, no CSV needed) ───────────────────
    let words: Vec<&str> = req
        .text
        .split(|c: char| c.is_whitespace() || (c.is_ascii_punctuation() && c != '\''))
        .filter(|s| !s.is_empty())
        .collect();
    let token_count = words.len();

    // ── 2 + 3. Hash word → rank, XOR-bind with position vector ───────────
    //
    // Rank derivation: FNV-1a-style fold into 12 bits.
    //   hash = words[i].bytes().fold(2166136261u32, |h, b| {
    //       (h ^ b as u32).wrapping_mul(16777619)
    //   }) & 0x0FFF
    //
    // Position braid: XOR with VsaVec::random(pos * PHI) so
    // "dog bites man" ≠ "man bites dog".
    const PHI: u64 = 0x9E3779B97F4A7C15; // golden-ratio multiplier

    let word_vecs: Vec<VsaVec> = words
        .iter()
        .enumerate()
        .map(|(pos, word)| {
            // FNV-1a → 12-bit rank
            let hash = word
                .bytes()
                .fold(2166136261u32, |h, b| (h ^ b as u32).wrapping_mul(16777619));
            let rank = (hash & 0x0FFF) as u16;

            // Position seed: unique per (pos, golden-ratio)
            let pos_seed = (pos as u64).wrapping_mul(PHI);
            let pos_vec = VsaVec::random(pos_seed);

            // word_fp = from_rank(rank) XOR pos_vec
            VsaVec::from_rank(rank).bind(&pos_vec)
        })
        .collect();

    // ── 4. Bundle → 512-bit sentence fingerprint ─────────────────────────
    let sentence_vec = if word_vecs.is_empty() {
        VsaVec::ZERO
    } else {
        bundle(&word_vecs)
    };

    // ── 4b. Build fingerprint hex and popcount ────────────────────────────
    let vsa_words = sentence_vec.as_words(); // &[u64; VSA_WORDS] (VSA_WORDS = 8)
    let fingerprint_hex: String = vsa_words
        .iter()
        .map(|w| format!("{:016x}", w))
        .collect();
    let bits_set = sentence_vec.popcount() as usize;

    // ── 5. Expand 8 × u64 → 256 × u64 (16 Kbit) ─────────────────────────
    //
    // Tiling strategy: content_fp[i] = vsa_words[i / TILE_FACTOR]
    // TILE_FACTOR = CONTENT_WORDS / VSA_WORDS = 256 / 8 = 32.
    // Every source u64 occupies 32 consecutive words in the content plane.
    // This preserves all 512 VSA bits at stable positions; the dispatch
    // sweep correlates against them via Hamming distance.
    const CONTENT_WORDS: usize = 256; // WORDS_PER_FP in bindspace.rs
    const TILE_FACTOR: usize = CONTENT_WORDS / VSA_WORDS; // = 32
    let mut content_fp = [0u64; CONTENT_WORDS];
    for (i, w) in content_fp.iter_mut().enumerate() {
        *w = vsa_words[i / TILE_FACTOR];
    }

    // ── 6. Write to BindSpace, advance write_cursor ───────────────────────
    let row_written = {
        let mut st = state.lock().map_err(|_| {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "lock poisoned"})))
        })?;
        let cursor = st.write_cursor;
        if cursor >= st.driver.bindspace.len {
            None
        } else {
            let bs = Arc::get_mut(&mut st.driver.bindspace).ok_or_else(|| {
                (StatusCode::CONFLICT, Json(json!({"error": "bindspace has multiple references"})))
            })?;
            bs.fingerprints.set_content(cursor, &content_fp);
            st.write_cursor = cursor + 1;
            Some(cursor as u32)
        }
    };

    Ok(Json(WireEncodeResponse {
        text: req.text,
        token_count,
        fingerprint_hex,
        bits_set,
        row_written,
    }))
}

/// Runbook-step dispatcher for Plan. Maps the shared planner state +
/// request into a runbook step result, yielding an error string on the
/// with-planner=off build to flow through the runbook's error channel.
fn plan_runbook_step(
    state: &AppState,
    req: &WirePlanRequest,
    label: &str,
) -> Result<WireRunbookStepResult, String> {
    match run_plan(state, req) {
        Ok(response) => Ok(WireRunbookStepResult::Plan {
            label: label.to_string(),
            response,
        }),
        Err((_code, msg)) => Err(msg),
    }
}

async fn runbook_handler(
    State(state): State<AppState>,
    Json(req): Json<WireRunbookRequest>,
) -> Result<Json<WireRunbookResponse>, (StatusCode, Json<Value>)> {
    let total = req.steps.len();
    let t0 = std::time::Instant::now();
    let mut results: Vec<WireRunbookStepResult> = Vec::with_capacity(total);
    let mut errors = 0usize;
    let mut completed = 0usize;

    for s in req.steps.into_iter() {
        let label = s.label.clone();
        let step_name = match &s.step {
            WireRunbookStep::Tensors(_) => "tensors",
            WireRunbookStep::Calibrate(_) => "calibrate",
            WireRunbookStep::Probe(_) => "probe",
            WireRunbookStep::Dispatch(_) => "dispatch",
            WireRunbookStep::Ingest(_) => "ingest",
            WireRunbookStep::Plan(_) => "plan",
        };
        let outcome: Result<WireRunbookStepResult, String> = match s.step {
            WireRunbookStep::Tensors(r) => codec_research::list_tensors(&r)
                .map(|response| WireRunbookStepResult::Tensors { label: label.clone(), response }),
            WireRunbookStep::Calibrate(r) => codec_research::calibrate_tensor(&r)
                .map(|response| WireRunbookStepResult::Calibrate { label: label.clone(), response }),
            WireRunbookStep::Probe(r) => codec_research::row_count_probe(&r)
                .map(|response| WireRunbookStepResult::Probe { label: label.clone(), response }),
            WireRunbookStep::Dispatch(wd) => match state.lock() {
                Err(_) => Err("lock poisoned".to_string()),
                Ok(st) => {
                    let crystal = st.driver.dispatch(&wd.to_internal());
                    Ok(WireRunbookStepResult::Dispatch {
                        label: label.clone(),
                        response: WireCrystal::from(&crystal),
                    })
                }
            },
            WireRunbookStep::Ingest(wi) => match state.lock() {
                Err(_) => Err("lock poisoned".to_string()),
                Ok(mut st) => {
                    let cursor = st.write_cursor;
                    match Arc::get_mut(&mut st.driver.bindspace) {
                        None => Err("bindspace has multiple references".to_string()),
                        Some(bs) => {
                            let (start, end) = engine_bridge::ingest_codebook_indices(
                                bs, &wi.codebook_indices, wi.source_ordinal, wi.timestamp, cursor,
                            );
                            st.write_cursor = end as usize;
                            Ok(WireRunbookStepResult::Ingest {
                                label: label.clone(),
                                ingested: end - start,
                                row_start: start,
                                row_end: end,
                                write_cursor: st.write_cursor as u32,
                            })
                        }
                    }
                }
            },
            WireRunbookStep::Plan(wp) => plan_runbook_step(&state, &wp, &label),
        };

        match outcome {
            Ok(r) => {
                completed += 1;
                results.push(r);
            }
            Err(e) => {
                errors += 1;
                results.push(WireRunbookStepResult::Error {
                    label,
                    step: step_name.to_string(),
                    error: e,
                });
                if req.stop_on_error {
                    break;
                }
            }
        }
    }

    Ok(Json(WireRunbookResponse {
        label: req.label,
        total_steps: total,
        completed,
        errors,
        total_elapsed_ms: t0.elapsed().as_millis() as u64,
        results,
    }))
}
