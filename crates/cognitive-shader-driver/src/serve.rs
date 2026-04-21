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
    WireCalibrateRequest, WireCalibrateResponse, WireCrystal, WireDispatch, WireHealth,
    WireIngest, WirePlanRequest, WirePlanResponse, WireProbeRequest, WireProbeResponse,
    WireQualia, WireRunbookRequest, WireRunbookResponse, WireRunbookStep,
    WireRunbookStepResult, WireStepResult, WireStyleInfo, WireTensorsRequest,
    WireTensorsResponse, WireTokenAgreement, WireTokenAgreementResult, WireUnifiedStep,
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

async fn route_handler(
    State(_state): State<AppState>,
    Json(wire): Json<WireUnifiedStep>,
) -> Result<Json<WireStepResult>, (StatusCode, Json<Value>)> {
    use lance_graph_contract::orchestration::{
        OrchestrationBridge, StepStatus, UnifiedStep,
    };

    let mut step = UnifiedStep {
        step_id: wire.step_id.clone(),
        step_type: wire.step_type.clone(),
        status: StepStatus::Pending,
        thinking: None,
        reasoning: wire.reasoning,
        confidence: None,
    };

    // Try codec research bridge first (nd.*)
    let codec_bridge = crate::codec_bridge::CodecResearchBridge;
    let result = codec_bridge.route(&mut step);

    // If codec bridge rejected with DomainUnavailable, try planner bridge (lg.*)
    if matches!(result, Err(lance_graph_contract::orchestration::OrchestrationError::DomainUnavailable(_))) {
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
