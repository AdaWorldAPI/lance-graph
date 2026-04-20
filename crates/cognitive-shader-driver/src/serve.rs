//! Axum REST server — external API for live testing cognitive shader cycles.
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
use crate::wire::{
    WireCalibrateRequest, WireCalibrateResponse, WireCrystal, WireDispatch, WireHealth,
    WireIngest, WireProbeRequest, WireProbeResponse, WireQualia, WireRunbookRequest,
    WireRunbookResponse, WireRunbookStep, WireRunbookStepResult, WireStyleInfo,
    WireTensorsRequest, WireTensorsResponse,
};
use lance_graph_contract::cognitive_shader::CognitiveShaderDriver;

struct ServerState {
    driver: ShaderDriver,
    write_cursor: usize,
}

type AppState = Arc<Mutex<ServerState>>;

pub fn router(driver: ShaderDriver) -> Router {
    let state: AppState = Arc::new(Mutex::new(ServerState {
        driver,
        write_cursor: 0,
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
        // Scheduled runbook: one POST runs a list of steps. Test injection
        // lands here — a client script submits its full codec-research
        // protocol as a single DTO, the server executes and returns all
        // results correlated by per-step label.
        .route("/v1/shader/runbook", post(runbook_handler))
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
