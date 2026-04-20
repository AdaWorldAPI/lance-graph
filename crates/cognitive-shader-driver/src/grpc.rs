//! gRPC service — protobuf interface for live shader testing.
//!
//! Behind `--features grpc`. Uses tonic + prost. Same quarantine as REST:
//! debug-only, never in production binary.
//!
//! ```bash
//! # Start gRPC server:
//! cargo run -p cognitive-shader-driver --features grpc --bin shader-grpc
//!
//! # Test with grpcurl:
//! grpcurl -plaintext -d '{"row_start":0,"row_end":100}' \
//!   localhost:50051 cognitive_shader.CognitiveShaderService/Dispatch
//! ```

use std::sync::{Arc, Mutex};
use tonic::{Request, Response, Status};

use crate::driver::ShaderDriver;
use crate::engine_bridge::{self, unified_style, UNIFIED_STYLES};
use lance_graph_contract::cognitive_shader::{
    ColumnWindow, CognitiveShaderDriver as DriverTrait, EmitMode, MetaFilter,
    RungLevel, ShaderDispatch, StyleSelector,
};

pub mod pb {
    tonic::include_proto!("cognitive_shader");
}

use pb::cognitive_shader_service_server::{CognitiveShaderService, CognitiveShaderServiceServer};

pub struct ShaderGrpcService {
    driver: Arc<Mutex<ShaderDriver>>,
    write_cursor: Arc<Mutex<usize>>,
}

impl ShaderGrpcService {
    pub fn new(driver: ShaderDriver) -> Self {
        Self {
            driver: Arc::new(Mutex::new(driver)),
            write_cursor: Arc::new(Mutex::new(0)),
        }
    }

    pub fn into_server(self) -> CognitiveShaderServiceServer<Self> {
        CognitiveShaderServiceServer::new(self)
    }
}

#[tonic::async_trait]
impl CognitiveShaderService for ShaderGrpcService {
    async fn dispatch(
        &self,
        request: Request<pb::DispatchRequest>,
    ) -> Result<Response<pb::CrystalResponse>, Status> {
        let req = request.into_inner();
        let internal = proto_to_dispatch(&req);
        let drv = self.driver.lock().map_err(|_| Status::internal("lock poisoned"))?;
        let crystal = drv.dispatch(&internal);

        // Convert cycle_fingerprint [u64; 256] → bytes (2048)
        let fp_bytes: Vec<u8> = crystal.bus.cycle_fingerprint.iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();

        let gate = if crystal.bus.gate.is_flow() { 0 }
            else if crystal.bus.gate.is_block() { 1 }
            else { 2 };

        let hits: Vec<pb::HitMessage> = crystal.bus.resonance.top_k.iter()
            .filter(|h| h.resonance > 0.0)
            .map(|h| pb::HitMessage {
                row: h.row,
                distance: h.distance as u32,
                predicates: h.predicates as u32,
                resonance: h.resonance,
                cycle_index: h.cycle_index,
            })
            .collect();

        Ok(Response::new(pb::CrystalResponse {
            bus: Some(pb::BusMessage {
                cycle_fingerprint: fp_bytes,
                emitted_edges: crystal.bus.emitted_edges[..crystal.bus.emitted_edge_count as usize].to_vec(),
                gate,
                resonance: Some(pb::ResonanceMessage {
                    top_k: hits,
                    hit_count: crystal.bus.resonance.hit_count as u32,
                    cycles_used: crystal.bus.resonance.cycles_used as u32,
                    entropy: crystal.bus.resonance.entropy,
                    std_dev: crystal.bus.resonance.std_dev,
                    style_ord: crystal.bus.resonance.style_ord as u32,
                }),
            }),
            persisted_row: crystal.persisted_row,
            meta: Some(pb::MetaSummary {
                confidence: crystal.meta.confidence,
                meta_confidence: crystal.meta.meta_confidence,
                brier: crystal.meta.brier,
                should_admit_ignorance: crystal.meta.should_admit_ignorance,
            }),
        }))
    }

    async fn ingest(
        &self,
        request: Request<pb::IngestRequest>,
    ) -> Result<Response<pb::IngestResponse>, Status> {
        let req = request.into_inner();
        let indices: Vec<u16> = req.codebook_indices.iter().map(|&v| v as u16).collect();

        let mut cursor = self.write_cursor.lock().map_err(|_| Status::internal("lock"))?;
        let mut drv = self.driver.lock().map_err(|_| Status::internal("lock"))?;
        let bs = Arc::get_mut(&mut drv.bindspace)
            .ok_or_else(|| Status::failed_precondition("bindspace has multiple refs"))?;

        let c = *cursor;
        let (start, end) = engine_bridge::ingest_codebook_indices(
            bs, &indices, req.source_ordinal as u8, req.timestamp, c,
        );
        *cursor = end as usize;

        Ok(Response::new(pb::IngestResponse {
            ingested: end - start,
            row_start: start,
            row_end: end,
            write_cursor: *cursor as u32,
        }))
    }

    async fn health(
        &self,
        _request: Request<pb::HealthRequest>,
    ) -> Result<Response<pb::HealthResponse>, Status> {
        let drv = self.driver.lock().map_err(|_| Status::internal("lock"))?;
        Ok(Response::new(pb::HealthResponse {
            row_count: drv.row_count(),
            byte_footprint: drv.byte_footprint() as u64,
            styles: unified_styles_proto(),
            neural_debug: None,
        }))
    }

    async fn qualia(
        &self,
        request: Request<pb::QualiaRequest>,
    ) -> Result<Response<pb::QualiaResponse>, Status> {
        let row = request.into_inner().row;
        let drv = self.driver.lock().map_err(|_| Status::internal("lock"))?;
        let bs = drv.bindspace();
        if row as usize >= bs.len {
            return Err(Status::not_found("row out of range"));
        }
        let (experienced, cd) = engine_bridge::read_qualia_decomposed(bs, row as usize);
        let style_ord = crate::auto_style::style_from_qualia(&experienced);

        Ok(Response::new(pb::QualiaResponse {
            row,
            experienced: experienced.to_vec(),
            classification_distance: cd,
            style_name: unified_style(style_ord).name.to_string(),
        }))
    }

    async fn styles(
        &self,
        _request: Request<pb::StylesRequest>,
    ) -> Result<Response<pb::StylesResponse>, Status> {
        Ok(Response::new(pb::StylesResponse {
            styles: unified_styles_proto(),
        }))
    }

    async fn tensors(
        &self,
        request: Request<pb::TensorsRequest>,
    ) -> Result<Response<pb::TensorsResponse>, Status> {
        let req = request.into_inner();
        let wire_req = crate::wire::WireTensorsRequest {
            model_path: req.model_path,
            route_filter: if req.route_filter.is_empty() { None } else { Some(req.route_filter) },
        };
        let r = crate::codec_research::list_tensors(&wire_req)
            .map_err(|e| Status::invalid_argument(e))?;
        Ok(Response::new(pb::TensorsResponse {
            total: r.total as u32,
            shown: r.shown as u32,
            cam_pq: r.cam_pq as u32,
            passthrough: r.passthrough as u32,
            skip: r.skip as u32,
            tensors: r.tensors.iter().map(|t| pb::TensorEntry {
                name: t.name.clone(),
                dims: t.dims.clone(),
                dtype: t.dtype.clone(),
                route: t.route.clone(),
                n_elements: t.n_elements,
            }).collect(),
        }))
    }

    async fn calibrate(
        &self,
        request: Request<pb::CalibrateRequest>,
    ) -> Result<Response<pb::CalibrateResponse>, Status> {
        let req = request.into_inner();
        let wire_req = crate::wire::WireCalibrateRequest {
            model_path: req.model_path,
            tensor_name: req.tensor_name,
            num_subspaces: if req.num_subspaces == 0 { 6 } else { req.num_subspaces as usize },
            num_centroids: if req.num_centroids == 0 { 256 } else { req.num_centroids as usize },
            kmeans_iterations: if req.kmeans_iterations == 0 { 20 } else { req.kmeans_iterations as usize },
            max_rows: if req.max_rows == 0 { None } else { Some(req.max_rows as usize) },
            icc_samples: if req.icc_samples == 0 { 512 } else { req.icc_samples as usize },
        };
        let r = crate::codec_research::calibrate_tensor(&wire_req)
            .map_err(|e| Status::invalid_argument(e))?;
        Ok(Response::new(pb::CalibrateResponse {
            tensor_name: r.tensor_name,
            dims: r.dims,
            n_rows: r.n_rows as u32,
            row_dim: r.row_dim as u32,
            adjusted_dim: r.adjusted_dim as u32,
            num_subspaces: r.num_subspaces as u32,
            num_centroids: r.num_centroids as u32,
            calibration_rows: r.calibration_rows as u32,
            icc_3_1: r.icc_3_1,
            mean_reconstruction_error: r.mean_reconstruction_error,
            relative_l2_error: r.relative_l2_error,
            codebook_bytes: r.codebook_bytes as u64,
            fingerprints_bytes: r.fingerprints_bytes as u64,
            elapsed_ms: r.elapsed_ms,
        }))
    }

    async fn probe(
        &self,
        request: Request<pb::ProbeRequest>,
    ) -> Result<Response<pb::ProbeResponse>, Status> {
        let req = request.into_inner();
        let wire_req = crate::wire::WireProbeRequest {
            model_path: req.model_path,
            tensor_name: req.tensor_name,
            row_counts: req.row_counts.iter().map(|&n| n as usize).collect(),
            icc_samples: if req.icc_samples == 0 { 512 } else { req.icc_samples as usize },
        };
        let r = crate::codec_research::row_count_probe(&wire_req)
            .map_err(|e| Status::invalid_argument(e))?;
        Ok(Response::new(pb::ProbeResponse {
            tensor_name: r.tensor_name,
            n_rows: r.n_rows as u32,
            row_dim: r.row_dim as u32,
            adjusted_dim: r.adjusted_dim as u32,
            num_subspaces: r.num_subspaces as u32,
            num_centroids: r.num_centroids as u32,
            entries: r.entries.iter().map(|e| pb::ProbeEntry {
                n_train: e.n_train as u32,
                icc_train: e.icc_train,
                icc_all_rows: e.icc_all_rows,
                relative_l2_error: e.relative_l2_error,
                elapsed_ms: e.elapsed_ms,
            }).collect(),
        }))
    }
}

fn proto_to_dispatch(req: &pb::DispatchRequest) -> ShaderDispatch {
    let style = req.style.as_ref().map(|s| {
        match &s.selector {
            Some(pb::style_selector::Selector::Auto(_)) => StyleSelector::Auto,
            Some(pb::style_selector::Selector::Ordinal(n)) => StyleSelector::Ordinal(*n as u8),
            Some(pb::style_selector::Selector::Named(name)) => {
                StyleSelector::Ordinal(crate::auto_style::resolve(
                    StyleSelector::Named(Box::leak(name.to_lowercase().into_boxed_str())),
                    &[0.0; 18],
                ))
            }
            None => StyleSelector::Auto,
        }
    }).unwrap_or(StyleSelector::Auto);

    let rung = match req.rung {
        0 => RungLevel::Surface,
        1 => RungLevel::Shallow,
        2 => RungLevel::Contextual,
        3 => RungLevel::Analogical,
        4 => RungLevel::Abstract,
        5 => RungLevel::Structural,
        6 => RungLevel::Counterfactual,
        7 => RungLevel::Meta,
        8 => RungLevel::Recursive,
        _ => RungLevel::Transcendent,
    };

    let emit = match pb::EmitMode::try_from(req.emit) {
        Ok(pb::EmitMode::Bundle) => EmitMode::Bundle,
        Ok(pb::EmitMode::Persist) => EmitMode::Persist,
        _ => EmitMode::Cycle,
    };

    let meta = req.meta_filter.as_ref().map(|f| MetaFilter {
        thinking_mask: f.thinking_mask,
        awareness_min: f.awareness_min as u8,
        nars_f_min: f.nars_f_min as u8,
        nars_c_min: f.nars_c_min as u8,
        free_e_max: f.free_e_max as u8,
    }).unwrap_or(MetaFilter::ALL);

    ShaderDispatch {
        meta_prefilter: meta,
        rows: ColumnWindow::new(req.row_start, req.row_end),
        layer_mask: req.layer_mask as u8,
        radius: req.radius as u16,
        style,
        rung,
        max_cycles: req.max_cycles as u16,
        entropy_floor: req.entropy_floor,
        emit,
    }
}

fn unified_styles_proto() -> Vec<pb::StyleInfo> {
    UNIFIED_STYLES.iter().map(|s| pb::StyleInfo {
        ordinal: s.ordinal as u32,
        name: s.name.to_string(),
        layer_mask: s.layer_mask as u32,
        density_target: s.density_target,
        resonance_threshold: s.resonance_threshold,
        fan_out: s.fan_out as u32,
        combine: s.combine as u32,
        contra: s.contra as u32,
        exploration: s.exploration,
        speed: s.speed,
        collapse_bias: s.collapse_bias,
        butterfly_sensitivity: s.butterfly_sensitivity,
    }).collect()
}
