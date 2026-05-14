//! `ConsumerEnvelope` + `ConsumerReply` — typed payload crossing the actor membrane.
//!
//! These are the gRPC-shaped request/response pairs stripped of the tonic wrapper.
//! They are NOT `UnifiedStep` / `UnifiedAuditEvent` — those are internal substrate.
//! The Blood-Brain Barrier (BBB) invariant from callcenter-membrane-v1.md §3 applies:
//! only these envelope types cross the actor mailbox boundary.
//!
//! Spec: pr-g2-ractor-supervisor.md §3.2
//! CC-2 fix: `ConsumerEnvelope` / `ConsumerReply` do NOT contain any `AuthOp` lifecycle
//!           variants — those live in `LifecycleAuditEvent` in `lifecycle_audit.rs`.

// ─── Request envelopes ────────────────────────────────────────────────────────

/// Typed request payload crossing from caller → consumer actor.
///
/// No `Box<dyn>` — closed enum, fixed at compile time over the gRPC-shaped arms.
/// Lab-only arms (`Tensors`, `Calibrate`, `Probe`) are always-present but
/// documented as lab-only via doc comments (see spec §13 Open Q 3).
#[derive(Clone, Debug)]
pub enum ConsumerEnvelope {
    Dispatch(DispatchRequest),
    Ingest(IngestRequest),
    Health,
    Qualia(QualiaRequest),
    Styles(StylesRequest),
    // Lab-only arms — present in all builds; gate use behind `cfg(feature = "lab")`
    // in the consumer actor handlers if you want lab isolation.
    /// Lab-only: raw tensor passthrough. Not for production routing.
    Tensors(TensorsRequest),
    /// Lab-only: calibration request. Not for production routing.
    Calibrate(CalibrateRequest),
    /// Lab-only: probe / introspection request. Not for production routing.
    Probe(ProbeRequest),
}

// ─── Reply envelopes ─────────────────────────────────────────────────────────

/// Typed reply payload crossing from consumer actor → caller.
#[derive(Clone, Debug)]
pub enum ConsumerReply {
    Crystal(CrystalResponse),
    Ingest(IngestAck),
    Health(HealthStatus),
    Qualia(Qualia17DResponse),
    Styles(StyleList),
    // Lab-only:
    Tensors(TensorsResponse),
    Calibrate(CalibrateResponse),
    Probe(ProbeResponse),
}

// ─── Payload types ────────────────────────────────────────────────────────────
// Minimal owned shapes for the crossing payload. Production implementations
// replace these with the actual gRPC proto-generated types (stripped of tonic).

#[derive(Clone, Debug)]
pub struct DispatchRequest {
    pub tenant_id: u32,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct IngestRequest {
    pub tenant_id: u32,
    pub records: Vec<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct QualiaRequest {
    pub tenant_id: u32,
    pub qualia_key: String,
}

#[derive(Clone, Debug)]
pub struct StylesRequest {
    pub tenant_id: u32,
    pub style_ids: Vec<u32>,
}

/// Lab-only: raw tensor payload.
#[derive(Clone, Debug)]
pub struct TensorsRequest {
    pub data: Vec<f32>,
}

/// Lab-only: calibration parameters.
#[derive(Clone, Debug)]
pub struct CalibrateRequest {
    pub params: Vec<u8>,
}

/// Lab-only: probe / introspection request.
#[derive(Clone, Debug)]
pub struct ProbeRequest {
    pub probe_id: u32,
}

// ─── Response types ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CrystalResponse {
    pub tenant_id: u32,
    pub result: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct IngestAck {
    pub accepted: u32,
    pub rejected: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HealthStatus {
    pub ok: bool,
    pub detail: String,
}

impl HealthStatus {
    pub fn healthy() -> Self {
        Self {
            ok: true,
            detail: "ok".to_string(),
        }
    }

    pub fn unhealthy(detail: impl Into<String>) -> Self {
        Self {
            ok: false,
            detail: detail.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Qualia17DResponse {
    pub values: [f32; 17],
}

#[derive(Clone, Debug)]
pub struct StyleList {
    pub style_ids: Vec<u32>,
}

/// Lab-only: tensor response.
#[derive(Clone, Debug)]
pub struct TensorsResponse {
    pub data: Vec<f32>,
}

/// Lab-only: calibration result.
#[derive(Clone, Debug)]
pub struct CalibrateResponse {
    pub ok: bool,
}

/// Lab-only: probe response.
#[derive(Clone, Debug)]
pub struct ProbeResponse {
    pub payload: Vec<u8>,
}
