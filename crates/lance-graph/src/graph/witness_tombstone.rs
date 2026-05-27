// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Memory lifecycle for the AriGraph hot→cold→tombstone pipeline.
//!
//! **D-ATOM-5 scaffold only — `todo!()` bodies, no implementation.**
//!
//! ## Architecture (E-LADDER-SERVES-MAILBOX §6, 2026-05-27)
//!
//! AriGraph is **not** a persisted singleton (E-BATON-1). Its lifecycle is:
//!
//! ```text
//! hot (full-fidelity ephemeral AriGraph, inside mailbox)
//!  → calcified semantic  (SPO-G quads, cold, Lance)            — "what is believed"
//!  + tombstone witness   (Lance versioned, compressed ~Scent/Base17) — "what happened / who committed it"
//!  + counterfactual residue (CausalEdge64 −6 mantissa, 4 bits) — "the road not taken"
//! ```
//!
//! Because Lance is **append-only / versioned**, the tombstone layer *is* the
//! audit trail — GoBD/provenance falls out of the substrate by construction
//! (E-FIBU-GOBD-BY-CONSTRUCTION), not as bolted-on logging.
//! Cross-ref: E-LADDER-SERVES-MAILBOX §6, §6b.
//!
//! ## Link integrity invariant
//!
//! The calcified SPO fact holds a durable [`WitnessLink`] back-pointer to its
//! [`Tombstone`].  The tombstone **must outlive the mailbox** — Lance versioning
//! is the home for both.  Violating this invariant breaks the GoBD audit chain.
//!
//! ## Module placement
//!
//! This module lives in `crates/lance-graph` (the CORE crate), NOT in
//! `lance-graph-contract`, because it depends on Lance persistence and the SPO
//! store in `graph/spo/`.
//!
//! ## BLOCKED items (do NOT guess — see inline markers)
//!
//! - The exact SPO quad type name + constructor in `graph/spo/`.
//! - The Lance versioned-store write API for appending a tombstone row.
//! - The [`WitnessCorpus`] D-id/type path (found in
//!   `graph/arigraph/witness_corpus.rs` — wire-up is BLOCKED until the
//!   tombstone Arrow schema and the `WitnessCorpus` ingestion API are resolved).
//! - The Scent/Base17 compression entry point for the tombstone payload.

// ── HotWitness ────────────────────────────────────────────────────────────────

/// The ephemeral, in-mailbox episodic working record of a single AriGraph fact
/// before it stabilises and calcifies into the cold SPO ontology.
///
/// # NOT a persisted singleton (E-BATON-1)
///
/// `HotWitness` exists **only inside the owning mailbox** for the duration of
/// its sea-star lifecycle (spawn → work → die → merge).  It is **never**
/// transmitted across mailbox boundaries as a carrier — the Baton
/// (`CollapseGateEmission`) is the inter-mailbox handoff; `HotWitness` stays
/// local and ephemeral.  See `EPIPHANIES.md` E-BATON-1 and the Baton-scoping
/// section of `CLAUDE.md` §"The Click" for the canonical rationale.
///
/// When the mailbox dies the [`HotWitness`] is either:
/// - **calcified** (fact stabilised) → [`calcify`] produces an SPO cold record
///   and [`Tombstone::from_hot`] archives provenance in Lance; OR
/// - **discarded** (fact never reached quorum) — no tombstone is written.
///
/// # Fields
///
/// All fields use primitive/contract types so this struct has no Lance/Arrow
/// dependency and can live in the hot synchronous path (E-BATON-1: inner Click
/// stays sync; ractor async only at the swarm boundary).
#[derive(Debug, Clone)]
pub struct HotWitness {
    /// Identifier of the originating mailbox (sea-star node).
    ///
    /// Used as the provenance key in the [`Tombstone`] and the link key in
    /// [`WitnessLink`].  Type is `u64` until the mailbox identity type is
    /// resolved — BLOCKED: exact mailbox-id type from ractor/supervisor surface.
    pub mailbox_id: u64,

    /// The subject fingerprint key (dn_hash) of the fact being observed.
    pub subject_key: u64,

    /// The predicate fingerprint key of the fact being observed.
    pub predicate_key: u64,

    /// The object fingerprint key of the fact being observed.
    pub object_key: u64,

    /// NARS truth value: (frequency, confidence) for the observation.
    ///
    /// This is the per-axis quorum output — `(I4 position, quorum-confidence)`
    /// as described in E-LADDER-SERVES-MAILBOX §3.  Encoded as two `f32` here
    /// until `contract::atoms` `I4x32` is resolved (D-ATOM-1, BLOCKED on
    /// D-ATOM-0 basis decision).
    pub truth_frequency: f32,
    pub truth_confidence: f32,

    /// Nanosecond timestamp of the observation (UNIX epoch).
    pub observed_at_ns: u64,

    /// Optional CausalEdge64 counterfactual mantissa retained from a split
    /// quorum (§5: Counterfactual −6 nibble, 4 bits, road-not-taken).
    ///
    /// `None` if the quorum was uncontested; `Some(packed_edge)` if
    /// [`calcify`] was called after an `is_split` verdict.
    // BLOCKED: CausalEdge64 type import path — lives in the `causal-edge`
    // crate which may not be a direct dep of `lance-graph` core.  Use u64
    // placeholder until dep graph is confirmed.
    pub counterfactual_mantissa: Option<u64>,
}

// ── calcify ───────────────────────────────────────────────────────────────────

/// Harden a stabilised hot fact into the cold SPO ontology.
///
/// Takes a `&HotWitness` that has reached quorum (i.e. `truth_confidence` ≥
/// the commit threshold, `FreeEnergy < 0.2` per The Click) and produces the
/// **SPO quad** — the cold, persistent, permanently-believed record in the SPO
/// triple store.
///
/// # Codec placement (E-LADDER-SERVES-MAILBOX §6 compression hierarchy)
///
/// Calcification is one step *down* the codec atlas:
///
/// ```text
/// hot AriGraph fact (full-fidelity)
///   → cold SpoRecord (SPO-G quad, persistent, Lance)
/// ```
///
/// The returned value should be inserted into an [`SpoStore`] (or the Lance-
/// backed production SPO dataset) by the caller.  This function does NOT write
/// to Lance directly — it produces the record; the caller controls persistence.
///
/// # Counterfactual residue
///
/// If `hot.counterfactual_mantissa` is `Some`, the caller MUST also deposit
/// the CausalEdge64 −6 nibble into the episodic witness chain (§5 of the
/// epiphany) to preserve the road-not-taken.  This function does not do that
/// deposit — split concerns.
///
/// # Returns
///
/// // BLOCKED: exact return type.  The SPO cold record type is `SpoRecord`
/// // in `graph/spo/builder.rs` (pub struct SpoRecord { subject, predicate,
/// // object: Fingerprint; packed: Bitmap; truth: TruthValue }).
/// // The constructor is `SpoBuilder::build_edge` (stateless happy-path mode).
/// // BLOCKED: confirming the full constructor signature + TruthValue ctor.
/// // Using opaque `SpoRecord` placeholder; replace with direct import once
/// // confirmed.
pub fn calcify(_hot: &HotWitness) -> crate::graph::spo::builder::SpoRecord {
    // BLOCKED: SpoBuilder::build_edge constructor signature.
    // BLOCKED: TruthValue constructor (truth.rs — confirm field names).
    // BLOCKED: Fingerprint reconstruction from (subject_key, predicate_key,
    //   object_key) u64 hashes — need to confirm whether SpoRecord stores
    //   the full Fingerprint or just the hash key.
    todo!("D-ATOM-5: calcify HotWitness → SpoRecord (SPO cold quad)")
}

// ── Tombstone ─────────────────────────────────────────────────────────────────

/// The cold episodic provenance record left in Lance when an ephemeral mailbox
/// dies after committing a fact (sea-star spawn→die→merge lifecycle).
///
/// # Purpose: GoBD audit trail by construction (E-FIBU-GOBD-BY-CONSTRUCTION)
///
/// Because Lance is **append-only and versioned**, writing a `Tombstone` to a
/// Lance dataset IS the audit entry — no separate logging system is required.
/// GoBD compliance (append-only, tamper-evident, versioned) falls out of the
/// substrate by construction.  Cross-ref: E-LADDER-SERVES-MAILBOX §6 fallout +
/// §6b business atoms.
///
/// # Compression (~Scent/Base17 level)
///
/// The tombstone payload is compressed to approximately the Scent or Base17
/// level of the codec atlas (34 bytes for Base17, 1 byte for Scent — see
/// `docs/CODEC_COMPRESSION_ATLAS.md`).  This keeps the versioned Lance
/// tombstone dataset lean for long-running audit trails.
///
/// # Lifecycle
///
/// A `Tombstone` is created via [`Tombstone::from_hot`] at mailbox-death time.
/// It is persisted to a dedicated Lance dataset (the "tombstone store") via
/// [`Tombstone::persist`].  It is NOT deleted when the mailbox exits — it
/// outlives the mailbox by design (Lance versioning guarantees).
///
/// # Relationship to [`WitnessCorpus`]
///
/// BLOCKED: whether the tombstone is ingested INTO the existing
/// `graph/arigraph/witness_corpus::WitnessCorpus` (D-CSV-6 / sprint-12) or
/// written to a separate Lance dataset dedicated to tombstones.  The
/// `WitnessCorpus` currently holds `WitnessEntry { spo: u64, timestamp_ns,
/// source_url, evidence_blob }` — a tombstone is semantically different
/// (provenance-of-death vs provenance-of-observation).  Confirm before wiring.
#[derive(Debug, Clone)]
pub struct Tombstone {
    /// The mailbox that committed this fact and then died.
    ///
    /// Links the tombstone back to the originating [`HotWitness::mailbox_id`].
    pub mailbox_id: u64,

    /// The u64 key (dn_hash) of the calcified SPO fact this tombstone covers.
    ///
    /// This is the primary key used by [`WitnessLink`] to join the cold SPO
    /// record to its provenance.
    pub spo_key: u64,

    /// Lance version number at which this tombstone was written.
    ///
    /// Enables time-travel queries: `VersionedGraph::at_version(lance_version)`
    /// retrieves the snapshot at the moment of calcification.
    pub lance_version: u64,

    /// Nanosecond timestamp of the mailbox death (UNIX epoch).
    pub committed_at_ns: u64,

    /// Compressed tombstone payload (~Scent/Base17 level).
    ///
    /// Encodes the minimal provenance: which mailbox, which fact, what
    /// truth value, counterfactual residue presence flag.
    ///
    /// BLOCKED: exact compression entry point.  The codec atlas places Scent
    /// at 1 byte (ρ=0.937) and Base17 at 34 bytes (ρ=0.965).  The candidate
    /// entry points are:
    ///   - `ndarray::hpc::bgz17_bridge::Base17` (34-byte VSA, used in `neuron.rs`)
    ///   - Scent codec (1-byte) from `crates/bgz17` — confirm the public API
    ///     entry point (crate is in `exclude` in the workspace Cargo.toml;
    ///     direct dep may require adding it to `lance-graph`'s Cargo.toml).
    /// Using raw `Box<[u8]>` until resolved.
    pub compressed_payload: Box<[u8]>,

    /// Whether the originating `HotWitness` carried a counterfactual mantissa
    /// (CausalEdge64 −6 nibble, §5).  Presence flag only — the full mantissa
    /// is in the episodic witness chain, not duplicated here.
    pub has_counterfactual: bool,
}

impl Tombstone {
    /// Construct a `Tombstone` from a dying [`HotWitness`] and the Lance
    /// version number assigned by the versioned store at commit time.
    ///
    /// Compresses the hot-witness provenance to ~Scent/Base17 level.
    /// Called at mailbox-death time, before the `HotWitness` is dropped.
    ///
    /// # BLOCKED
    ///
    /// - Scent/Base17 compression entry point (see `compressed_payload` field).
    /// - Lance version number retrieval API — `VersionedGraph::commit_encounter_round`
    ///   returns a `u64` version per `versioned.rs`; confirm the exact call site.
    pub fn from_hot(_hot: &HotWitness, _lance_version: u64) -> Self {
        // BLOCKED: compression entry point for the payload.
        // BLOCKED: Lance version acquisition pattern.
        todo!("D-ATOM-5: construct Tombstone from HotWitness at mailbox-death")
    }

    /// Persist this `Tombstone` to the Lance versioned tombstone store.
    ///
    /// Appends one row to the tombstone Lance dataset (append-only — this is
    /// the GoBD audit invariant; rows are NEVER deleted or overwritten).
    ///
    /// # BLOCKED
    ///
    /// - Lance versioned-store write API.  The pattern in `versioned.rs` is
    ///   `Dataset::write(reader, path, Some(params)).await?` with
    ///   `WriteMode::Overwrite` (create-or-overwrite on first write).  For
    ///   tombstones the correct mode is **append** (`WriteMode::Append` or
    ///   `RecordBatchIterator` append path) — confirm the exact Lance 4.0.0
    ///   append API (workspace uses lance = 4.0.0 per `crates/lance-graph/
    ///   Cargo.toml:36`; `WriteMode::Append` availability in lance 4.x TBD).
    /// - Arrow schema for the tombstone RecordBatch (column names + types).
    /// - Dataset path convention (where does the tombstone store live relative
    ///   to the `VersionedGraph` base path?).
    pub async fn persist(&self, _base_path: &str) -> crate::error::Result<()> {
        // BLOCKED: Lance WriteMode::Append API in lance 4.0.0.
        // BLOCKED: Arrow schema + RecordBatch construction for tombstone row.
        // BLOCKED: tombstone store path convention.
        todo!("D-ATOM-5: persist Tombstone to Lance versioned tombstone store (append-only)")
    }
}

// ── WitnessLink ───────────────────────────────────────────────────────────────

/// Back-pointer linking a calcified cold SPO fact ↔ its [`Tombstone`].
///
/// # Link integrity (E-LADDER-SERVES-MAILBOX §6 "the one thing to nail")
///
/// Two invariants must hold:
///
/// 1. **SPO → Tombstone:** Every calcified SPO fact that was committed by a
///    mailbox (i.e. went through [`calcify`] + [`Tombstone::from_hot`]) carries
///    a `WitnessLink` with a valid `spo_key` + `tombstone_lance_version`.  The
///    link is stored alongside the SPO record (either as a field of the SPO
///    quad or as a side-table join on `spo_key`).
///
/// 2. **Tombstone outlives mailbox:** The [`Tombstone`] is written to Lance
///    BEFORE the mailbox is dropped.  Lance versioning then guarantees
///    immutability — the tombstone cannot be deleted.  The `WitnessLink` points
///    to a version number that is always readable via
///    `VersionedGraph::at_version(lance_version)`.
///
/// # GoBD audit fallout (E-FIBU-GOBD-BY-CONSTRUCTION)
///
/// The append-only Lance versioned tombstone store IS the audit log.  Any
/// regulatory query ("which mailbox committed fact X, when, with what
/// confidence?") resolves by:
/// 1. Look up the SPO fact's `WitnessLink.spo_key`.
/// 2. Open the tombstone store at `tombstone_lance_version`.
/// 3. Read the `Tombstone` row for that key.
///
/// No separate audit log, no bolted-on provenance system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessLink {
    /// The SPO fact key (dn_hash of the subject–predicate–object triple).
    ///
    /// Primary join key between the cold SPO record and this link.
    pub spo_key: u64,

    /// The mailbox that calcified this fact.
    ///
    /// Redundant with `Tombstone::mailbox_id` but stored in the link for
    /// O(1) provenance lookup without opening the tombstone dataset.
    pub mailbox_id: u64,

    /// Lance version number at which the corresponding [`Tombstone`] was
    /// written.
    ///
    /// Use `VersionedGraph::at_version(tombstone_lance_version)` to open the
    /// exact snapshot containing this tombstone row.
    pub tombstone_lance_version: u64,
}

impl WitnessLink {
    /// Construct a `WitnessLink` from the SPO key and its corresponding
    /// `Tombstone`.
    ///
    /// Called immediately after [`Tombstone::persist`] returns the committed
    /// Lance version, so the link is always consistent with a durable row.
    pub fn new(spo_key: u64, tombstone: &Tombstone) -> Self {
        Self {
            spo_key,
            mailbox_id: tombstone.mailbox_id,
            tombstone_lance_version: tombstone.lance_version,
        }
    }

    /// Verify that the linked [`Tombstone`] is readable at the stored Lance
    /// version.
    ///
    /// Used by audit / integrity-check tooling (GoBD audit path).
    ///
    /// # BLOCKED
    ///
    /// - Lance `Dataset::checkout_version` API path — `versioned.rs` uses
    ///   `ds.checkout_version(version).await?`; confirm the tombstone dataset
    ///   open pattern (path + schema + version pin).
    pub async fn verify(&self, _base_path: &str) -> crate::error::Result<bool> {
        // BLOCKED: Lance checkout_version + tombstone-row lookup API.
        todo!("D-ATOM-5: verify WitnessLink → open tombstone dataset at lance_version, confirm row exists for spo_key")
    }
}
