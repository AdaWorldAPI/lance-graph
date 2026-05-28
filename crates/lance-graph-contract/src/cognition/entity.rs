//! The typed carrier — [`NormalizedEntity<Stage>`].
//!
//! `NormalizedEntity` is a typed lens into a [`MailboxSoA`] row; it
//! does NOT own the four cognitive columns. The mailbox does. Per
//! E-CE64-MB-4: Rust move/ownership semantics prove no aliasing / no
//! data race at compile time.
//!
//! ## Cross-references
//! - Plan: `.claude/plans/normalized-entity-holy-grail-v1.md`
//! - Epiphanies: E-NORMALIZED-ENTITY-1, E-CODEBOOK-INHERITS-FROM-OGIT
//! - PR #427 (thoughtspace columns, WitnessTable widening)

use core::marker::PhantomData;
use super::stages::{Raw, Stage};

// ── MailboxRow ────────────────────────────────────────────────────────────────

/// Typed handle into a `MailboxSoA` row.
///
/// `mailbox_ref` is the per-cohort mailbox identifier; `row_idx` is
/// the row inside that mailbox. Both fields are `Copy`; the mailbox
/// owns the actual SoA columns.
///
/// `u32` for `mailbox_ref` matches PR #427's WitnessTable widening
/// (was `u16`, promoted to accommodate > 65 K cohorts).
/// `u16` for `row_idx` matches the per-mailbox envelope (64 K–256 K
/// rows per mailbox per `D-MBX-A4`).
///
/// Per E-CE64-MB-4: mailbox-as-owner topology makes Rust ownership
/// prove no aliasing / no data race at compile time. `MailboxRow` is
/// `Copy` (two ints); the mailbox owns the actual SoA columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MailboxRow {
    /// Wide enough for PR #427's witness_table widening.
    pub mailbox_ref: u32,
    /// Row inside the per-mailbox SoA (64 K–256 K envelope).
    pub row_idx: u16,
}

// ── Zero-dep placeholder handles ─────────────────────────────────────────────
//
// TODO(Stage 2): These placeholder handles stand in for the real types that live in
// other crates (`OdooEntity` in lance-graph-ontology, `OgitUri` in
// callcenter::ogit_uris, `OwlClass` in the OWL registry). For Stage 1 we
// use static-str wrappers so this crate stays zero-dep. Stage 2 replaces
// these with `&'static OdooEntity` etc. once the ontology crate depends on
// contract (or the relevant type is moved down to contract).

/// Zero-dep handle for an Odoo entity — wraps the `model_name` string
/// (e.g. `"account.move"`).
///
// TODO(Stage 2): Stage 2 replaces with `&'static lance_graph_ontology::OdooEntity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OdooEntityRef(pub &'static str);

/// Zero-dep handle for an OGIT URI (e.g. `"https://ogit.adaworldapi.com/callcenter#Invoice"`).
///
// TODO(Stage 2): Stage 2 replaces with `&'static lance_graph_contract::callcenter::ogit_uris::OgitUri`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OgitUriRef(pub &'static str);

/// Zero-dep handle for an OWL class IRI.
///
// TODO(Stage 2): Stage 2 replaces with a real `&'static OwlClass` from the
// OWL registry once it lands in the contract crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OwlClassRef(pub &'static str);

/// Zero-dep handle for a FIBU/FIBO alignment frame identifier.
///
// TODO(Stage 2): Stage 2 replaces with a typed alignment struct once the
// FIBU overlay is defined in a dependent crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FibuAlignmentRef(pub &'static str);

// ── DolceCategory ─────────────────────────────────────────────────────────────

/// DOLCE upper-ontology category for the entity.
///
/// Populated by the `classify_dolce` verb (see [`super::advance`]).
///
// TODO(Stage 2): the existing DOLCE classifier in
// `lance-graph-ontology::dolce_odoo` has richer sub-categories;
// Stage 2 expands this enum to match its full discriminant set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DolceCategory {
    /// DOLCE `Endurant` — spatially and temporally extended objects
    /// (e.g. a legal entity, a product).
    Endurant,
    /// DOLCE `Perdurant` — events and processes (e.g. a posting, a
    /// fiscal-year close).
    Perdurant,
    /// DOLCE `Abstract Object` — mathematical or informational objects
    /// (e.g. an account template, an OGIT class).
    AbstractObject,
    /// DOLCE `Quality` — properties that inhere in other particulars
    /// (e.g. a VAT rate, a currency).
    Quality,
    /// DOLCE `Region` — value spaces for qualities (e.g. a date range,
    /// an account-code interval).
    Region,
    /// Catch-all for DOLCE sub-categories not yet enumerated.
    ///
    // TODO(Stage 2): Stage 2 expands by auditing all `_inherit`/`_inherits`
    // chains in the EXT-2 output and mapping each to a DOLCE sub-
    // category from the `lance-graph-ontology::dolce_odoo` classifier.
    Other,
}

// ── NormalizedEntity ──────────────────────────────────────────────────────────

/// The single typed carrier holding the 4-way inheritance chain
/// (E-NORMALIZED-ENTITY-1).
///
/// Stage is phantom-typed; advancement is gated by the
/// [`super::advance`] methods. The struct is `Copy` (all fields are
/// small `Copy` types); the SoA columns live in the owning mailbox.
///
/// ## Inheritance chain
///
/// ```text
/// odoo  (Odoo model_name) ← source of truth from EXT-2..6
/// ogit  (OGIT URI)        ← codebook entry resolved from model_name
/// owl   (OWL class)       ← TTL join on the OGIT URI
/// dolce (DOLCE category)  ← upper-ontology classifier
/// fibu  (FIBU alignment)  ← domain overlay (German accounting frames)
/// ```
///
/// Slots populate as stages advance via the five-verb algebra.
///
/// ## Zero-dep placeholder types
///
/// For Stage 1 the inheritance slots use static-str placeholder handles
/// (`OdooEntityRef`, `OgitUriRef`, `OwlClassRef`, `FibuAlignmentRef`)
/// so this crate stays zero-dep. Stage 2 replaces these with the real
/// types from dependent crates.
///
/// ## Cross-references
/// - Carrier shape: `.claude/plans/normalized-entity-holy-grail-v1.md`
///   §"The carrier"
/// - Epiphanies: E-NORMALIZED-ENTITY-1, E-CODEBOOK-INHERITS-FROM-OGIT
/// - MailboxSoA columns: PR #427
#[derive(Debug, Clone, Copy)]
pub struct NormalizedEntity<S: Stage = Raw> {
    /// Source-of-truth Odoo identity (the `model_name` handle).
    ///
    // TODO(Stage 2): Stage 2 replaces `OdooEntityRef` with
    // `&'static lance_graph_ontology::OdooEntity` (or moves the
    // type down to contract if the ontology crate can depend on
    // contract without a cycle).
    pub odoo: OdooEntityRef,

    /// OGIT URI resolved in the `WithOgit` stage.
    ///
    /// `None` until `resolve_ogit` is called.
    pub ogit: Option<OgitUriRef>,

    /// OWL class hydrated in the `WithOwl` stage.
    ///
    /// `None` until `hydrate_owl` is called.
    pub owl: Option<OwlClassRef>,

    /// DOLCE upper-ontology category classified in the `WithDolce` stage.
    ///
    /// `None` until `classify_dolce` is called.
    pub dolce: Option<DolceCategory>,

    /// FIBU/FIBO alignment frame populated in the `Normalized` stage.
    ///
    /// `None` until `align_fibu` is called.
    pub fibu: Option<FibuAlignmentRef>,

    /// Typed handle into the owning `MailboxSoA` row.
    ///
    /// The mailbox owns the actual SoA columns (edges / qualia / meta /
    /// entity_type); `NormalizedEntity` is a typed lens onto them.
    ///
    // TODO(Stage 2): in Stage 2 the advancement verbs also write back into
    // the mailbox's SoA fingerprint column with the resolved OGIT
    // identity, once `cognitive-shader-driver` is a hard dependency.
    pub row: MailboxRow,

    /// Phantom stage marker. Zero size; never stored at runtime.
    _stage: PhantomData<S>,
}

impl<S: Stage> NormalizedEntity<S> {
    /// Read the Odoo entity reference (always present regardless of stage).
    #[inline]
    pub fn odoo(&self) -> OdooEntityRef {
        self.odoo
    }

    /// Read the resolved OGIT URI (present from `WithOgit` onwards).
    #[inline]
    pub fn ogit(&self) -> Option<OgitUriRef> {
        self.ogit
    }

    /// Read the resolved OWL class (present from `WithOwl` onwards).
    #[inline]
    pub fn owl(&self) -> Option<OwlClassRef> {
        self.owl
    }

    /// Read the DOLCE category (present from `WithDolce` onwards).
    #[inline]
    pub fn dolce(&self) -> Option<DolceCategory> {
        self.dolce
    }

    /// Read the FIBU alignment (present from `Normalized` onwards).
    #[inline]
    pub fn fibu(&self) -> Option<FibuAlignmentRef> {
        self.fibu
    }

    /// Read the MailboxSoA row handle.
    #[inline]
    pub fn row(&self) -> MailboxRow {
        self.row
    }
}

impl NormalizedEntity<Raw> {
    /// Construct a `NormalizedEntity` at the `Raw` stage from an Odoo
    /// model_name.
    ///
    /// This is the only public constructor. All other stages are reached
    /// via the five-verb advancement methods in [`super::advance`].
    ///
    /// `const fn` so callers can create static sentinel entities (e.g.
    /// in test fixtures or codebook tables).
    pub const fn raw(odoo: OdooEntityRef, row: MailboxRow) -> NormalizedEntity<Raw> {
        NormalizedEntity {
            odoo,
            ogit: None,
            owl: None,
            dolce: None,
            fibu: None,
            row,
            _stage: PhantomData,
        }
    }
}

// ── Internal advancement helper ───────────────────────────────────────────────

impl<S: Stage> NormalizedEntity<S> {
    /// Internal: transition the phantom stage to `T`, keeping all data
    /// fields identical. Only called from [`super::advance`].
    ///
    /// Not `pub` — consumers go through the typed verb methods, never
    /// raw stage-casts.
    /// Advance the phantom stage to `T`, copying all data fields.
    ///
    /// Used by [`Op<I,O>::apply`](super::op::Op) implementations to
    /// construct the output entity after applying business logic. The
    /// type parameter `T` is constrained by the `Op<I,O>` trait bounds
    /// at the call site, so stage correctness is enforced by the Op
    /// trait, not by this method's signature.
    ///
    /// Not callable on the advancement verbs themselves (those use it
    /// internally); exposed `pub` so external Op implementors can
    /// construct stage-transitioned entities in their `apply` bodies.
    pub fn advance_stage<T: Stage>(self) -> NormalizedEntity<T> {
        NormalizedEntity {
            odoo: self.odoo,
            ogit: self.ogit,
            owl: self.owl,
            dolce: self.dolce,
            fibu: self.fibu,
            row: self.row,
            _stage: PhantomData,
        }
    }
}
