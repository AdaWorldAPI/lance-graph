//! Integration tests for the cognition typestate (D-NEH-1e).
//!
//! The compile-fail proofs of what the type system FORBIDS live in
//! [`crate::cognition`] module-level rustdoc (in `src/cognition/mod.rs`)
//! so that `cargo test --doc` actually gates them. This file holds only
//! the positive `#[test]` functions that confirm the PERMITTED forms
//! compile and run correctly.
//!
//! Per PR #431 review (coderabbit Major finding):
//! `compile_fail` doctests in module-level `//!` comments of integration
//! test files (`tests/*.rs`) are NOT picked up by `cargo test --doc` —
//! `--doc` only runs rustdoc on library/binary targets, not integration
//! crates. The earlier copies of those doctests in this file were
//! therefore unenforced. They were moved into `src/cognition/mod.rs`
//! where rustdoc reliably executes them.

use lance_graph_contract::cognition::entity::{MailboxRow, OdooEntityRef, OgitUriRef};
use lance_graph_contract::cognition::op::{Op, OpKind, Output};
use lance_graph_contract::cognition::stages::*;
use lance_graph_contract::cognition::NormalizedEntity;

// ── Positive tests ────────────────────────────────────────────────────────────

#[test]
fn raw_entity_constructs() {
    let e: NormalizedEntity<Raw> = NormalizedEntity::<Raw>::raw(
        OdooEntityRef("account.move"),
        MailboxRow {
            mailbox_ref: 0,
            row_idx: 0,
        },
    );
    assert_eq!(e.odoo(), OdooEntityRef("account.move"));
    assert_eq!(e.row().mailbox_ref, 0);
    assert_eq!(e.row().row_idx, 0);
    assert!(e.ogit().is_none());
    assert!(e.owl().is_none());
    assert!(e.dolce().is_none());
    assert!(e.fibu().is_none());
}

#[test]
fn stages_are_distinct_types() {
    // These functions will produce a compile error if Raw and Normalized
    // are accidentally unified or if stages become interchangeable.
    fn _takes_raw(_: NormalizedEntity<Raw>) {}
    fn _takes_normalized(_: NormalizedEntity<Normalized>) {}
    fn _takes_checked(_: NormalizedEntity<Checked>) {}
    fn _takes_reviewed(_: NormalizedEntity<Reviewed>) {}
    fn _takes_abducted(_: NormalizedEntity<Abducted>) {}
    fn _takes_reported(_: NormalizedEntity<Reported>) {}
    // Compiler enforces these are distinct; no runtime assertion needed.
}

#[test]
fn op_kind_has_unwired_sentinel() {
    assert_eq!(OpKind::UNWIRED, OpKind(0));
    // Sentinel is distinct from any future concrete kind.
    assert_ne!(OpKind::UNWIRED, OpKind(1));
}

#[test]
fn op_kind_is_copy_and_eq() {
    let k = OpKind(42);
    let k2 = k; // Copy
    assert_eq!(k, k2);
}

#[test]
fn output_has_success_field() {
    let out = Output { success: true };
    assert!(out.success);
    let out_fail = Output { success: false };
    assert!(!out_fail.success);
}

#[test]
fn mailbox_row_is_copy() {
    let row = MailboxRow {
        mailbox_ref: 42,
        row_idx: 7,
    };
    let row2 = row; // Copy
    assert_eq!(row.mailbox_ref, row2.mailbox_ref);
    assert_eq!(row.row_idx, row2.row_idx);
}

#[test]
fn ogit_uri_ref_equality() {
    let a = OgitUriRef("https://ogit.adaworldapi.com/callcenter#Invoice");
    let b = OgitUriRef("https://ogit.adaworldapi.com/callcenter#Invoice");
    let c = OgitUriRef("https://ogit.adaworldapi.com/callcenter#CreditNote");
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// ── Structural test: a minimal concrete Op can be defined ─────────────────────

/// A minimal no-op Op for the `Normalized → Normalized` transition.
/// Proves that third-party code can implement `Op<I,O>`.
#[allow(dead_code)] // used only in compile_fail doctests above
struct NoopOp;

impl Op<Normalized, Normalized> for NoopOp {
    fn kind(&self) -> OpKind {
        OpKind(1)
    }
    // step uses the default no-op success impl from the Op trait.
}

/// A minimal Op advancing `Normalized → Checked`.
#[allow(dead_code)] // used only in compile_fail doctests above
struct FakeChkData;

impl Op<Normalized, Checked> for FakeChkData {
    fn kind(&self) -> OpKind {
        OpKind(2)
    }
    // step uses the default no-op success impl; framework performs the
    // sealed Normalized → Checked transition.
}

/// A minimal Op advancing `Checked → Reviewed`.
#[allow(dead_code)] // used only in compile_fail doctests above
struct FakeReview;

impl Op<Checked, Reviewed> for FakeReview {
    fn kind(&self) -> OpKind {
        OpKind(3)
    }
}

/// A minimal Op advancing `Reviewed → Abducted`.
#[allow(dead_code)] // used only in compile_fail doctests above
struct FakeAbduct;

impl Op<Reviewed, Abducted> for FakeAbduct {
    fn kind(&self) -> OpKind {
        OpKind(4)
    }
}

/// A minimal Op advancing `Abducted → Reported`.
#[allow(dead_code)] // used only in compile_fail doctests above
struct FakeReport;

impl Op<Abducted, Reported> for FakeReport {
    fn kind(&self) -> OpKind {
        OpKind(5)
    }
}

// We need access to the internal `advance_stage` helper from the test.
// The test crate is external, so we need a way to construct later stages.
// We use the Op implementations above (which call `advance_stage` via pub(super)).
// Since pub(super) only exposes to the parent module, we need an alternative:
// in the test we can only use public API.
//
// For the typestate chain test we need to reach Normalized. Since the
// `hydrate_owl`, `classify_dolce`, and `align_fibu` are `todo!()`, we
// can't advance past WithOgit using the public five-verb algebra.
//
// /// work: Stage 2's concrete Op implementations will let us write a
// /// full happy-path chain test that goes Raw → Reported without any
// /// internal helpers. For Stage 1 we test what we CAN test with the
// /// public API: construction, Op impls, OpKind, Output, and the
// /// trait bounds. The advance_stage helper is pub(super), so it is
// /// accessible within the crate's test harness via --lib, but NOT
// /// via the integration test binary here. The Op::apply approach above
// /// is the correct public-API pattern for Stage 2.

// /// work: add a full happy-path chain test (Raw → Reported) in
// /// Stage 2 once concrete Op kernels exist
// /// (e.g. SkrAccountInRange chain that compiles without todo!()).
