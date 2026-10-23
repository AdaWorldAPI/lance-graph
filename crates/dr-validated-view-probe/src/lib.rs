//! D-NFR-6: validate a DR64 binding once, then run several native operations
//! over a view whose lifetime keeps the binding valid. Lab only.
//!
//! Three responsibilities, kept apart:
//!
//! - **Method resolution.** Which native function implements an operation is
//!   fixed per method and resolved once ([`KERNELS`], a `LazyLock`). Nothing
//!   here chooses a route per call.
//! - **Data binding.** Whether a frame, focus and scope still hold. This can go
//!   stale, so it is validated, by the existing `bind_dr64` and nothing new,
//!   when a [`View`] is acquired.
//! - **Execution.** A [`View`]'s operations call `ndarray::simd` directly on
//!   borrowed data and validate nothing.
//!
//! What makes skipping validation per operation sound is not a cache but
//! ownership. A [`Table`] owns its data, its [`FrameStore`] and its focus map
//! together. Every change to any of them needs `&mut Table`, and a [`View`]
//! holds `&Table` for its whole life. So while a view exists none of them can
//! change, and that is checked by the compiler, not at run time:
//!
//! ```compile_fail,E0502
//! use dr_validated_view_probe::fixture::*;
//! let (mut t, g, m, guards) = setup(64);
//! let view = validate_presence(&t, &g, m, &guards).unwrap();
//! t.advance(revisions(2), data(64, 5)); // F1: the source cannot move under a view
//! drop(view);
//! ```
//!
//! ```compile_fail,E0502
//! use dr_validated_view_probe::fixture::*;
//! let (mut t, g, m, guards) = setup(64);
//! let view = validate_presence(&t, &g, m, &guards).unwrap();
//! t.refocus([0; 64]); // F2: nor can the focus map
//! drop(view);
//! ```
//!
//! ```compile_fail,E0502
//! use dr_validated_view_probe::fixture::*;
//! let (t, mut g, m, guards) = setup(64);
//! let view = validate_presence(&t, &g, m, &guards).unwrap();
//! g.revoke(); // F3, owned grants: nor can the scope be revoked
//! drop(view);
//! ```
//!
//! ```compile_fail,E0597
//! use dr_validated_view_probe::fixture::*;
//! let leaked;
//! {
//!     let (t, g, m, guards) = setup(64);
//!     leaked = validate_presence(&t, &g, m, &guards).unwrap();
//! } // F4: a view cannot outlive the table it was validated against
//! let _ = leaked.count();
//! ```
//!
//! ```compile_fail,E0597
//! use dr_validated_view_probe::fixture::*;
//! let (t, g, m, guards) = setup(64);
//! let view = validate_presence(&t, &g, m, &guards).unwrap();
//! let rows: &'static [u64; 64] = view.rows(); // F7: a borrow cannot escape as 'static
//! ```
//!
//! What ownership does NOT cover, and where the runtime check therefore stays:
//!
//! - **Revocation owned elsewhere.** If scope revocation is owned by another
//!   component (an ACL service, an `Arc` shared across threads), the view's
//!   borrow cannot stop it. [`SharedGrants`] models that: revocation is an
//!   atomic, and a [`SharedView`] reads one epoch per operation instead of
//!   re-running the whole guard.
//! - **Data outside the table.** `bind_dr64` accepts any borrowed operand
//!   with a valid frame. Its frame seal identifies a revision issue, not the
//!   bytes; coupling bytes to the frame is what [`Table`] adds. Data handed in
//!   from outside a `Table` keeps the per-use check.
#![forbid(unsafe_code)]

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub use dr_math_binding_probe::{Focus, Frame};
use dr_math_binding_probe::{
    FrameRevisions, FrameStore, Guards, MethodSignature, Operand, Refusal, Scope, bind_dr64,
};

/// Rows covered by one DR64 register: 64 words of 64 bits.
pub const MAX_ROWS: usize = 64 * 64;

/// The data a frame refers to: a DR64 selection register and the `i32` value
/// lane it selects from. Owned by a [`Table`] and changed only through
/// [`Table::advance`].
#[derive(Clone, Debug)]
pub struct Data {
    pub rows: [u64; 64],
    pub values: Vec<i32>,
}

/// Owner of the data, the frame store and the local-to-global focus map. Each
/// change goes through `&mut self`, so it cannot happen while a [`View`]
/// borrows the table.
#[derive(Debug)]
pub struct Table {
    store: FrameStore,
    data: Data,
    focus_revision: u64,
    local_to_global: [u32; 64],
}

impl Table {
    /// # Panics
    /// If `data.values` is longer than [`MAX_ROWS`].
    pub fn new(
        revisions: FrameRevisions,
        data: Data,
        focus_revision: u64,
        local_to_global: [u32; 64],
    ) -> Self {
        assert!(
            data.values.len() <= MAX_ROWS,
            "more rows than one DR64 register"
        );
        Self {
            store: FrameStore::new(revisions),
            data,
            focus_revision,
            local_to_global,
        }
    }

    pub fn seal(&self) -> Frame {
        self.store.seal()
    }

    pub fn focus(&self) -> Focus {
        Focus {
            revision: self.focus_revision,
        }
    }

    pub fn store(&self) -> &FrameStore {
        &self.store
    }

    pub fn local_to_global(&self) -> &[u32; 64] {
        &self.local_to_global
    }

    /// Replace the data and its revisions together, so new bytes never sit
    /// under an old frame.
    ///
    /// # Panics
    /// If `data.values` is longer than [`MAX_ROWS`].
    pub fn advance(&mut self, revisions: FrameRevisions, data: Data) {
        assert!(
            data.values.len() <= MAX_ROWS,
            "more rows than one DR64 register"
        );
        self.store.advance(revisions);
        self.data = data;
    }

    /// Replace the focus map. The table issues the next revision itself:
    /// focus identity is a revision number, so a caller-chosen number could
    /// change the map while keeping the revision, and nothing downstream
    /// would see it. Even an identical map gets a new revision.
    pub fn refocus(&mut self, local_to_global: [u32; 64]) {
        self.focus_revision += 1;
        self.local_to_global = local_to_global;
    }
}

/// Scope grants owned by the caller: revoking needs `&mut self`, so it cannot
/// happen while a [`View`] borrows them.
#[derive(Debug)]
pub struct Grants {
    scope: Option<Scope>,
}

impl Grants {
    pub fn new(scope: Scope) -> Self {
        Self { scope: Some(scope) }
    }

    pub fn current(&self) -> Option<Scope> {
        self.scope
    }

    pub fn revoke(&mut self) {
        self.scope = None;
    }
}

/// Scope grants owned elsewhere and revocable through `&self`, as when they
/// are shared through an `Arc`. A borrow cannot prevent revocation here; see
/// [`SharedView`].
#[derive(Debug)]
pub struct SharedGrants {
    scope: Scope,
    /// Even: granted. Odd: revoked. Bumped by every revocation.
    epoch: AtomicU64,
}

impl SharedGrants {
    pub fn new(scope: Scope) -> Self {
        Self {
            scope,
            epoch: AtomicU64::new(0),
        }
    }

    pub fn revoke(&self) {
        self.epoch.fetch_or(1, Ordering::Release);
    }

    fn current(&self) -> Option<(Scope, u64)> {
        let e = self.epoch.load(Ordering::Acquire);
        (e & 1 == 0).then_some((self.scope, e))
    }
}

/// Why a view was not issued or an operation did not answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewRefusal {
    /// The existing validator's refusal, unchanged.
    Bind(Refusal),
    /// The grants no longer carry a scope.
    ScopeRevoked,
}

impl From<Refusal> for ViewRefusal {
    fn from(r: Refusal) -> Self {
        ViewRefusal::Bind(r)
    }
}

// ── method resolution, once ────────────────────────────────────────────────

/// The native kernels a view runs, resolved once. A copied `fn` pointer
/// captures nothing, so it cannot hold a view's borrow past a call.
#[derive(Clone, Copy)]
pub struct Kernels {
    pub count: fn(&[u64]) -> u64,
    pub sum: fn(&[i32], &[u64]) -> i64,
    pub min: fn(&[i32], &[u64]) -> Option<i32>,
    pub max: fn(&[i32], &[u64]) -> Option<i32>,
    pub gt: fn(&[i32], i32, &mut [u64]),
    pub and: fn(&[u64], &[u64], &mut [u64]),
}

pub static KERNELS: LazyLock<Kernels> = LazyLock::new(|| Kernels {
    count: ndarray::simd::popcount_batch_u64,
    sum: ndarray::simd::masked_sum_i32,
    min: ndarray::simd::masked_min_i32,
    max: ndarray::simd::masked_max_i32,
    gt: ndarray::simd::gt_i32_to_mask,
    and: ndarray::simd::mask_and,
});

// ── acquisition: the one validation ───────────────────────────────────────

/// Validate once and return a view. This is the only check: the existing
/// `bind_dr64` (method, axes, semantics, validity, frame, focus, scope),
/// with the focus taken from the table and the scope from the grants rather
/// than from the caller.
pub fn validate<'t>(
    table: &'t Table,
    grants: &'t Grants,
    declared: MethodSignature,
    requested: MethodSignature,
    bound_at: &Guards,
) -> Result<View<'t>, ViewRefusal> {
    let scope = grants.current().ok_or(ViewRefusal::ScopeRevoked)?;
    let rows = bind_rows(table, declared, requested, bound_at, scope)?;
    Ok(View {
        rows,
        values: &table.data.values,
        kernels: *KERNELS,
        _borrow: std::marker::PhantomData,
    })
}

/// [`validate`] against grants another owner can revoke. The view records the
/// grant epoch and checks it on every operation.
pub fn validate_shared<'t>(
    table: &'t Table,
    grants: &'t SharedGrants,
    declared: MethodSignature,
    requested: MethodSignature,
    bound_at: &Guards,
) -> Result<SharedView<'t>, ViewRefusal> {
    let (scope, epoch) = grants.current().ok_or(ViewRefusal::ScopeRevoked)?;
    let rows = bind_rows(table, declared, requested, bound_at, scope)?;
    Ok(SharedView {
        view: View {
            rows,
            values: &table.data.values,
            kernels: *KERNELS,
            _borrow: std::marker::PhantomData,
        },
        grants,
        epoch,
    })
}

fn bind_rows<'t>(
    table: &'t Table,
    declared: MethodSignature,
    requested: MethodSignature,
    bound_at: &Guards,
    scope: Scope,
) -> Result<&'t [u64; 64], ViewRefusal> {
    let bound = bind_dr64(
        &table.store,
        declared,
        requested,
        bound_at,
        (table.focus(), scope),
        Operand {
            rows: &table.data.rows,
            validity: None,
        },
    )?;
    // The view passes no validity plane, so `bind_dr64` refuses every
    // three-valued method (`ThreeValuedNeedsValidity`) and every Bag
    // (`BagNeedsMultiplicity`): only two-valued Distinct reaches here.
    Ok(bound.rows)
}

// ── execution: no validation ───────────────────────────────────────────────

/// A validated, lifetime-borrowed binding. Its operations call the resolved
/// native kernels directly and check nothing; the borrow of the table and the
/// grants is what keeps the validation true.
#[derive(Clone, Copy)]
pub struct View<'t> {
    rows: &'t [u64; 64],
    values: &'t [i32],
    kernels: Kernels,
    _borrow: std::marker::PhantomData<&'t ()>,
}

impl<'t> View<'t> {
    pub fn rows(&self) -> &'t [u64; 64] {
        self.rows
    }

    pub fn values(&self) -> &'t [i32] {
        self.values
    }

    fn words(&self) -> &'t [u64] {
        &self.rows[..self.values.len().div_ceil(64)]
    }

    /// `COUNT(*)` of the selection.
    pub fn count(&self) -> u64 {
        (self.kernels.count)(self.words())
    }

    /// `SUM(v)` over the selection.
    pub fn sum(&self) -> i64 {
        (self.kernels.sum)(self.values, self.words())
    }

    /// `MIN(v)` over the selection; `None` for an empty selection.
    pub fn min(&self) -> Option<i32> {
        (self.kernels.min)(self.values, self.words())
    }

    /// `MAX(v)` over the selection; `None` for an empty selection.
    pub fn max(&self) -> Option<i32> {
        (self.kernels.max)(self.values, self.words())
    }

    /// `COUNT(*) WHERE v > t` within the selection: one mask comparison, one
    /// mask AND, one count, into the caller's scratch (two masks of
    /// `words` words). No allocation.
    ///
    /// # Panics
    /// If `scratch` holds fewer than two masks' worth of words.
    pub fn count_gt(&self, t: i32, scratch: &mut [u64]) -> u64 {
        let w = self.words().len();
        let (cmp, rest) = scratch.split_at_mut(w);
        let both = &mut rest[..w];
        (self.kernels.gt)(self.values, t, cmp);
        (self.kernels.and)(cmp, self.words(), both);
        (self.kernels.count)(both)
    }
}

/// A view whose scope can be revoked by another owner. Each operation loads
/// the grant epoch once and refuses if it moved; frame and focus are still
/// held by the table borrow and are not rechecked.
pub struct SharedView<'t> {
    view: View<'t>,
    grants: &'t SharedGrants,
    epoch: u64,
}

impl<'t> SharedView<'t> {
    #[inline]
    fn live(&self) -> Result<&View<'t>, ViewRefusal> {
        if self.grants.epoch.load(Ordering::Acquire) == self.epoch {
            Ok(&self.view)
        } else {
            Err(ViewRefusal::ScopeRevoked)
        }
    }

    pub fn count(&self) -> Result<u64, ViewRefusal> {
        Ok(self.live()?.count())
    }

    pub fn sum(&self) -> Result<i64, ViewRefusal> {
        Ok(self.live()?.sum())
    }

    pub fn min(&self) -> Result<Option<i32>, ViewRefusal> {
        Ok(self.live()?.min())
    }

    pub fn max(&self) -> Result<Option<i32>, ViewRefusal> {
        Ok(self.live()?.max())
    }

    pub fn count_gt(&self, t: i32, scratch: &mut [u64]) -> Result<u64, ViewRefusal> {
        Ok(self.live()?.count_gt(t, scratch))
    }
}

/// Shared fixtures for the tests, the example and the compile-fail doctests.
pub mod fixture {
    use super::*;
    use dr_math_binding_probe::{Axis, AxisKind};
    use lance_graph_contract::class_view::ClassId;

    pub fn revisions(source: u64) -> FrameRevisions {
        FrameRevisions {
            source,
            relation_map: 1,
            coordinates: 1,
        }
    }

    /// `n` rows; the selection keeps rows `r % 3 != 1`; values are signed.
    pub fn data(n: usize, salt: i32) -> Data {
        let mut rows = [0u64; 64];
        for r in (0..n).filter(|r| r % 3 != 1) {
            rows[r / 64] |= 1 << (r % 64);
        }
        Data {
            rows,
            values: (0..n)
                .map(|i| ((i as i32 * 37 + salt) % 201) - 100)
                .collect(),
        }
    }

    pub fn method(ordinal: u16) -> MethodSignature {
        let axis = |kind| Axis {
            class: 7 as ClassId,
            kind,
        };
        MethodSignature::presence(
            7,
            ordinal,
            axis(AxisKind::Identity),
            axis(AxisKind::Property),
        )
    }

    /// A table over `n` rows, grants for scope 3, the method, and the guards
    /// it was bound under (frame sealed now, focus 7, scope 3).
    pub fn setup(n: usize) -> (Table, Grants, MethodSignature, Guards) {
        let t = Table::new(revisions(1), data(n, 0), 7, [0; 64]);
        let guards = Guards {
            frame: t.seal(),
            focus: t.focus(),
            scope: Scope { id: 3 },
        };
        (t, Grants::new(Scope { id: 3 }), method(1), guards)
    }

    pub fn validate_presence<'t>(
        t: &'t Table,
        g: &'t Grants,
        m: MethodSignature,
        guards: &Guards,
    ) -> Result<View<'t>, ViewRefusal> {
        validate(t, g, m, m, guards)
    }
}
