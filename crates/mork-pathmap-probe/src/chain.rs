//! Composition probe: one question's answer supplies the addresses the next
//! question needs, and the relational geometry prepared for the first is
//! reused by the second wherever its witness still holds.
//!
//! STATUS: PROBE, lab only. This is the D-RPR-1 handoff of
//! `.claude/plans/recursive-perspective-rebinding-v1.md`, not an admission of
//! anything. No store, executor, opcode or relationship type is added: this
//! module calls what [`crate::rmap`] already has, in an order, and counts.
//!
//! The handoff, step by step:
//!
//! 1. **Fold.** A question is reachability from seeds over its own predicates,
//!    run by [`TileExec::reach`]: a mask-risc program over tile planes that
//!    all share one `(target, source)` address space. No operand is indexed by
//!    another operand's values ([`fold_program_is_co_addressed`]).
//! 2. **Addresses out.** The fold returns the slots it reached and, from the
//!    prepared CSR's escape list, the global identities its own predicates
//!    lead to outside the focus within the hop budget. Only those are
//!    demanded; other escapes of the CSR are not.
//! 3. **Lookup.** While escapes exist the answer is not global
//!    ([`crate::rmap::ScopedReach::global`] refuses `IncompleteFocus`). The
//!    escaped identities are bound into evictable slots and [`rebind`] seeks
//!    each new identity's outgoing edges on the sorted run. That seek, keyed by
//!    values the previous fold produced, is the gather-shaped step. It runs in
//!    preparation, counted, outside the fold (H14c §1 rule 6); whether that
//!    placement is admitted is D-RMAP-1 §6's open review.
//! 4. **Translation.** Inside `rebind` (and [`Csr::prepare`]) each target is
//!    translated once from a global identity to a slot ([`Focus::slot_of`]).
//! 5. **Reuse.** Unchanged slots keep their prepared edges for every
//!    predicate; a later question over another predicate of the same frame
//!    starts from that geometry and pays a lookup only for what it adds.
//!
//! Slots are evicted only if they hold no identity an answer still needs
//! (`keep`); if the demanded identities do not fit, the question is refused,
//! never answered from an incomplete focus.
//!
//! Semantics are Boolean reachability, as in [`crate::rmap`]: no bags, no
//! path counts.

use std::cell::Cell;
use std::collections::BTreeSet;

use crate::rmap::{
    Csr, Focus, Navigate, Refusal, SlotSet, SortedRun, Tile, TileExec, Witness, rebind, slot_set,
    slots_of,
};

/// Counts the seeks a preparation makes through [`Navigate`]: one per
/// `objects(p, s)` call, plus every `(o, row)` it yields. Behaviour is the
/// wrapped index's.
pub struct CountingIndex<'a, 'f> {
    pub inner: &'a SortedRun<'f>,
    pub seeks: Cell<usize>,
    pub yielded: Cell<usize>,
}

impl<'a, 'f> CountingIndex<'a, 'f> {
    pub fn new(inner: &'a SortedRun<'f>) -> Self {
        CountingIndex {
            inner,
            seeks: Cell::new(0),
            yielded: Cell::new(0),
        }
    }
}

impl Navigate for CountingIndex<'_, '_> {
    fn objects(&self, p: u32, s: u32, f: &mut dyn FnMut(u32, u32)) {
        self.seeks.set(self.seeks.get() + 1);
        let y = &self.yielded;
        self.inner.objects(p, s, &mut |o, r| {
            y.set(y.get() + 1);
            f(o, r)
        });
    }
}

/// True if the tile fold's mask-risc program for `n` predicate planes (and an
/// optional thought delta) reads only planes of one address space: no
/// [`lance_graph_mask_risc::MaskOp::Gather`], no foreign group key, no
/// scatter terminal.
pub fn fold_program_is_co_addressed(n: usize, delta: bool) -> bool {
    is_co_addressed(&TileExec::program(n, delta))
}

/// True if `p` reads only planes of one address space: no
/// [`lance_graph_mask_risc::MaskOp::Gather`], and a `Keep`, `Count`, `Any` or
/// `All` terminal (no scatter, no foreign group key).
pub fn is_co_addressed(p: &lance_graph_mask_risc::Program) -> bool {
    use lance_graph_mask_risc::{MaskOp, Terminal};
    !p.ops.iter().any(|op| matches!(op, MaskOp::Gather { .. }))
        && matches!(
            p.terminal,
            Terminal::Keep { .. }
                | Terminal::Count { .. }
                | Terminal::Any { .. }
                | Terminal::All { .. }
        )
}

/// Why a chain step did not answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainRefusal {
    /// The frame was built against other facts, seal, focus or scope.
    Stale(Refusal),
    /// The demanded identities do not fit beside the ones that must be kept.
    FocusExhausted { needed: usize, free: usize },
}

/// How a focus change reaches the prepared operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rebuild {
    /// [`rebind`]: unchanged slots keep their edges; only changed slots seek.
    Incremental,
    /// [`Csr::prepare`] and [`Tile::from_csr`] from scratch: every slot seeks.
    Full,
}

/// A focus and the operands prepared for it, for every predicate the chain uses.
pub struct Frame {
    pub focus: Focus,
    pub csr: Csr,
    pub tile: Tile,
    pub preds: Vec<u32>,
    /// Seeks made by the initial preparation.
    pub prepare_seeks: usize,
}

/// One admission: the demanded identities that were outside the focus, and
/// what bringing them in cost in index seeks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admission {
    /// Global identities produced by the previous fold (or the step's seeds)
    /// that were not in the focus: the addresses that drove the lookup.
    pub demanded: BTreeSet<u32>,
    /// Slots whose identity changed.
    pub changed: usize,
    /// Index seeks performed. Incremental: `changed × preds`, one
    /// `neighbours` per changed slot and predicate (the loop in [`rebind`]).
    /// Full: counted through [`CountingIndex`].
    pub seeks: usize,
    /// Seeks a full preparation of the new focus would have made.
    pub full_seeks: usize,
}

impl Frame {
    pub fn prepare(run: &SortedRun<'_>, focus: Focus, preds: &[u32], current: &Witness) -> Frame {
        let idx = CountingIndex::new(run);
        let csr = Csr::prepare(&idx, &focus, preds, at(current, &focus));
        let tile = Tile::from_csr(&csr);
        Frame {
            focus,
            csr,
            tile,
            preds: preds.to_vec(),
            prepare_seeks: idx.seeks.get(),
        }
    }

    /// The slots of the given identities. `None` if one is not in the focus.
    pub fn slots(&self, ids: &BTreeSet<u32>) -> Option<SlotSet> {
        let mut out = Vec::with_capacity(ids.len());
        for &g in ids {
            out.push(self.focus.slot_of(g)?);
        }
        Some(slot_set(out))
    }

    /// Bring `ids` into the focus, evicting slots whose identity is not in
    /// `keep`. `None` if every id was already focused (no lookup made).
    pub fn admit(
        &mut self,
        run: &SortedRun<'_>,
        current: &Witness,
        keep: &BTreeSet<u32>,
        ids: &BTreeSet<u32>,
        how: Rebuild,
    ) -> Result<Option<Admission>, ChainRefusal> {
        let demanded: BTreeSet<u32> = ids
            .iter()
            .copied()
            .filter(|g| self.focus.slot_of(*g).is_none())
            .collect();
        if demanded.is_empty() {
            return Ok(None);
        }
        let free: Vec<u8> = (0..self.focus.len())
            .rev()
            .map(|s| s as u8)
            .filter(|&s| !keep.contains(&self.focus.id(s)) && !ids.contains(&self.focus.id(s)))
            .collect();
        if free.len() < demanded.len() {
            return Err(ChainRefusal::FocusExhausted {
                needed: demanded.len(),
                free: free.len(),
            });
        }
        let changes: Vec<(u8, u32)> = free.into_iter().zip(demanded.iter().copied()).collect();
        let focus = self.focus.replace(&changes);
        let w = at(current, &focus);
        let full_seeks = focus.len() * self.preds.len();
        let (csr, tile, seeks) = match how {
            Rebuild::Incremental => {
                let (csr, tile) = rebind(run, &self.focus, &focus, &self.csr, w);
                (csr, tile, changes.len() * self.preds.len())
            }
            Rebuild::Full => {
                let idx = CountingIndex::new(run);
                let csr = Csr::prepare(&idx, &focus, &self.preds, w);
                let tile = Tile::from_csr(&csr);
                (csr, tile, idx.seeks.get())
            }
        };
        self.focus = focus;
        self.csr = csr;
        self.tile = tile;
        Ok(Some(Admission {
            demanded,
            changed: changes.len(),
            seeks,
            full_seeks,
        }))
    }
}

/// The caller's witness, read at this focus's generation.
fn at(current: &Witness, focus: &Focus) -> Witness {
    Witness {
        focus: focus.generation,
        ..*current
    }
}

/// One answer and the admissions it took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Answer {
    /// Global identities reached within the hop budget, seeds included.
    pub reached: BTreeSet<u32>,
    /// Each admission before the answer became global; empty if the frame
    /// already held everything the question demanded.
    pub admissions: Vec<Admission>,
    /// Folds run (one per ask; each refused-as-incomplete fold is one of them).
    pub folds: usize,
}

/// Reachability from `seeds` over `preds` within `hops`: fold on the frame,
/// and while the fold's answer has escapes, admit them and fold again. `keep`
/// holds the identities earlier answers still need; this answer's are added.
#[expect(clippy::too_many_arguments)]
pub fn ask(
    frame: &mut Frame,
    exec: &mut TileExec,
    run: &SortedRun<'_>,
    current: &Witness,
    preds: &[u32],
    seeds: &BTreeSet<u32>,
    hops: usize,
    keep: &mut BTreeSet<u32>,
    how: Rebuild,
) -> Result<Answer, ChainRefusal> {
    keep.extend(seeds.iter().copied());
    let mut admissions = Vec::new();
    admissions.extend(frame.admit(run, current, keep, seeds, how)?);
    let mut folds = 0;
    loop {
        let seed_slots = frame.slots(seeds).expect("seeds were admitted");
        let w = at(current, &frame.focus);
        let r = exec
            .reach(
                &frame.tile,
                &frame.csr,
                &w,
                None,
                None,
                preds,
                &seed_slots,
                hops,
            )
            .map_err(ChainRefusal::Stale)?;
        folds += 1;
        let reached: BTreeSet<u32> = slots_of(&r.reached)
            .into_iter()
            .map(|s| frame.focus.id(s))
            .collect();
        keep.extend(reached.iter().copied());
        if r.global().is_ok() {
            return Ok(Answer {
                reached,
                admissions,
                folds,
            });
        }
        admissions.extend(frame.admit(run, current, keep, &r.escaped, how)?);
    }
}
