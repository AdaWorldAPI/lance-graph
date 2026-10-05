//! # lance-graph-dir-sim — explore a directory future over the SoA substrate
//!
//! ```text
//! observed G0 ──rule──► G1 ──rule──► G2 ──validate──► "desired" ──diff(G0,G2)──► ExecutionPlan ──X
//! ```
//!
//! OGAR owns the meaning (`ogar-dir-sim`: `Change`, provenance, `Violation`,
//! `ExecutionPlan`); this crate owns execution:
//!
//! * [`snapshot`] — one observation as two populations (users, groups) of
//!   SoA lanes and bit planes, each at most 65,536 nodes with its own `u16`
//!   ordinal space; sparse `user × group` membership; [`Dn128`] hierarchy;
//!   strings only in the store's cold label/value table, as [`ValueId`]s.
//! * [`view`] — a version = shared `Arc<Snapshot>` + delta-sized overlay.
//! * [`rule`] — pure population rules over a borrowed [`View`].
//! * [`validate`] — invariants as Quack programs (`Semijoin` anti-joins,
//!   `GroupReduce` counts).
//! * [`store`] — append-only versions, tags, diff, plan, audit.
//! * [`observe`] — `ogar-ad` records → observation (`OuHhtl` → `Dn128`,
//!   failing closed).
//!
//! No network, process or file I/O. Nothing writes to AD, Entra, Exchange,
//! LDAP or PowerShell. `#![forbid(unsafe_code)]`.

#![forbid(unsafe_code)]

mod exec;
pub mod observe;
pub mod rule;
pub mod snapshot;
pub mod store;
pub mod validate;
pub mod view;

pub use exec::Kept;
pub use ogar_dir_sim::{KeyId, ValueId};
pub use rule::{member_counts, GrantGroup, ImplyGroup, Rule, SetPrimarySmtp};
pub use snapshot::{
    BuildError, Dict, Dicts, GroupOrdinal, NodeKind, Observation, ObservedNode, Population,
    Snapshot, UserOrdinal, MAX_GROUPS, MAX_USERS, NONE,
};
pub use store::{Rejection, SimError, VersionStore};
pub use view::{ApplyError, View};

use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Planes, StridedRef};
use lance_graph_quack::{Cmp, Col, Filter, Mask};
use ogar_dir_core::{DirectoryScope, Dn128};

/// A subtree query in a directory the version does not describe. Codes are
/// only meaningful under their scope, so the query is refused, not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScopeMismatch {
    /// The version's scope.
    pub version: DirectoryScope,
    /// The query's scope.
    pub query: DirectoryScope,
}

/// Nodes of `kind` located in the subtree of `prefix` (ancestor-or-self),
/// as a bitmap over that population's ordinals.
///
/// One program, no strings: `located ∧ depth ≥ d ∧ dn ⊇ prefix`, where the
/// last term is a 16-byte ternary match (`Cmp::MatchFacet16Strided`) read
/// in place over the population's `[[u8; 16]]` lane. The depth gate is what
/// keeps a shallower node whose zero tail happens to equal the prefix out.
/// The same program runs over the base lane (deleted nodes gated out) and
/// over the created nodes' lane (delta-sized); the two kept sets are
/// concatenated, never fed onward.
pub fn subtree(
    v: &View<'_>,
    scope: DirectoryScope,
    kind: NodeKind,
    prefix: &Dn128,
) -> Result<Kept, ScopeMismatch> {
    if scope != v.snap.scope {
        return Err(ScopeMismatch {
            version: v.snap.scope,
            query: scope,
        });
    }
    let (pattern, care) = prefix.subtree_mask();
    let p = exec::program(
        Filter::and([
            Filter::plane(Mask(0)),
            Filter::cmp(Col(0), Cmp::GeI32(prefix.depth() as i32)),
            Filter::cmp(Col(1), Cmp::MatchFacet16Strided { pattern, care }),
        ]),
        lance_graph_quack::Agg::Rows,
    );
    let run = |present: &[u64], depth: &[i32], dn: &[[u8; 16]]| {
        let lanes = [
            LaneRef::I32(depth),
            LaneRef::Strided(StridedRef {
                bytes: dn.as_flattened(),
                first_offset: 0,
                stride: 16,
                records: dn.len(),
            }),
        ];
        let masks: [&[u64]; 1] = [present];
        exec::keep(
            &p,
            &Planes {
                n_rows: dn.len(),
                masks: &masks,
                lanes: &lanes,
            },
            &Foreign::NONE,
        )
    };
    let (pop, ov) = v.pop(kind);
    let present = v.base_live(kind, &pop.dn_present);
    let base = run(&present, &pop.dn_depth, &pop.dn);

    let cr = &ov.created.dn;
    let mut located = vec![0u64; words_for(cr.len())];
    let (mut depth, mut dn) = (Vec::with_capacity(cr.len()), Vec::with_capacity(cr.len()));
    for (i, d) in cr.iter().enumerate() {
        if d.is_some() {
            located[i / 64] |= 1 << (i % 64);
        }
        let d = d.unwrap_or(Dn128::ROOT);
        depth.push(d.depth() as i32);
        dn.push(d.bytes());
    }
    Ok(base.concat(&run(&located, &depth, &dn)))
}
