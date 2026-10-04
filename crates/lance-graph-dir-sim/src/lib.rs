//! # lance-graph-dir-sim — explore a directory future over the SoA substrate
//!
//! ```text
//! observed G0 ──rule──► G1 ──rule──► G2 ──validate──► "desired" ──diff(G0,G2)──► ExecutionPlan ──X
//! ```
//!
//! OGAR owns the meaning (`ogar-dir-sim`: `Change`, provenance, `Violation`,
//! `ExecutionPlan`); this crate owns execution:
//!
//! * [`snapshot`] — one observation as SoA lanes and bit planes, `Guid128`
//!   sorted so the ordinal is the index; strings in store dictionaries.
//! * [`view`] — a version = shared `Arc<Snapshot>` + delta-sized overlay.
//! * [`rule`] — pure population rules over a borrowed [`View`].
//! * [`validate`] — invariants as Quack programs (`Semijoin` anti-joins,
//!   `GroupReduce` counts).
//! * [`store`] — append-only versions, tags, diff, plan, audit.
//! * [`observe`] — `ogar-ad` records → observation.
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
pub use rule::{member_counts, GrantGroup, ImplyGroup, Rule, SetPrimarySmtp};
pub use snapshot::{
    pack_ou, BuildError, Dict, Dicts, NodeKind, Observation, ObservedNode, Snapshot, NONE,
};
pub use store::{Rejection, SimError, VersionStore};
pub use view::{ApplyError, View};

use lance_graph_mask_risc::{Foreign, LaneRef, Planes};
use lance_graph_quack::{Cmp, Col, Filter, Mask};
use ogar_dir_core::OuHhtl;

/// A subtree prefix deeper than the packed lane (4 levels) can address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubtreeTooDeep(pub usize);

/// Nodes located in the OU subtree `prefix` (ancestor-or-self), as a node
/// bitmap over the version's ordinals — one ternary match on the packed OU
/// lane (`Cmp::MatchU64`), no DN strings. The same program runs over the
/// base lane (deleted nodes gated out) and over the created nodes' lane
/// (delta-sized); the two kept sets are concatenated, never fed onward.
/// Prefixes up to depth 4 are exact; deeper ones are refused.
pub fn subtree(v: &View<'_>, prefix: &OuHhtl) -> Result<exec::Kept, SubtreeTooDeep> {
    let d = prefix.depth();
    if d > 4 {
        return Err(SubtreeTooDeep(d));
    }
    let care = if d == 0 { 0 } else { u64::MAX << (64 - 16 * d) };
    let p = exec::program(
        Filter::and([
            Filter::plane(Mask(0)),
            Filter::cmp(
                Col(0),
                Cmp::MatchU64 {
                    pattern: pack_ou(prefix),
                    care,
                },
            ),
        ]),
        lance_graph_quack::Agg::Rows,
    );
    let s = v.snap;
    let present = v.base_live(&s.ou_present);
    let lanes = [LaneRef::U64(&s.ou_hi)];
    let masks: [&[u64]; 1] = [&present];
    let base = exec::keep(
        &p,
        &Planes {
            n_rows: s.len(),
            masks: &masks,
            lanes: &lanes,
        },
        &Foreign::NONE,
    );
    let cr = &v.ov.created;
    let packed: Vec<u64> = cr
        .ou
        .iter()
        .map(|o| o.as_ref().map_or(0, pack_ou))
        .collect();
    let mut located = vec![0u64; lance_graph_mask_risc::words_for(cr.ou.len())];
    for (i, o) in cr.ou.iter().enumerate() {
        if o.is_some() {
            located[i / 64] |= 1 << (i % 64);
        }
    }
    let lanes = [LaneRef::U64(&packed)];
    let masks: [&[u64]; 1] = [&located];
    let created = exec::keep(
        &p,
        &Planes {
            n_rows: cr.ou.len(),
            masks: &masks,
            lanes: &lanes,
        },
        &Foreign::NONE,
    );
    Ok(base.concat(&created))
}
