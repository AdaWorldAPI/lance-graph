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
/// bitmap — one ternary match on the packed OU lane (`Cmp::MatchU64`), no DN
/// strings. Prefixes up to depth 4 are exact; deeper ones are refused.
pub fn subtree(v: &View<'_>, prefix: &OuHhtl) -> Result<Vec<u64>, SubtreeTooDeep> {
    let d = prefix.depth();
    if d > 4 {
        return Err(SubtreeTooDeep(d));
    }
    let care = if d == 0 { 0 } else { u64::MAX << (64 - 16 * d) };
    let s = v.snap;
    let lanes = [LaneRef::U64(&s.ou_hi)];
    let masks: [&[u64]; 1] = [&s.ou_present];
    let planes = Planes {
        n_rows: s.len(),
        masks: &masks,
        lanes: &lanes,
    };
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
    Ok(exec::keep(&p, &planes, &Foreign::NONE))
}
