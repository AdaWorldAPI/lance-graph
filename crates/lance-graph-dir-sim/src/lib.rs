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

pub mod bind;
mod exec;
pub mod observe;
pub mod proxy;
pub mod rule;
pub mod snapshot;
pub mod store;
pub mod validate;
pub mod view;

pub use bind::{where_eq, UserBinder, WhereEqError};
pub use exec::Kept;
pub use ogar_dir_sim::{KeyId, ValueId};
pub use proxy::{ProxyKind, ProxyRelation, ProxyRow};
pub use rule::{member_counts, GrantGroup, ImplyGroup, Rule, SetPrimarySmtp};
pub use snapshot::{
    BuildError, Dict, DictCounters, Dicts, GroupOrdinal, NodeKind, Observation, ObservedNode,
    Population, Snapshot, UserOrdinal, MAX_GROUPS, MAX_USERS, NONE,
};
pub use store::{Rejection, SimError, VersionStore};
pub use view::{ApplyError, View};

use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Planes, Program, StridedRef};
use lance_graph_quack::{Cmp, Col, Filter, Mask};
use ogar_dir_core::{DirectoryScope, Dn128};
use ogar_dir_sim::Attribute;

/// V4 ("active" across AD and Entra) as one Quack filter over four resident
/// planes: each source's known (validity) plane and its enabled plane.
/// Keeps exactly the rows [`ogar_dir_sim::effective_active`] calls
/// `Some(true)`:
///
/// ```text
/// (ad_known ∨ entra_known) ∧ (¬ad_known ∨ ad_enabled) ∧ (¬entra_known ∨ entra_enabled)
/// ```
///
/// An enabled bit outside its known plane is ignored, so a source's stale
/// payload cannot vote. Unknown rows are kept by neither this filter nor
/// [`effectively_inactive`].
pub fn effectively_active(
    ad_known: Mask,
    ad_enabled: Mask,
    entra_known: Mask,
    entra_enabled: Mask,
) -> Filter {
    let p = Filter::plane;
    Filter::and([
        Filter::or([p(ad_known), p(entra_known)]),
        Filter::or([Filter::negate(p(ad_known)), p(ad_enabled)]),
        Filter::or([Filter::negate(p(entra_known)), p(entra_enabled)]),
    ])
}

/// The rows [`ogar_dir_sim::effective_active`] calls `Some(false)`: some
/// known source says disabled.
pub fn effectively_inactive(
    ad_known: Mask,
    ad_enabled: Mask,
    entra_known: Mask,
    entra_enabled: Mask,
) -> Filter {
    let p = Filter::plane;
    Filter::or([
        Filter::and([p(ad_known), Filter::negate(p(ad_enabled))]),
        Filter::and([p(entra_known), Filter::negate(p(entra_enabled))]),
    ])
}

/// The program behind [`users_with_key`]: plane 0 = candidate users,
/// lane 0 = an attribute's key lane, one `EqU32` on the key. It holds only
/// numbers: the literal was resolved before it was built.
pub fn key_eq_program(key: KeyId) -> Program {
    exec::program(
        Filter::and([
            Filter::plane(Mask(0)),
            Filter::cmp(Col(0), Cmp::EqU32(key.0)),
        ]),
        lance_graph_quack::Agg::Rows,
    )
}

/// `WHERE <attribute> = <literal>` over the existing users of a version, as
/// a bitmap over user ordinals. The literal arrives as a [`KeyId`] —
/// resolved once at the boundary ([`Dicts::key_lookup`]) — so execution is
/// one [`key_eq_program`] over the base key lane (overridden and deleted
/// users gated out), the same program over the created users' key lane,
/// and a delta-sized check of the overrides. No string is read.
///
/// The text form is [`where_eq`], which binds a field name and a literal to
/// this same program.
pub fn users_with_key(v: &View<'_>, a: Attribute, key: KeyId) -> Kept {
    users_matching(v, a, key, &key_eq_program(key))
}

/// The executor behind [`users_with_key`] and [`where_eq`]: run `p` (plane 0
/// = live users, lane 0 = `a`'s key lane) over the base and the created
/// users, and merge the overrides of `a` by `key`.
pub(crate) fn users_matching(v: &View<'_>, a: Attribute, key: KeyId, p: &Program) -> Kept {
    let s = v.snap;
    let (pop, ov) = v.pop(NodeKind::User);
    let base_key = match a {
        Attribute::Upn => &pop.upn_key,
        Attribute::PrimarySmtp => &pop.smtp_key,
    };
    let mut live = v.base_live(NodeKind::User, &pop.all).into_owned();
    for o in ov.overrides(a).keys() {
        snapshot::clear_bit(&mut live, usize::from(*o));
    }
    let run = |plane: &[u64], lane: &[u32]| {
        let lanes = [LaneRef::U32(lane)];
        let masks: [&[u64]; 1] = [plane];
        exec::keep(
            p,
            &Planes {
                n_rows: lane.len(),
                masks: &masks,
                lanes: &lanes,
            },
            &Foreign::NONE,
        )
    };
    let mut base = run(&live, base_key);
    for (o, (_, k)) in ov.overrides(a) {
        if *k == key.0 {
            base.set(usize::from(*o));
        }
    }
    debug_assert_eq!(base_key.len(), s.users.len());
    let created = ov.created.key(a);
    base.concat(&run(&snapshot::ones(created.len()), created))
}

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
