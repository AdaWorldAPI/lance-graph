//! Read-only hybrid-coordinate lab. NO AD/Graph/Exchange I/O or production admission.
//!
//! Existing OGAR Index + fold owns the cross-source match. The lab binds
//! snapshot-local ordinals once, then reads two unrelated source-property
//! lanes through the same tiny descriptor. No joined-row/PSCustomObject store.
//!
//! Coordinate lifetime is witnessed by the existing DT64 DR `FrameStore`:
//! a bound coordinate carries the owner-issued `Frame` it was bound under
//! (source / relation-map / coordinate revisions), and every read checks it.
//! A frame cannot be built from numbers, so two snapshots that happen to share
//! revision numbers never alias.
//!
//! Not proved here: a production DR MapTable-to-SoA layout witness, a general
//! FACET STRIDE GUID join, an admitted arbitrary cross-table Gather, or full
//! historic data replay. See the companion PR description.

#![deny(unsafe_code)]

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Frame, FrameRefusal, FrameRevisions, FrameStore};
use lance_graph_dt64::{
    BindStore, ClassSchema, Extent, MethodBinding, MethodId, MethodRef, Refusal, SchemaRev, SigRev,
    ValueDemand,
};
use lance_graph_mask_risc::{
    execute_into, Foreign, LaneRef, Out, Planes as QueryPlanes, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Col, Filter, Mask, Query};
use ogar_dir_core::correspond::{
    fold, fold_aligned, state, AdStatus, IdColumn, Index, Lanes, Output, Planes, Rows, Ruler,
    NO_ROW,
};
use ogar_dir_core::identity::Encoding;
use ogar_dir_core::label::LabelPattern;
use ogar_dir_core::Guid128;

/// Counts heap allocations made by the current thread, so the hot-path test
/// is not disturbed by tests running in parallel.
struct CountingAlloc;

thread_local! {
    static ALLOCATIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[allow(unsafe_code)]
// SAFETY: delegates every call to the system allocator unchanged.
unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
        unsafe { std::alloc::System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn allocations() -> u64 {
    ALLOCATIONS.with(|c| c.get())
}

fn id(x: u128) -> Guid128 {
    Guid128(x.to_be_bytes())
}

fn known(ids: &[u128]) -> IdColumn {
    IdColumn {
        id: ids.iter().copied().map(id).collect(),
        state: vec![state::PRESENT; ids.len()],
    }
}

/// The synthetic input is *already decoded* at the OGAR boundary.
/// AD objectGUIDs (10,11) differ from cloud object ids (30,31).
/// Source anchors (20,21) are a third, independently matched identity.
fn snapshot() -> Lanes {
    Lanes {
        ad: Rows {
            owner: vec![id(10), id(11)],
            scope: vec![0, 0],
            at: vec![1_000, 1_000],
        },
        ad_anchor: known(&[20, 21]),
        ad_backsync: known(&[30, 31]),
        entra: Rows {
            owner: vec![id(30), id(31)],
            scope: vec![0, 0],
            at: vec![1_100, 1_100],
        },
        entra_anchor: known(&[20, 21]),
        exo: Rows {
            owner: vec![id(40), id(41)],
            scope: vec![0, 0],
            at: vec![1_100, 1_100],
        },
        exo_external: known(&[30, 31]),
        anchor_is_object_guid: false,
        max_skew_ms: 60_000,
    }
}

/// Revisions of the snapshot the lab binds against: `source` = the
/// observation, `relation_map` = the OGAR correspondence run over it,
/// `coordinates` = the row order of the borrowed lanes.
fn revisions(source: u64, relation_map: u64, coordinates: u64) -> FrameRevisions {
    FrameRevisions {
        source,
        relation_map,
        coordinates,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadError {
    /// The owner no longer issues the frame the coordinate was bound under.
    StaleFrame(FrameRefusal),
    /// The frame is current but the lanes do not hold the bound identities:
    /// the owner forgot to advance. Defence in depth, not the witness.
    StaleCoordinate,
    MissingProperty,
}

#[derive(Clone, Copy, Debug)]
enum Family {
    Ad,
    Entra,
    Exo,
}

/// A conceptual chained-coordinate result, not a new canonical object type.
/// Each ordinal is valid only for its owning population/scope/layout.
#[derive(Clone, Debug)]
struct BoundCoordinates {
    ad: usize,
    entra: usize,
    exo: usize,
    scope: u32,
    ad_id: Guid128,
    entra_id: Guid128,
    exo_id: Guid128,
    frame: Frame,
}

impl BoundCoordinates {
    /// The only hot read: check that the caller's existing borrowed lanes
    /// still match the bound identities/scope and index a demanded property.
    /// No GUID matching, string decoding or joined-row construction here.
    fn read(
        &self,
        l: &Lanes,
        frames: &FrameStore,
        family: Family,
        values: &[i32],
    ) -> Result<i32, ReadError> {
        frames.check(&self.frame).map_err(ReadError::StaleFrame)?;
        if l.ad.owner.get(self.ad) != Some(&self.ad_id)
            || l.entra.owner.get(self.entra) != Some(&self.entra_id)
            || l.exo.owner.get(self.exo) != Some(&self.exo_id)
            || l.ad.scope.get(self.ad) != Some(&self.scope)
            || l.entra.scope.get(self.entra) != Some(&self.scope)
            || l.exo.scope.get(self.exo) != Some(&self.scope)
            || l.ad_backsync.id.get(self.ad) != Some(&self.entra_id)
            || l.exo_external.id.get(self.exo) != Some(&self.entra_id)
        {
            return Err(ReadError::StaleCoordinate);
        }
        let row = match family {
            Family::Ad => self.ad,
            Family::Entra => self.entra,
            Family::Exo => self.exo,
        };
        values.get(row).copied().ok_or(ReadError::MissingProperty)
    }
}

/// Existing OGAR cold Index/fold and same-slot Ruler/Planes are the
/// correspondence authority. This helper adds *no* new hot join engine.
struct Prepared {
    out: Output,
    planes: Planes,
}

impl Prepared {
    fn new(l: &Lanes) -> Self {
        let index = Index::build(l);
        let mut out = Output::for_index(l, &index);
        fold(l, &index, &mut out);
        let ruler = Ruler::from_fold(l, &index, &out);
        let mut planes = Planes::for_ruler(&ruler);
        fold_aligned(&ruler, &mut planes);
        Self { out, planes }
    }

    /// Resolve once at the cold boundary. Every ordinal comes from OGAR's
    /// fold: AD→Entra from `ad_target`/`ad_back`, Entra→EXO from
    /// `entra_mailbox`. The lab performs no lookup of its own.
    fn bind(&self, l: &Lanes, ad: usize, frames: &FrameStore) -> Option<BoundCoordinates> {
        let status = AdStatus::of(*self.out.ad.get(ad)?);
        if status != AdStatus::Confirmed {
            return None;
        }
        let entra = usize::try_from(*self.out.ad_target.get(ad)?).ok()?;
        if entra == NO_ROW as usize || *self.out.ad_back.get(ad)? != entra as u32 {
            return None;
        }
        let confirmed = self.planes.confirmed(ad / 64) >> (ad % 64) & 1;
        let cloud = self.planes.cloud[ad / 64] >> (ad % 64) & 1;
        if confirmed != 1 || cloud != 1 {
            return None;
        }

        let scope = *l.ad.scope.get(ad)?;
        if l.entra.scope.get(entra) != Some(&scope) {
            return None;
        }
        let cloud_id = *l.entra.owner.get(entra)?;
        let exo = match *self.out.entra_mailbox.get(entra)? {
            NO_ROW => return None,
            x => x as usize,
        };
        Some(BoundCoordinates {
            ad,
            entra,
            exo,
            scope,
            ad_id: l.ad.owner[ad],
            entra_id: cloud_id,
            exo_id: l.exo.owner[exo],
            frame: frames.seal(),
        })
    }
}

/// Independently specified scalar reference: no OGAR Index, Output, Ruler,
/// Planes, local-candidate index or DR position selection.
fn oracle(l: &Lanes, ad: usize) -> Option<(usize, usize)> {
    let scope = *l.ad.scope.get(ad)?;
    let anchor = *l.ad_anchor.id.get(ad)?;
    let backsync = *l.ad_backsync.id.get(ad)?;
    if l.ad_anchor.state[ad] != state::PRESENT || l.ad_backsync.state[ad] != state::PRESENT {
        return None;
    }
    let mut matching = (0..l.entra.len()).filter(|&j| {
        l.entra.scope[j] == scope
            && l.entra_anchor.state[j] == state::PRESENT
            && l.entra_anchor.id[j] == anchor
            && l.entra.owner[j] == backsync
            && l.entra.at[j] != 0
            && l.ad.at[ad] != 0
            && (l.ad.at[ad] - l.entra.at[j]).abs() <= l.max_skew_ms
    });
    let entra = matching.next()?;
    if matching.next().is_some() {
        return None;
    }
    // A different AD source must not claim the same anchor or backsync.
    if (0..l.ad.len()).any(|i| {
        i != ad
            && l.ad.scope[i] == scope
            && ((l.ad_anchor.state[i] == state::PRESENT && l.ad_anchor.id[i] == anchor)
                || (l.ad_backsync.state[i] == state::PRESENT && l.ad_backsync.id[i] == backsync))
    }) {
        return None;
    }
    let mut mailbox = (0..l.exo.len()).filter(|&x| {
        l.exo.scope[x] == scope
            && l.exo_external.state[x] == state::PRESENT
            && l.exo_external.id[x] == backsync
    });
    let exo = mailbox.next()?;
    mailbox.next().is_none().then_some((entra, exo))
}

#[test]
fn backsync_label_is_decoded_to_128_bit_cloud_identity_once() {
    let cloud = id(30);
    let wire = format!("User_{cloud}");
    let decoded = Encoding::Labeled(LabelPattern("User_{0}"))
        .decode(wire.as_bytes())
        .unwrap();
    assert_eq!(decoded, cloud);
    assert_ne!(decoded, id(10)); // the AD objectGUID
    assert_ne!(decoded, id(20)); // the source anchor
    assert!(Encoding::Labeled(LabelPattern("User_{0}"))
        .decode(b"User_not-a-guid")
        .is_err());
}

#[test]
fn a_single_bound_coordinate_serves_many_late_properties() {
    let l = snapshot();
    let frames = FrameStore::new(revisions(1, 1, 1));
    let prepared = Prepared::new(&l);
    let r = prepared
        .bind(&l, 0, &frames)
        .expect("confirmed AD→Entra→EXO");
    assert_eq!((r.entra, r.exo), oracle(&l, 0).unwrap());

    // Three separate source-owned property lanes. No Vec<JoinedRow>.
    let ad_department = [6_i32, 7];
    let entra_license_count = [25_i32, 40];
    let exo_archive_state = [1_i32, 0];
    assert_eq!(r.read(&l, &frames, Family::Ad, &ad_department), Ok(6));
    assert_eq!(
        r.read(&l, &frames, Family::Entra, &entra_license_count),
        Ok(25)
    );
    assert_eq!(r.read(&l, &frames, Family::Exo, &exo_archive_state), Ok(1));
    assert_eq!(
        r.read(&l, &frames, Family::Exo, &[]),
        Err(ReadError::MissingProperty)
    );
}

/// Every ordinal of every bindable row equals the independent scan. A
/// non-vacuous fixture: both rows bind, so the comparison is made twice.
#[test]
fn bound_ordinals_equal_the_independent_oracle_for_every_row() {
    let l = snapshot();
    let frames = FrameStore::new(revisions(1, 1, 1));
    let prepared = Prepared::new(&l);
    let mut bound = 0;
    for ad in 0..l.ad.len() {
        let got = prepared.bind(&l, ad, &frames).map(|r| (r.entra, r.exo));
        assert_eq!(got, oracle(&l, ad), "AD row {ad}");
        bound += usize::from(got.is_some());
    }
    assert_eq!(bound, l.ad.len());
}

#[test]
fn moved_ordinals_refuse_stale_read_then_rebind_to_same_identity() {
    let l = snapshot();
    let mut frames = FrameStore::new(revisions(1, 1, 1));
    let old = Prepared::new(&l).bind(&l, 0, &frames).unwrap();
    let mut moved = l.clone();
    moved.entra.owner.swap(0, 1);
    moved.entra.scope.swap(0, 1);
    moved.entra.at.swap(0, 1);
    moved.entra_anchor.id.swap(0, 1);
    moved.entra_anchor.state.swap(0, 1);

    // The owner did not advance: the frame is current, the identity
    // cross-check is what refuses.
    assert_eq!(
        old.read(&moved, &frames, Family::Entra, &[40, 25]),
        Err(ReadError::StaleCoordinate)
    );
    // The owner advances the coordinate revision: the frame refuses first,
    // and names what moved.
    frames.advance(revisions(1, 1, 2));
    assert_eq!(
        old.read(&moved, &frames, Family::Entra, &[40, 25]),
        Err(ReadError::StaleFrame(FrameRefusal::CoordinatesChanged))
    );
    let rebound = Prepared::new(&moved).bind(&moved, 0, &frames).unwrap();
    assert_eq!(rebound.entra, 1);
    assert_eq!((rebound.entra, rebound.exo), oracle(&moved, 0).unwrap());
    assert_eq!(
        rebound.read(&moved, &frames, Family::Entra, &[40, 25]),
        Ok(25)
    );
}

/// The lab's former value stamp aliased two snapshots that shared revision
/// numbers. An owner-issued frame does not: equal numbers, different owner,
/// refused; equal numbers re-sealed by the same owner, refused.
#[test]
fn a_frame_from_another_owner_or_a_reseal_never_aliases() {
    let l = snapshot();
    let frames = FrameStore::new(revisions(1, 1, 1));
    let r = Prepared::new(&l).bind(&l, 0, &frames).unwrap();

    let twin = FrameStore::new(revisions(1, 1, 1));
    assert_eq!(
        r.read(&l, &twin, Family::Ad, &[6, 7]),
        Err(ReadError::StaleFrame(FrameRefusal::ForeignFrame))
    );

    let mut frames = frames;
    frames.advance(revisions(1, 1, 1));
    assert_eq!(
        r.read(&l, &frames, Family::Ad, &[6, 7]),
        Err(ReadError::StaleFrame(FrameRefusal::FrameResealed))
    );
}

/// A source change that makes a unique correspondence ambiguous: the old
/// coordinate is refused by its frame, and rebinding refuses rather than
/// picking the first mailbox.
#[test]
fn a_unique_match_turned_ambiguous_is_refused_not_reused() {
    let l = snapshot();
    let mut frames = FrameStore::new(revisions(1, 1, 1));
    let old = Prepared::new(&l).bind(&l, 0, &frames).unwrap();

    let mut next = l.clone();
    next.exo.owner.push(id(42));
    next.exo.scope.push(0);
    next.exo.at.push(1_100);
    next.exo_external.id.push(id(30));
    next.exo_external.state.push(state::PRESENT);
    frames.advance(revisions(2, 2, 1));

    assert_eq!(
        old.read(&next, &frames, Family::Exo, &[1, 0, 1]),
        Err(ReadError::StaleFrame(FrameRefusal::SourceChanged))
    );
    assert!(Prepared::new(&next).bind(&next, 0, &frames).is_none());
    assert_eq!(oracle(&next, 0), None);
    // The other identity is untouched by the new mailbox and still binds.
    assert_eq!(
        Prepared::new(&next)
            .bind(&next, 1, &frames)
            .map(|r| (r.entra, r.exo)),
        oracle(&next, 1)
    );
}

#[test]
fn duplicate_cloud_mailbox_and_foreign_scope_never_choose_first() {
    let frames = FrameStore::new(revisions(1, 1, 1));
    let mut l = snapshot();
    l.exo.owner.push(id(42));
    l.exo.scope.push(0);
    l.exo.at.push(1_100);
    l.exo_external.id.push(id(30));
    l.exo_external.state.push(state::PRESENT);
    assert!(Prepared::new(&l).bind(&l, 0, &frames).is_none());

    // Same raw Entra ID in another tenant is NOT a match.
    let mut l = snapshot();
    l.exo.scope[0] = 1;
    assert!(Prepared::new(&l).bind(&l, 0, &frames).is_none());

    let mut l = snapshot();
    l.entra_anchor.id[0] = id(999); // contradicts AD source anchor
    assert!(Prepared::new(&l).bind(&l, 0, &frames).is_none());

    // Missing backsync: the anchor alone is not a confirmed hybrid identity.
    // The status says so explicitly; the bind refuses; the oracle agrees.
    let mut l = snapshot();
    l.ad_backsync.state[0] = state::ABSENT;
    let p = Prepared::new(&l);
    assert_ne!(AdStatus::of(p.out.ad[0]), AdStatus::Confirmed);
    assert!(p.bind(&l, 0, &frames).is_none());
    assert_eq!(oracle(&l, 0), None);
}

#[test]
fn dt64_method_pointer_is_not_a_coordinate_or_an_excel_value() {
    let class = 0x1100_0001;
    let schema = ClassSchema {
        schema_rev: SchemaRev(1),
        field_count: 2,
        extent: Extent::DR64_CUBE,
    };
    let method = MethodBinding {
        id: MethodId(7),
        sig_rev: SigRev(1),
        name: "lookup_hybrid_coordinate".into(),
        on: WideFieldMask::EMPTY,
        when: None,
        reads: WideFieldMask::from_positions(&[0]),
        effects: Vec::new(),
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::MapOnly,
    };
    let mut dt = BindStore::new();
    dt.register(class, schema, vec![method.clone()]).unwrap();
    let pointer = MethodRef::of(class, &method);
    assert_eq!(dt.admit(&pointer).unwrap(), &method);
    let before = dt.route(0, class).unwrap();
    assert_eq!(dt.routed_at(before, 0).unwrap(), Some(class));

    let mut changed = method.clone();
    changed.sig_rev = SigRev(2);
    assert_eq!(
        dt.register(class, schema, vec![changed.clone()]),
        Err(Refusal::BehaviourImmutable(class))
    );
    let incompatible_pointer = MethodRef {
        sig_rev: SigRev(2),
        ..pointer
    };
    assert!(matches!(
        dt.admit(&incompatible_pointer),
        Err(Refusal::SignatureRevisionMismatch { .. })
    ));

    let derived = class + 1;
    dt.register_derived(class, derived, schema, vec![changed])
        .unwrap();
    let after = dt.route(0, derived).unwrap();
    assert_eq!(dt.routed_at(before, 0).unwrap(), Some(class));
    assert_eq!(dt.routed_at(after, 0).unwrap(), Some(derived));
}

#[test]
fn existing_quack_mask_risc_fold_uses_borrowed_property_lane() {
    let l = snapshot();
    let frames = FrameStore::new(revisions(1, 1, 1));
    assert_eq!(
        Prepared::new(&l).bind(&l, 0, &frames).map(|r| r.entra),
        Some(0)
    );
    assert_eq!(
        Prepared::new(&l).bind(&l, 1, &frames).map(|r| r.entra),
        Some(1)
    );

    let values = [25_i32, 40];
    let live = [0b11_u64]; // one resident, source-owned mask plane
    let masks: [&[u64]; 1] = [&live];
    let lanes = [LaneRef::I32(&values)];
    let inputs = QueryPlanes {
        n_rows: values.len(),
        masks: &masks,
        lanes: &lanes,
    };
    let query = Query {
        filter: Filter::plane(Mask(0)),
        agg: Agg::SumI32(Col(0)),
    };
    let p = lower(&query).unwrap();
    let mut scratch = Scratch::for_program(&p, values.len()).unwrap();
    let result = execute_into(&p, &inputs, &Foreign::NONE, &mut scratch, Out::None).unwrap();
    assert_eq!(result, Value::SumI64(65));
    let independent_scalar: i64 = values.iter().map(|&v| i64::from(v)).sum();
    assert_eq!(result, Value::SumI64(independent_scalar));
}

/// Execution evidence. Cold (index, fold, ruler, bind) and hot (reads, typed
/// fold) are timed separately; the hot side performs zero heap allocations on
/// this thread, i.e. no per-row joined object is built. Timings are reported,
/// not gated: the fixture is two rows.
#[test]
fn hot_reads_and_fold_allocate_nothing() {
    use std::time::Instant;
    let l = snapshot();
    let frames = FrameStore::new(revisions(1, 1, 1));

    let t = Instant::now();
    let prepared = Prepared::new(&l);
    let r = prepared.bind(&l, 0, &frames).unwrap();
    let cold = t.elapsed();

    let ad_department = [6_i32, 7];
    let entra_license_count = [25_i32, 40];
    let exo_archive_state = [1_i32, 0];
    let live = [0b11_u64];
    let masks: [&[u64]; 1] = [&live];
    let lanes = [LaneRef::I32(&entra_license_count)];
    let inputs = QueryPlanes {
        n_rows: entra_license_count.len(),
        masks: &masks,
        lanes: &lanes,
    };
    let query = Query {
        filter: Filter::plane(Mask(0)),
        agg: Agg::SumI32(Col(0)),
    };
    let p = lower(&query).unwrap();
    let mut scratch = Scratch::for_program(&p, entra_license_count.len()).unwrap();

    const N: u32 = 10_000;
    let before = allocations();
    let t = Instant::now();
    let mut acc = 0_i64;
    for _ in 0..N {
        acc += i64::from(r.read(&l, &frames, Family::Ad, &ad_department).unwrap());
        acc += i64::from(
            r.read(&l, &frames, Family::Entra, &entra_license_count)
                .unwrap(),
        );
        acc += i64::from(
            r.read(&l, &frames, Family::Exo, &exo_archive_state)
                .unwrap(),
        );
    }
    let reads = t.elapsed();
    let t = Instant::now();
    let mut folded = Value::SumI64(0);
    for _ in 0..N {
        folded = execute_into(&p, &inputs, &Foreign::NONE, &mut scratch, Out::None).unwrap();
    }
    let fold_time = t.elapsed();
    let hot_allocations = allocations() - before;

    assert_eq!(acc, i64::from(N) * (6 + 25 + 1));
    assert_eq!(folded, Value::SumI64(65));
    assert_eq!(hot_allocations, 0, "the hot path allocated");
    eprintln!(
        "cold prepare+bind {} ns | read {} ns/property | fold {} ns/call | hot allocations {}",
        cold.as_nanos(),
        reads.as_nanos() / u128::from(3 * N),
        fold_time.as_nanos() / u128::from(N),
        hot_allocations
    );
}

// ---- batch selection over the existing Ruler planes -------------------------
//
// The Ruler (`ogar_dir_core::correspond::Ruler::from_fold`) is OGAR's aligned
// identity-witness view: one slot per AD row, Entra/EXO witnesses placed only
// for an uncontested counterpart. `fold_aligned` turns it into six 64-slot
// mask planes. The tests below select over those planes with one lowered
// Quack program, 64 slots per word, and never bind or visit a row on the hot
// path. Only identity witnesses are aligned; no Entra/EXO property value is
// copied into slot order, and none is read here.

/// Plane order handed to Quack as `Mask(0..6)`.
const P_FORWARD: u16 = 0;
const P_BACKSYNC: u16 = 1;
const P_CLOUD: u16 = 2;
const P_ALIGNED: u16 = 3;
const P_FWD_CONFLICT: u16 = 4;
const P_BACK_CONFLICT: u16 = 5;

fn plane_refs(p: &Planes) -> [&[u64]; 6] {
    [
        &p.forward,
        &p.backsync,
        &p.cloud,
        &p.aligned,
        &p.forward_conflict,
        &p.backsync_conflict,
    ]
}

/// A confirmed hybrid identity with exactly one mailbox: forward ∧ backsync ∧
/// aligned (OGAR's `Planes::confirmed`) ∧ cloud.
fn confirmed_hybrid() -> Filter {
    Filter::and([
        Filter::plane(Mask(P_FORWARD)),
        Filter::plane(Mask(P_BACKSYNC)),
        Filter::plane(Mask(P_ALIGNED)),
        Filter::plane(Mask(P_CLOUD)),
    ])
}

/// dir-sim's existing `CloudMailboxes::from_fold` predicate, as a program:
/// cloud ∧ ¬forward_conflict ∧ ¬backsync_conflict.
fn cloud_mailbox() -> Filter {
    Filter::and([
        Filter::plane(Mask(P_CLOUD)),
        Filter::negate(Filter::plane(Mask(P_FWD_CONFLICT))),
        Filter::negate(Filter::plane(Mask(P_BACK_CONFLICT))),
    ])
}

/// Cold: OGAR's index, fold, ruler and planes over one snapshot.
struct Fold {
    index: Index,
    out: Output,
    planes: Planes,
    n: usize,
}

impl Fold {
    fn new(l: &Lanes) -> Self {
        let index = Index::build(l);
        let mut out = Output::for_index(l, &index);
        fold(l, &index, &mut out);
        let ruler = Ruler::from_fold(l, &index, &out);
        let mut planes = Planes::for_ruler(&ruler);
        fold_aligned(&ruler, &mut planes);
        Self {
            index,
            out,
            planes,
            n: l.ad.len(),
        }
    }
}

/// Hot: one frame check for the whole pass, then the lowered program over the
/// planes. Writes the surviving-slot mask into `kept` and returns the count.
fn select(
    f: &Fold,
    frames: &FrameStore,
    frame: &Frame,
    program: &lance_graph_mask_risc::Program,
    scratch: &mut Scratch,
    kept: &mut [u64],
) -> Result<usize, FrameRefusal> {
    frames.check(frame)?;
    let masks = plane_refs(&f.planes);
    let inputs = QueryPlanes {
        n_rows: f.n,
        masks: &masks,
        lanes: &[],
    };
    execute_into(program, &inputs, &Foreign::NONE, scratch, Out::Mask(kept)).unwrap();
    Ok(kept.iter().map(|w| w.count_ones() as usize).sum())
}

fn bit(words: &[u64], i: usize) -> bool {
    words[i / 64] >> (i % 64) & 1 == 1
}

/// Scalar reference, without OGAR's Index, Output, Ruler or Planes: the
/// lab's `oracle` plus the scope rule OGAR's index applies (an unbound scope
/// never matches, even another unbound one).
fn oracle_bound(l: &Lanes, ad: usize) -> Option<(usize, usize)> {
    use ogar_dir_core::correspond::UNBOUND;
    let (entra, exo) = oracle(l, ad)?;
    (l.ad.scope[ad] != UNBOUND).then_some((entra, exo))
}

/// A seeded population with every fault class the falsifier needs, and Entra/EXO rows shuffled so their
/// positions never coincide with AD slots.
fn faulted(n: usize, seed: u64) -> Lanes {
    use ogar_dir_core::correspond::UNBOUND;
    let mut s = seed | 1;
    let mut rnd = move |m: u64| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s % m
    };
    let mut l = Lanes {
        ad: Rows::default(),
        ad_anchor: IdColumn::default(),
        ad_backsync: IdColumn::default(),
        entra: Rows::default(),
        entra_anchor: IdColumn::default(),
        exo: Rows::default(),
        exo_external: IdColumn::default(),
        anchor_is_object_guid: false,
        max_skew_ms: 60_000,
    };
    let push = |r: &mut Rows, owner: u128, scope: u32, at: i64| {
        r.owner.push(id(owner));
        r.scope.push(scope);
        r.at.push(at);
    };
    let col = |c: &mut IdColumn, v: u128, st: u8| {
        c.id.push(id(v));
        c.state.push(st);
    };
    let mut entra_rows = Vec::new();
    let mut exo_rows = Vec::new();
    for k in 0..n as u128 {
        let (guid, anchor, cloud) = (1_000_000 + k, 2_000_000 + k, 3_000_000 + k);
        let mut ad_scope = 0;
        let mut ad_anchor = anchor;
        let mut backsync = (cloud, state::PRESENT);
        let mut entra = Some((cloud, 0u32, 1_100i64, anchor, state::PRESENT));
        let mut exo: Vec<(u128, u32)> = vec![(cloud, 0)];
        // Ten fault classes in 24: roughly half the slots stay valid once the
        // shared-anchor and contradicting-backsync faults hit a neighbour too.
        match rnd(24) {
            0 => backsync.1 = state::ABSENT,                  // missing backsync
            1 => ad_anchor = 2_000_000 + (k + 1) % n as u128, // shared / foreign anchor
            2 => entra = None,                                // absent correspondence
            3 => exo.clear(),                                 // no mailbox
            4 => exo.push((cloud, 0)),                        // duplicate mailbox
            5 => entra.as_mut().unwrap().1 = 1,               // Entra in another tenant
            6 => exo[0].1 = 1,                                // mailbox in another tenant
            7 => entra.as_mut().unwrap().2 = 900_000,         // stale observation
            8 => backsync.0 = 3_000_000 + (k + 7) % n as u128, // backsync contradicts
            9 => ad_scope = UNBOUND,                          // unbound AD domain
            _ => {}
        }
        push(&mut l.ad, guid, ad_scope, 1_000);
        col(&mut l.ad_anchor, ad_anchor, state::PRESENT);
        col(&mut l.ad_backsync, backsync.0, backsync.1);
        if let Some(e) = entra {
            entra_rows.push(e);
        }
        for (x, (ext, sc)) in exo.into_iter().enumerate() {
            exo_rows.push((4_000_000 + 10 * k + x as u128, sc, ext));
        }
    }
    // Reorder both cloud populations (Fisher-Yates with the same generator).
    for i in (1..entra_rows.len()).rev() {
        entra_rows.swap(i, rnd(i as u64 + 1) as usize);
    }
    for i in (1..exo_rows.len()).rev() {
        exo_rows.swap(i, rnd(i as u64 + 1) as usize);
    }
    for (owner, sc, at, anchor, st) in entra_rows {
        push(&mut l.entra, owner, sc, at);
        col(&mut l.entra_anchor, anchor, st);
    }
    for (owner, sc, ext) in exo_rows {
        push(&mut l.exo, owner, sc, 1_100);
        col(&mut l.exo_external, ext, state::PRESENT);
    }
    l
}

fn lowered(filter: Filter, agg: Agg) -> lance_graph_mask_risc::Program {
    lower(&Query { filter, agg }).unwrap()
}

/// The core falsifier: over several words and every fault class, the slots
/// the program keeps are exactly the AD rows the scalar oracle confirms, and
/// each kept slot's coordinate references (OGAR `Output`) name the oracle's
/// Entra and EXO rows.
#[test]
fn batch_selection_over_ruler_planes_equals_the_scalar_oracle() {
    for seed in [1_u64, 7, 42, 0x9E37_79B9] {
        let l = faulted(300, seed);
        let f = Fold::new(&l);
        let frames = FrameStore::new(revisions(1, 1, 1));
        let frame = frames.seal();
        let p = lowered(confirmed_hybrid(), Agg::Rows);
        let mut scratch = Scratch::for_program(&p, f.n).unwrap();
        let mut kept = vec![0_u64; f.n.div_ceil(64)];
        let count = select(&f, &frames, &frame, &p, &mut scratch, &mut kept).unwrap();

        assert!(kept.len() >= 4, "multi-word population");
        let mut expected = 0;
        for i in 0..f.n {
            let want = oracle_bound(&l, i);
            assert_eq!(bit(&kept, i), want.is_some(), "seed {seed} slot {i}");
            if let Some((entra, exo)) = want {
                expected += 1;
                // The tuple is described by references, never stored.
                let j = f.out.ad_target[i] as usize;
                assert_eq!(j, entra, "seed {seed} slot {i} Entra row");
                assert_eq!(
                    f.out.entra_mailbox[j] as usize, exo,
                    "seed {seed} slot {i} EXO row"
                );
            }
        }
        assert_eq!(count, expected);
        // Anti-vacuity: both outcomes occur in volume.
        assert!(
            count * 4 > f.n && count * 4 < 3 * f.n,
            "seed {seed}: kept {count} of {}",
            f.n
        );
        // The tail of the last word stays clear.
        assert_eq!(kept.last().unwrap() >> (f.n % 64), 0);
    }
}

/// The same program family reproduces dir-sim's existing per-row selection
/// (`CloudMailboxes::from_fold`) as one batch pass.
#[test]
fn batch_selection_reproduces_cloud_mailboxes_from_fold() {
    for seed in [3_u64, 11, 99] {
        let l = faulted(300, seed);
        let f = Fold::new(&l);
        let frames = FrameStore::new(revisions(1, 1, 1));
        let p = lowered(cloud_mailbox(), Agg::Rows);
        let mut scratch = Scratch::for_program(&p, f.n).unwrap();
        let mut kept = vec![0_u64; f.n.div_ceil(64)];
        select(&f, &frames, &frames.seal(), &p, &mut scratch, &mut kept).unwrap();
        let mut ours: Vec<Guid128> = (0..f.n)
            .filter(|&i| bit(&kept, i))
            .map(|i| l.ad.owner[i])
            .collect();
        ours.sort_unstable();
        let existing = lance_graph_dir_sim::CloudMailboxes::from_fold(&l, &f.index, &f.out);
        assert_eq!(ours.as_slice(), existing.owners(), "seed {seed}");
        assert!(!ours.is_empty());
    }
}

/// Reordering every population changes slot positions and row ordinals, never
/// the selected identities.
#[test]
fn reordered_populations_select_the_same_identities() {
    let identities = |l: &Lanes| {
        let f = Fold::new(l);
        let frames = FrameStore::new(revisions(1, 1, 1));
        let p = lowered(confirmed_hybrid(), Agg::Rows);
        let mut scratch = Scratch::for_program(&p, f.n).unwrap();
        let mut kept = vec![0_u64; f.n.div_ceil(64)];
        select(&f, &frames, &frames.seal(), &p, &mut scratch, &mut kept).unwrap();
        let mut ids: Vec<(Guid128, Guid128, Guid128)> = (0..f.n)
            .filter(|&i| bit(&kept, i))
            .map(|i| {
                let j = f.out.ad_target[i] as usize;
                let x = f.out.entra_mailbox[j] as usize;
                (l.ad.owner[i], l.entra.owner[j], l.exo.owner[x])
            })
            .collect();
        ids.sort_unstable();
        ids
    };
    let l = faulted(300, 5);
    let mut r = l.clone();
    // Reverse AD; the generator already shuffled Entra and EXO.
    r.ad.owner.reverse();
    r.ad.scope.reverse();
    r.ad.at.reverse();
    r.ad_anchor.id.reverse();
    r.ad_anchor.state.reverse();
    r.ad_backsync.id.reverse();
    r.ad_backsync.state.reverse();
    let before = identities(&l);
    assert!(before.len() > 50);
    assert_eq!(before, identities(&r));
}

/// The witness applies per pass: a foreign frame, a re-seal, and source or
/// coordinate revision changes all refuse before the program runs.
#[test]
fn a_batch_pass_refuses_stale_and_foreign_frames() {
    let l = faulted(130, 9);
    let f = Fold::new(&l);
    let mut frames = FrameStore::new(revisions(1, 1, 1));
    let p = lowered(confirmed_hybrid(), Agg::Rows);
    let mut scratch = Scratch::for_program(&p, f.n).unwrap();
    let mut kept = vec![0_u64; f.n.div_ceil(64)];
    let frame = frames.seal();
    assert!(select(&f, &frames, &frame, &p, &mut scratch, &mut kept).is_ok());

    let twin = FrameStore::new(revisions(1, 1, 1));
    assert_eq!(
        select(&f, &twin, &frame, &p, &mut scratch, &mut kept),
        Err(FrameRefusal::ForeignFrame)
    );
    frames.advance(revisions(1, 1, 2));
    assert_eq!(
        select(&f, &frames, &frame, &p, &mut scratch, &mut kept),
        Err(FrameRefusal::CoordinatesChanged)
    );
    let frame = frames.seal();
    frames.advance(revisions(2, 2, 2));
    assert_eq!(
        select(&f, &frames, &frame, &p, &mut scratch, &mut kept),
        Err(FrameRefusal::SourceChanged)
    );
    let frame = frames.seal();
    frames.advance(revisions(2, 2, 2));
    assert_eq!(
        select(&f, &frames, &frame, &p, &mut scratch, &mut kept),
        Err(FrameRefusal::FrameResealed)
    );
}

/// Cold and hot costs at 64K slots, and zero allocations per hot pass.
/// Reported, not gated.
#[test]
fn batch_selection_at_64k_allocates_nothing() {
    use std::time::Instant;
    let n = 1 << 16;
    let t = Instant::now();
    let l = faulted(n, 0xDEAD_BEEF);
    let build = t.elapsed();
    let t = Instant::now();
    let f = Fold::new(&l);
    let cold = t.elapsed();

    let frames = FrameStore::new(revisions(1, 1, 1));
    let frame = frames.seal();
    let keep = lowered(confirmed_hybrid(), Agg::Rows);
    let count = lowered(confirmed_hybrid(), Agg::Count);
    let mut scratch_k = Scratch::for_program(&keep, n).unwrap();
    let mut scratch_c = Scratch::for_program(&count, n).unwrap();
    let mut kept = vec![0_u64; n.div_ceil(64)];
    let masks = plane_refs(&f.planes);
    let inputs = QueryPlanes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };

    const PASSES: u32 = 200;
    let before = allocations();
    let t = Instant::now();
    let mut k = 0;
    for _ in 0..PASSES {
        k = select(&f, &frames, &frame, &keep, &mut scratch_k, &mut kept).unwrap();
    }
    let keep_time = t.elapsed();
    let t = Instant::now();
    let mut c = Value::Count(0);
    for _ in 0..PASSES {
        frames.check(&frame).unwrap();
        c = execute_into(&count, &inputs, &Foreign::NONE, &mut scratch_c, Out::None).unwrap();
    }
    let count_time = t.elapsed();
    let hot_allocations = allocations() - before;

    assert_eq!(c, Value::Count(k));
    assert_eq!(hot_allocations, 0, "the hot batch path allocated");
    eprintln!(
        "64K slots: fixture {} µs | cold index+fold+ruler+planes {} µs (reported, not a gate) | keep {} ns/pass | count {} ns/pass | kept {k} | hot allocations {hot_allocations}",
        build.as_micros(),
        cold.as_micros(),
        keep_time.as_nanos() / u128::from(PASSES),
        count_time.as_nanos() / u128::from(PASSES),
    );
}
