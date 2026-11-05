//! Read-only hybrid-coordinate lab. NO AD/Graph/Exchange I/O or production admission.
//!
//! Existing OGAR Index + fold owns the cross-source match. The lab binds
//! snapshot-local ordinals once, then reads two unrelated source-property
//! lanes through the same tiny descriptor. No joined-row/PSCustomObject store.
//!
//! Not proved here: a production DR MapTable-to-SoA layout witness, a general
//! FACET STRIDE GUID join, an admitted arbitrary cross-table Gather, or full
//! historic data replay. See the companion PR description.

#![forbid(unsafe_code)]

use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::{
    BindStore, ClassSchema, Extent, MethodBinding, MethodId, MethodRef, Refusal, SchemaRev,
    SigRev, ValueDemand,
};
use lance_graph_mask_risc::{
    execute_into, Foreign, LaneRef, Out, Planes as QueryPlanes, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Col, Filter, Mask, Query};
use ogar_dir_core::correspond::{
    fold, fold_aligned, state, AdStatus, IdColumn, Index, Lanes, Output, Planes, Rows,
    Ruler, NO_ROW,
};
use ogar_dir_core::identity::Encoding;
use ogar_dir_core::label::LabelPattern;
use ogar_dir_core::Guid128;

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

/// This revision pair is a deliberately explicit *lab* witness, NOT a new
/// production receipt or DR-owned layout identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stamp {
    source: u64,
    layout: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadError {
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BoundCoordinates {
    ad: usize,
    entra: usize,
    exo: usize,
    scope: u32,
    ad_id: Guid128,
    entra_id: Guid128,
    exo_id: Guid128,
    stamp: Stamp,
}

impl BoundCoordinates {
    /// The only hot read: check that the caller's existing borrowed lanes
    /// still match the bound identities/scope and index a demanded property.
    /// No GUID matching, string decoding or joined-row construction here.
    fn read(
        &self,
        l: &Lanes,
        current: Stamp,
        family: Family,
        values: &[i32],
    ) -> Result<i32, ReadError> {
        if current != self.stamp
            || l.ad.owner.get(self.ad) != Some(&self.ad_id)
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

    /// Resolve once at the cold boundary. OGAR already provides AD→Entra
    /// ordinals. The current Output has no public Entra→EXO row-ordinal lane,
    /// so this lab alone performs a unique EXO lookup at BIND time. It does
    /// not repeat that lookup for each property read.
    fn bind(&self, l: &Lanes, ad: usize, stamp: Stamp) -> Option<BoundCoordinates> {
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
        // COLD only: use the same scope and 128-bit external ID as OGAR.
        // This supplementary scan is an integration-seam diagnostic, not a
        // canonical side index, nor proof of a SIMD-STRIDE join.
        let mut candidates = (0..l.exo.len()).filter(|&row| {
            l.exo.scope[row] == scope
                && l.exo_external.state[row] == state::PRESENT
                && l.exo_external.id[row] == cloud_id
        });
        let exo = candidates.next()?;
        if candidates.next().is_some() {
            return None;
        }
        Some(BoundCoordinates {
            ad,
            entra,
            exo,
            scope,
            ad_id: l.ad.owner[ad],
            entra_id: cloud_id,
            exo_id: l.exo.owner[exo],
            stamp,
        })
    }
}

/// Independently specified scalar reference: no OGAR Index, Output, Ruler,
/// Planes, local-candidate index or DR position selection.
fn oracle(l: &Lanes, ad: usize) -> Option<(usize, usize)> {
    let scope = *l.ad.scope.get(ad)?;
    let anchor = *l.ad_anchor.id.get(ad)?;
    let backsync = *l.ad_backsync.id.get(ad)?;
    if l.ad_anchor.state[ad] != state::PRESENT
        || l.ad_backsync.state[ad] != state::PRESENT
    {
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
            && ((l.ad_anchor.state[i] == state::PRESENT
                && l.ad_anchor.id[i] == anchor)
                || (l.ad_backsync.state[i] == state::PRESENT
                    && l.ad_backsync.id[i] == backsync))
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

const S0: Stamp = Stamp {
    source: 1,
    layout: 1,
};

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
    let prepared = Prepared::new(&l);
    let r = prepared.bind(&l, 0, S0).expect("confirmed AD→Entra→EXO");
    assert_eq!((r.entra, r.exo), oracle(&l, 0).unwrap());

    // Three separate source-owned property lanes. No Vec<JoinedRow>.
    let ad_department = [6_i32, 7];
    let entra_license_count = [25_i32, 40];
    let exo_archive_state = [1_i32, 0];
    assert_eq!(r.read(&l, S0, Family::Ad, &ad_department), Ok(6));
    assert_eq!(
        r.read(&l, S0, Family::Entra, &entra_license_count),
        Ok(25)
    );
    assert_eq!(
        r.read(&l, S0, Family::Exo, &exo_archive_state),
        Ok(1)
    );
    assert_eq!(
        r.read(&l, S0, Family::Exo, &[]),
        Err(ReadError::MissingProperty)
    );
}

#[test]
fn moved_ordinals_refuse_stale_read_then_rebind_to_same_identity() {
    let l = snapshot();
    let old = Prepared::new(&l).bind(&l, 0, S0).unwrap();
    let mut moved = l.clone();
    moved.entra.owner.swap(0, 1);
    moved.entra.scope.swap(0, 1);
    moved.entra.at.swap(0, 1);
    moved.entra_anchor.id.swap(0, 1);
    moved.entra_anchor.state.swap(0, 1);
    assert_eq!(
        old.read(&moved, S0, Family::Entra, &[40, 25]),
        Err(ReadError::StaleCoordinate)
    );
    let newer = Stamp { source: 1, layout: 2 };
    let rebound = Prepared::new(&moved).bind(&moved, 0, newer).unwrap();
    assert_eq!(rebound.entra, 1);
    assert_eq!((rebound.entra, rebound.exo), oracle(&moved, 0).unwrap());
    assert_eq!(
        rebound.read(&moved, newer, Family::Entra, &[40, 25]),
        Ok(25)
    );
    assert_eq!(
        rebound.read(&moved, S0, Family::Entra, &[40, 25]),
        Err(ReadError::StaleCoordinate)
    );
}

#[test]
fn duplicate_cloud_mailbox_and_foreign_scope_never_choose_first() {
    let mut l = snapshot();
    l.exo.owner.push(id(42));
    l.exo.scope.push(0);
    l.exo.at.push(1_100);
    l.exo_external.id.push(id(30));
    l.exo_external.state.push(state::PRESENT);
    assert!(Prepared::new(&l).bind(&l, 0, S0).is_none());

    // Same raw Entra ID in another tenant is NOT a match.
    let mut l = snapshot();
    l.exo.scope[0] = 1;
    assert!(Prepared::new(&l).bind(&l, 0, S0).is_none());

    let mut l = snapshot();
    l.entra_anchor.id[0] = id(999); // contradicts AD source anchor
    assert!(Prepared::new(&l).bind(&l, 0, S0).is_none());
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
    dt.register_derived(class, derived, schema, vec![changed]).unwrap();
    let after = dt.route(0, derived).unwrap();
    assert_eq!(dt.routed_at(before, 0).unwrap(), Some(class));
    assert_eq!(dt.routed_at(after, 0).unwrap(), Some(derived));
}

#[test]
fn existing_quack_mask_risc_fold_uses_borrowed_property_lane() {
    let l = snapshot();
    assert_eq!(Prepared::new(&l).bind(&l, 0, S0).map(|r| r.entra), Some(0));
    assert_eq!(Prepared::new(&l).bind(&l, 1, S0).map(|r| r.entra), Some(1));

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
