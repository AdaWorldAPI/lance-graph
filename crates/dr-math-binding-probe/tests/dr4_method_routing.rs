//! D-DRCA-3: the DR #4 falsifier (D-DRDT §5, lance-graph #1835).
//!
//! Two methods on one cell with the same read/write sets and a different effect
//! order must stay distinct through routing, and a dropped guard must refuse.
//!
//! DR carries method references as opaque values. Their meaning lives in the
//! DT stub below, which is also the oracle: it shows that two references DR
//! keeps apart really are different methods. The effect-order law is ndarray
//! #395's (`examples/pof_antennas/order_law.rs`), ported verbatim.

use dr_math_binding_probe::dr4::{
    Algebra, Dr4Refusal, EffectOrderRef, GuardDecl, GuardWitness, MethodKey, MethodRef, Op,
    Resolution, Value, route,
};
use dr_math_binding_probe::{
    Axis, AxisKind, BoundDr64, Focus, FrameRevisions, FrameStore, Guards, MethodSignature, Operand,
    Refusal, Scope, Truth, bind_dr64,
};
use lance_graph_contract::class_view::ClassId;
use sha2::{Digest, Sha256};

// ── #395 order law, verbatim (ndarray examples/pof_antennas/order_law.rs) ──

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Effect {
    Set(i64),
    Add(i64),
    Mul(i64),
}

impl Effect {
    fn apply(&self, cell: i64) -> Option<i64> {
        match *self {
            Effect::Set(v) => Some(v),
            Effect::Add(v) => cell.checked_add(v),
            Effect::Mul(v) => cell.checked_mul(v),
        }
    }
}

fn run_effects(init: i64, effs: &[Effect]) -> Option<i64> {
    effs.iter().try_fold(init, |cell, e| e.apply(cell))
}

fn commutes(a: &Effect, b: &Effect) -> bool {
    match (*a, *b) {
        (Effect::Add(x), Effect::Add(y)) => !((x < 0 && y > 0) || (x > 0 && y < 0)),
        (Effect::Mul(x), Effect::Mul(y)) => {
            x == y || x == 1 || y == 1 || (x.unsigned_abs() >= 2 && y.unsigned_abs() >= 2)
        }
        (Effect::Set(x), Effect::Set(y)) => x == y,
        _ => false,
    }
}

#[derive(Debug, PartialEq, Eq)]
struct OrderRefusal {
    first: usize,
    second: usize,
}

fn check_reorder(effs: &[Effect], order: &[usize]) -> Result<(), OrderRefusal> {
    let n = effs.len();
    assert_eq!(order.len(), n);
    let mut pos = vec![usize::MAX; n];
    for (k, &orig) in order.iter().enumerate() {
        assert_eq!(pos[orig], usize::MAX, "not a permutation");
        pos[orig] = k;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if !commutes(&effs[i], &effs[j]) && pos[i] > pos[j] {
                return Err(OrderRefusal {
                    first: i,
                    second: j,
                });
            }
        }
    }
    Ok(())
}

// ── DT stub: the meaning DR never sees ──

const CLASS: ClassId = 7;
const OTHER_CLASS: ClassId = 8;

/// One method as DT would hold it. DR receives only `to_ref()`.
struct DtMethod {
    key: [u8; 16],
    reads: u64,
    writes: u64,
    guard: Option<&'static str>,
    effects: Vec<Effect>,
    algebra: Algebra,
}

fn digest16(bytes: &[u8]) -> [u8; 16] {
    let h = Sha256::digest(bytes);
    let mut out = [0u8; 16];
    out.copy_from_slice(&h[..16]);
    out
}

fn effect_ref(effs: &[Effect]) -> EffectOrderRef {
    let mut enc = Vec::new();
    for e in effs {
        let (tag, v) = match *e {
            Effect::Set(v) => (0u8, v),
            Effect::Add(v) => (1u8, v),
            Effect::Mul(v) => (2u8, v),
        };
        enc.push(tag);
        enc.extend_from_slice(&v.to_le_bytes());
    }
    EffectOrderRef(digest16(&enc))
}

fn guard_witness(guard: &str) -> GuardWitness {
    GuardWitness(digest16(guard.as_bytes()))
}

impl DtMethod {
    fn to_ref(&self, class: ClassId) -> MethodRef {
        MethodRef {
            class,
            key: MethodKey(self.key),
            signature_rev: 1,
            reads: self.reads,
            writes: self.writes,
            guard: match self.guard {
                Some(g) => GuardDecl::Required(guard_witness(g)),
                None => GuardDecl::Unguarded,
            },
            effects: effect_ref(&self.effects),
            algebra: self.algebra,
        }
    }
}

fn method(key: u8, effects: Vec<Effect>, guard: Option<&'static str>) -> DtMethod {
    DtMethod {
        key: [key; 16],
        reads: 0b0110,
        writes: 0b1000,
        guard,
        effects,
        algebra: Algebra::Presence,
    }
}

// ── register fixture ──

fn revisions(n: u64) -> FrameRevisions {
    FrameRevisions {
        source: n,
        relation_map: 10,
        coordinates: 20,
    }
}

fn signature(truth: Truth) -> MethodSignature {
    let id = Axis {
        class: CLASS,
        kind: AxisKind::Identity,
    };
    let prop = Axis {
        class: CLASS,
        kind: AxisKind::Property,
    };
    let mut s = MethodSignature::presence(CLASS, 3, id, prop);
    s.truth = truth;
    s
}

fn rows_fixture() -> [u64; 64] {
    core::array::from_fn(|x| (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (1 << (x % 64)))
}

fn validity_fixture() -> [u64; 64] {
    core::array::from_fn(|x| !((x as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)) | 1)
}

const FOCUS: Focus = Focus { revision: 1 };
const SCOPE: Scope = Scope { id: 2 };

fn bind<'a>(
    store: &FrameStore,
    truth: Truth,
    rows: &'a [u64; 64],
    validity: Option<&'a [u64; 64]>,
) -> BoundDr64<'a> {
    let sig = signature(truth);
    let guards = Guards {
        frame: store.seal(),
        focus: FOCUS,
        scope: SCOPE,
    };
    bind_dr64(
        store,
        sig,
        sig,
        &guards,
        (FOCUS, SCOPE),
        Operand { rows, validity },
    )
    .unwrap()
}

fn resolution(store: &FrameStore, methods: &[&DtMethod]) -> Resolution {
    Resolution {
        class: CLASS,
        under: store.seal(),
        methods: methods.iter().map(|m| m.to_ref(CLASS)).collect(),
    }
}

/// The witnesses a caller observed, one per guarded method.
fn witnesses(methods: &[&DtMethod]) -> Vec<(MethodKey, GuardWitness)> {
    methods
        .iter()
        .filter_map(|m| m.guard.map(|g| (MethodKey(m.key), guard_witness(g))))
        .collect()
}

// ── 1. the falsifier ──

#[test]
fn same_read_write_sets_different_effect_order_stay_distinct() {
    let a = method(1, vec![Effect::Set(1), Effect::Add(1)], None);
    let b = method(2, vec![Effect::Add(1), Effect::Set(1)], None);
    // Anti-vacuity: the two methods are different by the oracle and look
    // identical by everything DR could compare except their opaque refs.
    assert_eq!((a.reads, a.writes), (b.reads, b.writes));
    assert_ne!(run_effects(0, &a.effects), run_effects(0, &b.effects));
    assert_eq!(
        check_reorder(&a.effects, &[1, 0]),
        Err(OrderRefusal {
            first: 0,
            second: 1
        })
    );

    let store = FrameStore::new(revisions(1));
    let rows = rows_fixture();
    let reg = bind(&store, Truth::TwoValued, &rows, None);
    let res = resolution(&store, &[&a, &b]);
    let routed = route(&store, &reg, &res, Op::Cell { x: 5, y: 9 }, &[]).unwrap();

    assert_eq!(routed.entries().len(), 2);
    let (ea, eb) = (&routed.entries()[0], &routed.entries()[1]);
    assert_eq!(ea.key, MethodKey(a.key));
    assert_eq!(eb.key, MethodKey(b.key));
    assert_eq!(ea.effects, effect_ref(&a.effects));
    assert_eq!(eb.effects, effect_ref(&b.effects));
    assert_ne!(ea.effects, eb.effects);
    assert_eq!((ea.reads, ea.writes), (eb.reads, eb.writes));
    // Same algebra over the same cell: same value, still two entries.
    assert_eq!(ea.value, eb.value);
    assert!(routed.frame().same_issue(&reg.frame));
}

#[test]
fn commuting_effects_are_still_not_merged() {
    let a = method(1, vec![Effect::Add(2), Effect::Add(3)], None);
    let b = method(2, vec![Effect::Add(3), Effect::Add(2)], None);
    assert_eq!(run_effects(0, &a.effects), run_effects(0, &b.effects));
    assert_eq!(check_reorder(&a.effects, &[1, 0]), Ok(()));
    // The ported law, both ways: equal-magnitude-class products commute,
    // a product across a sum does not.
    assert!(commutes(&Effect::Mul(2), &Effect::Mul(3)));
    assert!(!commutes(&Effect::Mul(2), &Effect::Add(1)));
    assert_ne!(
        run_effects(1, &[Effect::Mul(2), Effect::Add(1)]),
        run_effects(1, &[Effect::Add(1), Effect::Mul(2)])
    );

    let store = FrameStore::new(revisions(1));
    let rows = rows_fixture();
    let reg = bind(&store, Truth::TwoValued, &rows, None);
    let res = resolution(&store, &[&a, &b]);
    let routed = route(&store, &reg, &res, Op::Column { y: 3 }, &[]).unwrap();
    assert_eq!(routed.entries().len(), 2);
    assert_ne!(routed.entries()[0].effects, routed.entries()[1].effects);
}

// ── 2. guards ──

#[test]
fn a_dropped_guard_refuses() {
    let a = method(1, vec![Effect::Set(1)], Some("state == draft"));
    let store = FrameStore::new(revisions(1));
    let rows = rows_fixture();
    let reg = bind(&store, Truth::TwoValued, &rows, None);
    let res = resolution(&store, &[&a]);
    assert_eq!(
        route(&store, &reg, &res, Op::Cell { x: 0, y: 0 }, &[]).err(),
        Some(Dr4Refusal::GuardDropped)
    );
}

#[test]
fn a_different_guard_witness_refuses() {
    let a = method(1, vec![Effect::Set(1)], Some("state == draft"));
    let store = FrameStore::new(revisions(1));
    let rows = rows_fixture();
    let reg = bind(&store, Truth::TwoValued, &rows, None);
    let res = resolution(&store, &[&a]);
    let wrong = [(MethodKey(a.key), guard_witness("state == posted"))];
    assert_eq!(
        route(&store, &reg, &res, Op::Cell { x: 0, y: 0 }, &wrong).err(),
        Some(Dr4Refusal::GuardMismatch)
    );
}

#[test]
fn the_right_guard_witness_is_carried_and_unguarded_needs_none() {
    let a = method(1, vec![Effect::Set(1)], Some("state == draft"));
    let b = method(2, vec![Effect::Add(1)], None);
    let store = FrameStore::new(revisions(1));
    let rows = rows_fixture();
    let reg = bind(&store, Truth::TwoValued, &rows, None);
    let res = resolution(&store, &[&a, &b]);
    let routed = route(
        &store,
        &reg,
        &res,
        Op::Cell { x: 0, y: 0 },
        &witnesses(&[&a, &b]),
    )
    .unwrap();
    assert_eq!(
        routed.entries()[0].guard,
        Some(guard_witness("state == draft"))
    );
    assert_eq!(routed.entries()[1].guard, None);
}

// ── 3. frame, class, identity ──

#[test]
fn a_stale_register_frame_refuses() {
    let a = method(1, vec![Effect::Set(1)], None);
    let mut store = FrameStore::new(revisions(1));
    let rows = rows_fixture();
    let reg = bind(&store, Truth::TwoValued, &rows, None);
    let res = resolution(&store, &[&a]);
    store.advance(revisions(2));
    assert_eq!(
        route(&store, &reg, &res, Op::Cell { x: 0, y: 0 }, &[]).err(),
        Some(Dr4Refusal::Frame(Refusal::SourceChanged))
    );
}

#[test]
fn a_resolution_under_another_frame_refuses() {
    let a = method(1, vec![Effect::Set(1)], None);
    let mut store = FrameStore::new(revisions(1));
    let old = resolution(&store, &[&a]);
    store.advance(revisions(2));
    let rows = rows_fixture();
    let reg = bind(&store, Truth::TwoValued, &rows, None);
    assert_eq!(
        route(&store, &reg, &old, Op::Cell { x: 0, y: 0 }, &[]).err(),
        Some(Dr4Refusal::ResolvedUnderOtherFrame)
    );
    // Equal numbers in another store are another frame.
    let other = FrameStore::new(revisions(2));
    let foreign = resolution(&other, &[&a]);
    assert_eq!(
        route(&store, &reg, &foreign, Op::Cell { x: 0, y: 0 }, &[]).err(),
        Some(Dr4Refusal::ResolvedUnderOtherFrame)
    );
}

#[test]
fn a_foreign_class_refuses() {
    let a = method(1, vec![Effect::Set(1)], None);
    let store = FrameStore::new(revisions(1));
    let rows = rows_fixture();
    let reg = bind(&store, Truth::TwoValued, &rows, None);
    let mut res = resolution(&store, &[&a]);
    res.class = OTHER_CLASS;
    assert_eq!(
        route(&store, &reg, &res, Op::Cell { x: 0, y: 0 }, &[]).err(),
        Some(Dr4Refusal::ForeignClass)
    );
    let mut res = resolution(&store, &[&a]);
    res.methods[0].class = OTHER_CLASS;
    assert_eq!(
        route(&store, &reg, &res, Op::Cell { x: 0, y: 0 }, &[]).err(),
        Some(Dr4Refusal::ForeignClass)
    );
}

#[test]
fn a_duplicate_key_refuses_but_equal_sets_do_not() {
    let a = method(1, vec![Effect::Set(1)], None);
    let a2 = method(1, vec![Effect::Add(1)], None);
    let store = FrameStore::new(revisions(1));
    let rows = rows_fixture();
    let reg = bind(&store, Truth::TwoValued, &rows, None);
    let res = resolution(&store, &[&a, &a2]);
    assert_eq!(
        route(&store, &reg, &res, Op::Cell { x: 0, y: 0 }, &[]).err(),
        Some(Dr4Refusal::DuplicateMethod)
    );
}

#[test]
fn out_of_register_and_unsupported_algebra_refuse() {
    let mut a = method(1, vec![Effect::Set(1)], None);
    let store = FrameStore::new(revisions(1));
    let rows = rows_fixture();
    let reg = bind(&store, Truth::TwoValued, &rows, None);
    let res = resolution(&store, &[&a]);
    assert_eq!(
        route(&store, &reg, &res, Op::Cell { x: 64, y: 0 }, &[]).err(),
        Some(Dr4Refusal::OutOfRegister)
    );
    assert_eq!(
        route(&store, &reg, &res, Op::Column { y: 64 }, &[]).err(),
        Some(Dr4Refusal::OutOfRegister)
    );
    a.algebra = Algebra::NumericFold;
    let res = resolution(&store, &[&a]);
    assert_eq!(
        route(&store, &reg, &res, Op::Cell { x: 0, y: 0 }, &[]).err(),
        Some(Dr4Refusal::UnsupportedAlgebra)
    );
}

// ── 4. values against an independent scalar loop ──

fn bit(rows: &[u64; 64], x: usize, y: usize) -> bool {
    (rows[x] >> y) & 1 == 1
}

#[test]
fn boolean_values_match_the_scalar_oracle() {
    let a = method(1, vec![Effect::Set(1)], None);
    let store = FrameStore::new(revisions(1));
    let rows = rows_fixture();
    let reg = bind(&store, Truth::TwoValued, &rows, None);
    let res = resolution(&store, &[&a]);
    let mut seen = (false, false);
    for x in 0..64u8 {
        for y in [0u8, 1, 31, 63] {
            let v = route(&store, &reg, &res, Op::Cell { x, y }, &[]).unwrap();
            let want = bit(&rows, x as usize, y as usize);
            assert_eq!(v.entries()[0].value, Value::Bool(want), "cell {x},{y}");
            if want {
                seen.0 = true;
            } else {
                seen.1 = true;
            }
        }
        let r = route(&store, &reg, &res, Op::Row { x }, &[]).unwrap();
        assert_eq!(r.entries()[0].value, Value::Mask(rows[x as usize]));
    }
    assert!(seen.0 && seen.1, "fixture has no TRUE or no FALSE cell");
    for y in 0..64u8 {
        let want = (0..64).fold(0u64, |m, x| m | (u64::from(bit(&rows, x, y as usize)) << x));
        let c = route(&store, &reg, &res, Op::Column { y }, &[]).unwrap();
        assert_eq!(c.entries()[0].value, Value::Mask(want), "column {y}");
    }
}

#[test]
fn presence_over_three_valued_refuses_and_k3_masks_by_validity() {
    let mut a = method(1, vec![Effect::Set(1)], None);
    let store = FrameStore::new(revisions(1));
    let rows = rows_fixture();
    let validity = validity_fixture();
    let reg = bind(&store, Truth::ThreeValued, &rows, Some(&validity));
    let res = resolution(&store, &[&a]);
    assert_eq!(
        route(&store, &reg, &res, Op::Column { y: 2 }, &[]).err(),
        Some(Dr4Refusal::NotBoolean)
    );
    a.algebra = Algebra::PresenceK3;
    let res = resolution(&store, &[&a]);
    let mut unknown_true_payload = false;
    for y in 0..64u8 {
        let (mut value, mut known) = (0u64, 0u64);
        for x in 0..64 {
            let k = bit(&validity, x, y as usize);
            let v = bit(&rows, x, y as usize);
            known |= u64::from(k) << x;
            value |= u64::from(k && v) << x;
            unknown_true_payload |= !k && v;
        }
        let c = route(&store, &reg, &res, Op::Column { y }, &[]).unwrap();
        assert_eq!(
            c.entries()[0].value,
            Value::K3 { value, known },
            "column {y}"
        );
    }
    // Anti-vacuity: some UNKNOWN cell carries a 1 payload bit that must not
    // read as TRUE.
    assert!(unknown_true_payload);
    // A two-valued register under K3 is known everywhere.
    let reg2 = bind(&store, Truth::TwoValued, &rows, None);
    let c = route(&store, &reg2, &res, Op::Row { x: 4 }, &[]).unwrap();
    assert_eq!(
        c.entries()[0].value,
        Value::K3 {
            value: rows[4],
            known: u64::MAX
        }
    );
}
