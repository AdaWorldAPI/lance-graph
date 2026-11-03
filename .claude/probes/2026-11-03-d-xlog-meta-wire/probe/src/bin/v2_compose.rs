//! D-XLOG-META-WIRE P0b: can two registered MethodRefs be resolved at cold
//! preparation into ONE executable program, with identity, revision,
//! permission, guard semantics and Frame/Focus/Scope still checkable?
//!
//! LAB ONLY. Production `GuardNotExecutable` is untouched: the guarded M2 is
//! still refused by `prepare` (F0). A lab composer emulates the PROPOSED
//! D-DT64-6 ruling with existing contracts only:
//!
//! - the IR is DT64's own `MethodBinding` / `Effect` (no new IR);
//! - M2's WHEN is `SetExpr::Sym(k)`. The contract leaves a `Sym`'s meaning to
//!   the consumer ("What carries it is the consumer's business"), so the lab
//!   binds `k` to (M1's `MethodRef`, the predicate field M1 writes) in a lab
//!   table. That table is exactly the semantic edge the contract lacks;
//! - the composite is registered with `register_derived` and the position
//!   re-routed to it (D5), then prepared and run by r2il unchanged.
//!
//! No mask crosses a program boundary; nothing is written to the WORLD.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, FrameRevisions, FrameStore, GuardRefusal, Guards, Scope};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, Guard, MethodBinding, MethodId,
    MethodRef, Refusal, RouteVersion, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{Foreign, LaneRef, Planes};
use lance_graph_r2il::binding::{
    prepare, run, ExecPermit, ExecRefusal, FieldLanes, Prepared, R2ilLens, Receipt, Resolution,
};
use lance_graph_r2il::FoldDialect;
use ogar_loco::{FnIndex, Vocabulary};
use ogar_r2il::{R2ILVocabulary, CONCEPT_R2IL_FOLD};
use std::collections::BTreeSet;

const CLASS: u32 = 0x7F0B_0001;
const DERIVED: u32 = 0x7F0B_0002; // same concept, another app half: same ClassView
const OTHER: u32 = 0x7F0C_0001;
const CLASS_VIEW: ClassId = 0x7F0B;
const OTHER_VIEW: ClassId = 0x7F0C;
const LENS: u32 = (CONCEPT_R2IL_FOLD as u32) << 16;
const REV: SchemaRev = SchemaRev(1);
const R2IL: R2ilLens = R2ilLens { lens: LENS, schema_rev: REV };

const FIELDS: [&str; 8] = ["a", "b", "k", "m1", "s", "t", "u", "x"];
const LANES: [Option<u16>; 8] = [Some(0), Some(1), Some(2), None, None, None, None, None];
const F_A: u8 = 0;
const F_B: u8 = 1;
const F_K: u8 = 2;
const F_M1: u8 = 3;
const F_S: u8 = 4;
const F_T: u8 = 5;
const F_U: u8 = 6;
const F_X: u8 = 7;

const WORDS: [&str; 3] = ["IntSLess", "IntAnd", "PopCount"];
const W_LT: u8 = 0;
const W_AND: u8 = 1;
const W_POP: u8 = 2;
/// Words that end a program (fold opcodes) in this lens.
const FOLD_WORDS: [u8; 1] = [W_POP];

const NOW: (Focus, Scope) = (Focus { revision: 1 }, Scope { id: 1 });
const POSITION: u16 = 3;
const REVS: FrameRevisions = FrameRevisions { source: 1, relation_map: 1, coordinates: 1 };

const M1: u32 = 1; //  m1 = a < 5                       (the semantic predicate)
const M1B: u32 = 2; // m1 = a < 7                       (a second predicate on the same field)
const M2: u32 = 3; //  WHEN Sym(1): t = b < 10; s = POP_COUNT(t)
const M3: u32 = 4; //  WHEN Sym(3): x = k < 2; s = POP_COUNT(x)   (cycle with M4)
const M4: u32 = 5; //  WHEN Sym(4): x = k < 3; s = POP_COUNT(x)
const M_FOLDPRED: u32 = 6; // m1 = POP_COUNT(...) — a fold result as "predicate"
const M2_K3: u32 = 7; //   M2 under Kleene logic
const COMPOSITE: u32 = 100;

/// The lab guard-reference table: Sym k → (referenced method, the predicate
/// field it writes). The contract has no slot for this; see the module doc.
struct SymTable(Vec<(u32, MethodRef, u8)>);
impl SymTable {
    fn get(&self, k: u32) -> Option<(&MethodRef, u8)> {
        self.0.iter().find(|(s, _, _)| *s == k).map(|(_, r, f)| (r, *f))
    }
}

struct Views(Vec<FieldRef>, Vec<FieldRef>);
fn refs(labels: &[&str], ns: &str) -> Vec<FieldRef> {
    labels.iter().map(|l| FieldRef::new(format!("lab:{ns}/{l}"), (*l).to_string())).collect()
}
impl ClassView for Views {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        match class {
            CLASS_VIEW | OTHER_VIEW => &self.0,
            c if c == CONCEPT_R2IL_FOLD => &self.1,
            _ => &[],
        }
    }
    fn template(&self, _c: ClassId) -> DisplayTemplate { DisplayTemplate::Card }
    fn dolce_category_id(&self, _c: ClassId) -> u8 { 0 }
}
fn arities() -> Vec<u8> {
    WORDS.iter().map(|w| {
        (0..=u8::MAX).map(FnIndex).find(|&f| R2ILVocabulary.name(f) == Some(*w))
            .and_then(|f| R2ILVocabulary.stack_arity(f)).unwrap_or(1)
    }).collect()
}
fn step(target: u8, word: u8, args: Vec<Arg>) -> Effect {
    Effect { target, algebra: AlgebraRef { lens: LENS, word }, args }
}
fn method(id: u32, on: &[u8], reads: &[u8], when: Option<Guard>, effects: Vec<Effect>) -> MethodBinding {
    MethodBinding {
        id: MethodId(id), sig_rev: SigRev(1), name: format!("m{id}"),
        on: WideFieldMask::from_positions(on), when,
        reads: WideFieldMask::from_positions(reads), effects,
        population: Population::Distinct, truth: Truth::TwoValued, demand: ValueDemand::Values,
    }
}
fn sym(k: u32) -> Option<Guard> { Some(Guard::new(SetExpr::Sym(k), Logic::TwoValued)) }
fn schema() -> ClassSchema { ClassSchema { schema_rev: REV, field_count: FIELDS.len(), extent: Extent::DR64_CUBE } }

fn base_methods() -> Vec<MethodBinding> {
    use Arg::{Const, Field};
    let mut k3 = method(M2_K3, &[F_B], &[F_B], None, vec![
        step(F_T, W_LT, vec![Field(F_B), Const(10)]),
        step(F_S, W_POP, vec![Field(F_T)]),
    ]);
    k3.when = Some(Guard::new(SetExpr::Sym(1), Logic::Kleene));
    vec![
        method(M1, &[F_A], &[F_A], None, vec![step(F_M1, W_LT, vec![Field(F_A), Const(5)])]),
        method(M1B, &[F_A], &[F_A], None, vec![step(F_M1, W_LT, vec![Field(F_A), Const(7)])]),
        method(M2, &[F_B], &[F_B], sym(1), vec![
            step(F_T, W_LT, vec![Field(F_B), Const(10)]),
            step(F_S, W_POP, vec![Field(F_T)]),
        ]),
        method(M3, &[F_K], &[F_K], sym(3), vec![
            step(F_X, W_LT, vec![Field(F_K), Const(2)]),
            step(F_S, W_POP, vec![Field(F_X)]),
        ]),
        method(M4, &[F_K], &[F_K], sym(4), vec![
            step(F_X, W_LT, vec![Field(F_K), Const(3)]),
            step(F_S, W_POP, vec![Field(F_X)]),
        ]),
        method(M_FOLDPRED, &[F_A], &[F_A], None, vec![
            step(F_T, W_LT, vec![Field(F_A), Const(5)]),
            step(F_M1, W_POP, vec![Field(F_T)]),
        ]),
        k3,
    ]
}
fn mref(class: u32, b: &MethodBinding) -> MethodRef { MethodRef::of(class, b) }
fn find(id: u32) -> MethodBinding { base_methods().into_iter().find(|m| m.id == MethodId(id)).expect("known") }

#[derive(Debug, PartialEq)]
enum ComposeRefusal {
    Store(Refusal),
    NotPermitted(MethodRef),
    UnboundSym(u32),
    OtherClass { referenced: u32, class: u32 },
    Cycle(Vec<u32>),
    GuardShape,
    ThreeValued,
    PredicateIsFoldResult { method: u32 },
    PredicateNotWritten { method: u32, field: u8 },
    FieldClash(u8),
}

/// What the composite was resolved from: every MethodRef (with its signature
/// revision), in dependency order. The composite's identity rides on these.
#[derive(Debug, Clone, PartialEq)]
struct Composed { binding: MethodBinding, deps: Vec<MethodRef> }

/// LAB composer (the proposed ruling, emulated). Resolves `m2`'s WHEN leaves
/// through `syms` into the referenced predicate's effects, prepends them,
/// and gates each of `m2`'s folds by the guard register. Refuses by name.
fn compose(store: &BindStore, m2: &MethodRef, syms: &SymTable, permit: &ExecPermit, id: u32) -> Result<Composed, ComposeRefusal> {
    let mut deps = vec![];
    let mut stack = vec![];
    let prefix = resolve(store, m2, syms, permit, &mut stack, &mut deps)?;
    let b2 = store.admit(m2).map_err(ComposeRefusal::Store)?;
    let (pre_effects, guard_field, pre_reads) = prefix.expect("m2 is guarded");
    let written: BTreeSet<u8> = pre_effects.iter().map(|e| e.target).collect();
    let mut effects = pre_effects.clone();
    for e in &b2.effects {
        if written.contains(&e.target) || e.target == F_U { return Err(ComposeRefusal::FieldClash(e.target)); }
        if FOLD_WORDS.contains(&e.algebra.word) {
            // Gate the fold's mask operand by the guard register (in-program).
            let Some(Arg::Field(m)) = e.args.first().copied() else { return Err(ComposeRefusal::GuardShape) };
            effects.push(step(F_U, W_AND, vec![Arg::Field(guard_field), Arg::Field(m)]));
            let mut e2 = e.clone();
            e2.args[0] = Arg::Field(F_U);
            effects.push(e2);
        } else {
            effects.push(e.clone());
        }
    }
    deps.push(m2.clone());
    let binding = MethodBinding {
        id: MethodId(id), sig_rev: SigRev(1), name: format!("compose{id}"),
        on: b2.on.clone(), when: None,
        reads: b2.reads.union(&pre_reads), effects,
        population: b2.population, truth: Truth::TwoValued, demand: b2.demand,
    };
    Ok(Composed { binding, deps })
}

type Prefix = Option<(Vec<Effect>, u8, WideFieldMask)>;
fn resolve(store: &BindStore, r: &MethodRef, syms: &SymTable, permit: &ExecPermit, stack: &mut Vec<u32>, deps: &mut Vec<MethodRef>) -> Result<Prefix, ComposeRefusal> {
    if stack.contains(&r.method.0) {
        let mut c = stack.clone();
        c.push(r.method.0);
        return Err(ComposeRefusal::Cycle(c));
    }
    let b = store.admit(r).map_err(ComposeRefusal::Store)?;
    if !permit.methods.contains(r) { return Err(ComposeRefusal::NotPermitted(r.clone())); }
    let Some(g) = &b.when else { return Ok(None) };
    if g.logic != Logic::TwoValued || b.truth != Truth::TwoValued { return Err(ComposeRefusal::ThreeValued); }
    let SetExpr::Sym(k) = g.expr else { return Err(ComposeRefusal::GuardShape) };
    let (m1, field) = syms.get(k).ok_or(ComposeRefusal::UnboundSym(k))?;
    if m1.class != r.class { return Err(ComposeRefusal::OtherClass { referenced: m1.class, class: r.class }); }
    stack.push(r.method.0);
    let nested = resolve(store, m1, syms, permit, stack, deps)?;
    stack.pop();
    let b1 = store.admit(m1).map_err(ComposeRefusal::Store)?;
    let last = b1.effects.iter().rposition(|e| e.target == field)
        .ok_or(ComposeRefusal::PredicateNotWritten { method: m1.method.0, field })?;
    let pre: Vec<Effect> = b1.effects[..=last].to_vec();
    if pre.iter().any(|e| FOLD_WORDS.contains(&e.algebra.word)) {
        return Err(ComposeRefusal::PredicateIsFoldResult { method: m1.method.0 });
    }
    if nested.is_some() { return Err(ComposeRefusal::GuardShape); } // one level in this probe
    deps.push(m1.clone());
    Ok(Some((pre, field, b1.reads.clone())))
}

struct Data { a: Vec<i32>, b: Vec<i32>, k: Vec<u32> }
impl Data {
    fn new(n: usize) -> Data {
        Data {
            a: (0..n).map(|i| (i % 37) as i32 - 10).collect(),
            b: (0..n).map(|i| ((7 * i) % 23) as i32).collect(),
            k: (0..n).map(|i| (i % 5) as u32).collect(),
        }
    }
    fn both(&self, ta: i32) -> i64 { self.a.iter().zip(&self.b).filter(|(&a, &b)| a < ta && b < 10).count() as i64 }
    fn b_only(&self) -> i64 { self.b.iter().filter(|&&b| b < 10).count() as i64 }
}

struct World { views: Views, store: BindStore, frames: FrameStore, version: RouteVersion }
impl World {
    fn new() -> World {
        let mut store = BindStore::new();
        store.declare_algebra(LENS, REV, Extent::DR64_CUBE, arities()).expect("lens");
        store.register(CLASS, schema(), base_methods()).expect("register");
        store.register(OTHER, schema(), vec![method(M1, &[F_A], &[F_A], None, vec![step(F_M1, W_LT, vec![Arg::Field(F_A), Arg::Const(5)])])]).expect("other");
        let version = store.route(POSITION, CLASS).expect("route");
        World { views: Views(refs(&FIELDS, "c"), refs(&WORDS, "r2il")), store, frames: FrameStore::new(REVS), version }
    }
    fn prepare(&self, r: &MethodRef, rev: SchemaRev) -> Result<Prepared, ExecRefusal> {
        let g = Guards { frame: self.frames.seal(), focus: NOW.0, scope: NOW.1 };
        prepare(&self.store, &self.views, R2IL,
            &Resolution { method: r, schema_rev: rev, route: (self.version, POSITION), guards: &g },
            &self.frames, NOW)
    }
}
fn exec(w: &World, plan: &Prepared, d: &Data, permit: &ExecPermit, now: (Focus, Scope)) -> Result<Receipt, ExecRefusal> {
    let lanes = [LaneRef::I32(&d.a), LaneRef::I32(&d.b), LaneRef::U32(&d.k)];
    let planes = Planes { n_rows: d.a.len(), masks: &[], lanes: &lanes };
    let foreign = Foreign::NONE;
    let mut dl = FoldDialect::new(&planes, &foreign);
    run(plan, &w.store, &w.frames, now, permit, FieldLanes { lanes: &LANES }, &mut dl)
}

struct Tally(u32, u32);
impl Tally {
    fn check(&mut self, id: &str, ok: bool, detail: String) {
        if ok { self.0 += 1 } else { self.1 += 1 }
        println!("{} {:<5} {}", if ok { "PASS" } else { "FAIL" }, id, detail);
    }
}

fn main() {
    let mut t = Tally(0, 0);
    let mut w = World::new();
    let d = Data::new(1_000);
    let (r1, r1b, r2) = (mref(CLASS, &find(M1)), mref(CLASS, &find(M1B)), mref(CLASS, &find(M2)));
    let syms = SymTable(vec![(1, r1.clone(), F_M1), (3, mref(CLASS, &find(M4)), F_X), (4, mref(CLASS, &find(M3)), F_X)]);
    let permit_both = ExecPermit { methods: vec![r1.clone(), r2.clone()] };

    // F0 — production unchanged: the guarded M2 is still refused by prepare.
    t.check("F0", matches!(w.prepare(&r2, REV), Err(ExecRefusal::GuardNotExecutable)), "prepare(M2 with WHEN) = GuardNotExecutable (unchanged)".into());

    // F1 — M1 and M2 are independently registered and admitted.
    t.check("F1", w.store.admit(&r1).is_ok() && w.store.admit(&r2).is_ok() && r1.class == r2.class && r1.method != r2.method,
        format!("M1 {:?}/{:?}, M2 {:?}/{:?} (guard witness carried: {})", r1.method, r1.sig_rev, r2.method, r2.sig_rev, r2.guard.is_some()));

    // F2 — M2's WHEN names a symbol bound to M1's semantic predicate, not a slot.
    t.check("F2", matches!(find(M2).when, Some(Guard { expr: SetExpr::Sym(1), .. })) && syms.get(1) == Some((&r1, F_M1)),
        "WHEN = Sym(1); lab table: 1 → (M1, field m1)".into());

    // F3/F4 — cold preparation resolves both into one binding → one program.
    let c = compose(&w.store, &r2, &syms, &permit_both, COMPOSITE).expect("composes");
    t.check("F3a", c.deps == vec![r1.clone(), r2.clone()], format!("composite deps (identity + sig rev): {:?}", c.deps.iter().map(|r| (r.method.0, r.sig_rev.0)).collect::<Vec<_>>()));
    // F8 — register_derived + re-route succeeds; the base stays readable.
    let mut derived_methods = base_methods();
    derived_methods.push(c.binding.clone());
    let reg = w.store.register_derived(CLASS, DERIVED, schema(), derived_methods);
    let v2 = w.store.route(POSITION, DERIVED);
    t.check("F8a", reg.is_ok() && v2.is_ok(), format!("register_derived {reg:?}, route {v2:?}"));
    t.check("F8b", w.store.resolve(&w.views, CLASS, REV).is_ok()
        && w.store.register(CLASS, schema(), { let mut m = base_methods(); m.push(c.binding.clone()); m }) == Err(Refusal::BehaviourImmutable(CLASS)),
        "base class still resolves; adding the composite to the base is BehaviourImmutable".into());
    let old_version = w.version;
    w.version = v2.expect("routed");
    let rc = mref(DERIVED, &c.binding);
    let permit_c = ExecPermit { methods: vec![rc.clone()] };
    let plan = w.prepare(&rc, REV).expect("composite prepares");
    let rec = exec(&w, &plan, &d, &permit_c, NOW).expect("composite runs");
    t.check("F3b", rec.programs_run == 1, format!("one native fold program: programs_run = {}", rec.programs_run));
    t.check("F4", rec.output.values == vec![(F_S, d.both(5))], format!("{:?} = oracle |a<5 ∧ b<10| = {}", rec.output.values, d.both(5)));

    // F5 — disable M1's predicate: compose with the guard ignored → diverges.
    let b2 = find(M2);
    let unguarded = MethodBinding { id: MethodId(101), when: None, ..b2 };
    let mut dm = base_methods();
    dm.push(unguarded.clone());
    let mut w5 = World::new(); // its own world: re-routing here must not touch `w`
    w5.store.register_derived(CLASS, 0x7F0B_0003, schema(), dm).expect("derived 3");
    w5.version = w5.store.route(POSITION, 0x7F0B_0003).expect("route 3");
    let ru = mref(0x7F0B_0003, &unguarded);
    let pu = w5.prepare(&ru, REV).expect("prepares");
    let reu = exec(&w5, &pu, &d, &ExecPermit { methods: vec![ru] }, NOW).expect("runs");
    t.check("F5", reu.output.values == vec![(F_S, d.b_only())] && d.b_only() != d.both(5),
        format!("M1's predicate removed: {:?} = |b<10| {} ≠ {}", reu.output.values, d.b_only(), d.both(5)));
    // The referenced predicate is load-bearing: re-binding Sym(1) to M1B changes the answer.
    let syms_b = SymTable(vec![(1, r1b.clone(), F_M1)]);
    let cb = compose(&w.store, &r2, &syms_b, &ExecPermit { methods: vec![r1b.clone(), r2.clone()] }, 102).expect("composes");
    t.check("F5b", cb.binding.effects != c.binding.effects && cb.deps[0] == r1b, "Sym(1) bound to M1B composes a different program".into());

    // F6 — refusals.
    t.check("F6a", compose(&w.store, &r2, &syms, &ExecPermit { methods: vec![r2.clone()] }, 103) == Err(ComposeRefusal::NotPermitted(r1.clone())),
        "permit lacks M1 → NotPermitted(M1)".into());
    let other_m1 = MethodRef { class: OTHER, ..r1.clone() };
    let syms_o = SymTable(vec![(1, other_m1.clone(), F_M1)]);
    t.check("F6b", compose(&w.store, &r2, &syms_o, &ExecPermit { methods: vec![other_m1, r2.clone()] }, 104) == Err(ComposeRefusal::OtherClass { referenced: OTHER, class: CLASS }),
        "Sym bound to a method of another class → OtherClass".into());
    let stale_m1 = MethodRef { sig_rev: SigRev(2), ..r1.clone() };
    let syms_s = SymTable(vec![(1, stale_m1.clone(), F_M1)]);
    t.check("F6c", matches!(compose(&w.store, &r2, &syms_s, &ExecPermit { methods: vec![stale_m1, r2.clone()] }, 105), Err(ComposeRefusal::Store(Refusal::SignatureRevisionMismatch { .. }))),
        "referenced M1 at another signature revision → SignatureRevisionMismatch".into());
    let f6d = w.prepare(&rc, SchemaRev(2));
    t.check("F6d", matches!(f6d, Err(ExecRefusal::Binding(_))), format!("composite at another schema revision → {:?}", f6d.map(|_| ())));
    let rs = exec(&w, &plan, &d, &permit_c, (NOW.0, Scope { id: 2 }));
    t.check("F6e", rs == Err(ExecRefusal::Guards(GuardRefusal::ScopeChanged)), format!("another scope: {rs:?}"));
    let f6f = exec(&w, &plan, &d, &ExecPermit { methods: vec![r1.clone(), r2.clone()] }, NOW);
    t.check("F6f", f6f == Err(ExecRefusal::NotPermitted), format!("permits for M1 and M2 do not run the composite: {f6f:?}"));
    // A plan prepared before the re-route is refused after it (route moved).
    let g = Guards { frame: w.frames.seal(), focus: NOW.0, scope: NOW.1 };
    let old = prepare(&w.store, &w.views, R2IL, &Resolution { method: &mref(CLASS, &find(M1)), schema_rev: REV, route: (old_version, POSITION), guards: &g }, &w.frames, NOW);
    t.check("F6g", matches!(old, Err(ExecRefusal::RouteChanged { .. })), "a base-class plan after the re-route → RouteChanged".into());
    w.frames.advance(FrameRevisions { source: 2, ..REVS });
    t.check("F6h", matches!(exec(&w, &plan, &d, &permit_c, NOW), Err(ExecRefusal::Guards(GuardRefusal::Frame(_)))), "after the source moved → stale frame refused".into());

    // F7 — cyclic method dependency.
    let r3 = mref(CLASS, &find(M3));
    let r4 = mref(CLASS, &find(M4));
    let cy = compose(&w.store, &r3, &syms, &ExecPermit { methods: vec![r3.clone(), r4] }, 106);
    t.check("F7", matches!(cy, Err(ComposeRefusal::Cycle(ref p)) if p == &vec![M3, M4, M3]), format!("{cy:?}"));
    // Neighbours of F7: a fold result as "predicate", and a three-valued guard.
    let syms_f = SymTable(vec![(1, mref(CLASS, &find(M_FOLDPRED)), F_M1)]);
    t.check("F7b", compose(&w.store, &r2, &syms_f, &ExecPermit { methods: vec![mref(CLASS, &find(M_FOLDPRED)), r2.clone()] }, 107) == Err(ComposeRefusal::PredicateIsFoldResult { method: M_FOLDPRED }),
        "a predicate computed from a fold result → refused (FoldResultFeedsFold, kept)".into());
    let rk = mref(CLASS, &find(M2_K3));
    t.check("F7c", compose(&w.store, &rk, &syms, &ExecPermit { methods: vec![rk.clone(), r1.clone()] }, 108) == Err(ComposeRefusal::ThreeValued), "Kleene WHEN → refused".into());

    // F9 — in-program narrowing is not a DT64 population Scope.
    let s1 = exec(&w, &plan, &d, &permit_c, NOW).map(|_| ());
    t.check("F9a", s1.is_err(), "(frame already advanced; F9 restated below on a fresh world)".into());
    let mut w2 = World::new();
    w2.store.register_derived(CLASS, DERIVED, schema(), { let mut m = base_methods(); m.push(c.binding.clone()); m }).expect("derived");
    w2.version = w2.store.route(POSITION, DERIVED).expect("route");
    let p2 = w2.prepare(&rc, REV).expect("prepares");
    let a = exec(&w2, &p2, &d, &permit_c, NOW).expect("runs");
    let receipt_has_no_population = a.output.intermediates.iter().all(|f| [F_M1, F_T, F_U].contains(f));
    t.check("F9b", receipt_has_no_population && a.output.values == vec![(F_S, d.both(5))],
        format!("narrowing lives in the program (intermediates {:?}); the scope stays id {}", a.output.intermediates, NOW.1.id));
    t.check("F9c", exec(&w2, &p2, &d, &permit_c, (NOW.0, Scope { id: 7 })).is_err(), "the same narrowing under another Scope id is still refused: Scope is policy, not population".into());

    // The witness gap (the missing semantic edge, measured).
    let wit_m2 = find(M2).when.unwrap().witness();
    let wit_same = Guard::new(SetExpr::Sym(1), Logic::TwoValued).witness();
    t.check("GAP", wit_m2 == wit_same && c.deps[0] != cb.deps[0],
        "M2's GuardWitness is identical whether Sym(1) means M1 (a<5) or M1B (a<7): the witness cannot name the referenced method".into());

    println!("== checks: {} passed, {} failed", t.0, t.1);
    if t.1 > 0 { std::process::exit(1); }
}
