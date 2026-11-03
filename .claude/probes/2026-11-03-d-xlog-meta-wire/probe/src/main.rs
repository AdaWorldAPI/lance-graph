//! D-XLOG-META-WIRE P0: where does M1's result stop on its way to M2?
//!
//! Lab only, own workspace. Every call is a production API, used unchanged:
//! `lance_graph_dt64::{BindStore, Resolved::methods_on, methods_touching,
//! MethodRef}`, `lance_graph_r2il::binding::{prepare, run}`, `FoldDialect`.
//! Nothing here adds a relation, a scope store, a scheduler or an evaluator;
//! the probe only shows which of the existing contracts carry M1's result and
//! where the chain stops.
//!
//! Class fields: `a`, `b`, `k` (resident lanes 0, 1, 2), then `m1`, `s`, `t`,
//! `u`, `x` (computed, no lane). Oracle: scalar loops over the lane slices.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::{Focus, FrameRevisions, FrameStore, GuardRefusal, Guards, Scope};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, ClassSchema, Effect, Extent, Guard, MethodBinding, MethodId,
    MethodRef, RouteVersion, SchemaRev, SigRev, ValueDemand,
};
use lance_graph_mask_risc::{Foreign, LaneRef, Planes};
use lance_graph_r2il::binding::{
    prepare, run, ExecPermit, ExecRefusal, FieldLanes, Prepared, R2ilLens, Receipt, Resolution,
};
use lance_graph_r2il::{FoldDialect, FoldError, Val, INT_S_LESS, LOAD, POP_COUNT};
use ogar_loco::{FnIndex, Vocabulary};
use ogar_r2il::{R2ILVocabulary, CONCEPT_R2IL_FOLD, VIA};

const CLASS: u32 = 0x7F0B_0001;
const CLASS_VIEW: ClassId = 0x7F0B;
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

const WORDS: [&str; 5] = ["IntSLess", "IntAnd", "PopCount", "VIA", "IntEqual"];
const W_LT: u8 = 0;
const W_AND: u8 = 1;
const W_POP: u8 = 2;
const W_VIA: u8 = 3;
const W_EQ: u8 = 4;

const NOW: (Focus, Scope) = (Focus { revision: 1 }, Scope { id: 1 });
const POSITION: u16 = 3;
const REVS: FrameRevisions = FrameRevisions { source: 1, relation_map: 1, coordinates: 1 };

// Methods.
const M1: u32 = 1; //        ON a: m1 = a < 5                      (a masked result, no fold)
const M1_COUNT: u32 = 2; //  ON a: m1 = a < 5; s = POP_COUNT(m1)   (M1 with a scalar answer)
const M2: u32 = 3; //        ON m1: t = b < 10; u = m1 AND t; s = POP_COUNT(u)
const M2_WHEN: u32 = 4; //   ON m1, WHEN over m1: t = b < 10; s = POP_COUNT(t)
const FUSED: u32 = 5; //     M1's and M2's effects in ONE binding
const FUSED_CUT: u32 = 6; // FUSED with the M1 → M2 operand removed (u = t)
const XY_VIA: u32 = 7; //    a cross-row hop as a method effect (VIA)
const UNRELATED: u32 = 8; // ON k: x = k == 1; s = POP_COUNT(x)

struct Views { class: Vec<FieldRef>, lens: Vec<FieldRef> }
fn refs(labels: &[&str], ns: &str) -> Vec<FieldRef> {
    labels.iter().map(|l| FieldRef::new(format!("lab:{ns}/{l}"), (*l).to_string())).collect()
}
impl ClassView for Views {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        match class {
            CLASS_VIEW => &self.class,
            c if c == CONCEPT_R2IL_FOLD => &self.lens,
            _ => &[],
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate { DisplayTemplate::Card }
    fn dolce_category_id(&self, _class: ClassId) -> u8 { 0 }
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
fn method(id: u32, on: &[u8], reads: &[u8], effects: Vec<Effect>) -> MethodBinding {
    MethodBinding {
        id: MethodId(id), sig_rev: SigRev(1), name: format!("m{id}"),
        on: WideFieldMask::from_positions(on), when: None,
        reads: WideFieldMask::from_positions(reads), effects,
        population: Population::Distinct, truth: Truth::TwoValued, demand: ValueDemand::Values,
    }
}
fn methods() -> Vec<MethodBinding> {
    use Arg::{Const, Field};
    let m1 = || step(F_M1, W_LT, vec![Field(F_A), Const(5)]);
    let mut m2_when = method(M2_WHEN, &[F_M1], &[F_M1, F_B], vec![
        step(F_T, W_LT, vec![Field(F_B), Const(10)]),
        step(F_S, W_POP, vec![Field(F_T)]),
    ]);
    m2_when.when = Some(Guard::new(SetExpr::Sym(u32::from(F_M1)), Logic::TwoValued));
    vec![
        method(M1, &[F_A], &[F_A], vec![m1()]),
        method(M1_COUNT, &[F_A], &[F_A], vec![m1(), step(F_S, W_POP, vec![Field(F_M1)])]),
        method(M2, &[F_M1], &[F_M1, F_B], vec![
            step(F_T, W_LT, vec![Field(F_B), Const(10)]),
            step(F_U, W_AND, vec![Field(F_M1), Field(F_T)]),
            step(F_S, W_POP, vec![Field(F_U)]),
        ]),
        m2_when,
        method(FUSED, &[F_A], &[F_A, F_B], vec![
            m1(),
            step(F_T, W_LT, vec![Field(F_B), Const(10)]),
            step(F_U, W_AND, vec![Field(F_M1), Field(F_T)]),
            step(F_S, W_POP, vec![Field(F_U)]),
        ]),
        method(FUSED_CUT, &[F_A], &[F_A, F_B], vec![
            m1(),
            step(F_T, W_LT, vec![Field(F_B), Const(10)]),
            step(F_S, W_POP, vec![Field(F_T)]),
        ]),
        method(XY_VIA, &[F_A], &[F_A], vec![step(F_X, W_VIA, vec![Const(1), Const(0)])]),
        method(UNRELATED, &[F_K], &[F_K], vec![
            step(F_X, W_EQ, vec![Field(F_K), Const(1)]),
            step(F_S, W_POP, vec![Field(F_X)]),
        ]),
    ]
}

struct Fx { views: Views, store: BindStore, version: RouteVersion, frames: FrameStore }
impl Fx {
    fn new() -> Fx {
        let mut store = BindStore::new();
        store.declare_algebra(LENS, REV, Extent::DR64_CUBE, arities()).expect("lens");
        let schema = ClassSchema { schema_rev: REV, field_count: FIELDS.len(), extent: Extent::DR64_CUBE };
        store.register(CLASS, schema, methods()).expect("register");
        let version = store.route(POSITION, CLASS).expect("route");
        Fx {
            views: Views { class: refs(&FIELDS, "class"), lens: refs(&WORDS, "r2il") },
            store, version, frames: FrameStore::new(REVS),
        }
    }
    /// The faithful reference (carries the guard witness when there is one).
    fn mref(&self, id: u32) -> MethodRef {
        let all = methods();
        MethodRef::of(CLASS, all.iter().find(|m| m.id == MethodId(id)).expect("known"))
    }
    fn prepare(&self, id: u32) -> Result<Prepared, ExecRefusal> {
        let r = self.mref(id);
        let g = Guards { frame: self.frames.seal(), focus: NOW.0, scope: NOW.1 };
        prepare(&self.store, &self.views, R2IL,
            &Resolution { method: &r, schema_rev: REV, route: (self.version, POSITION), guards: &g },
            &self.frames, NOW)
    }
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
    fn oracle_m1(&self) -> i64 { self.a.iter().filter(|&&a| a < 5).count() as i64 }
    fn oracle_both(&self) -> i64 {
        self.a.iter().zip(&self.b).filter(|(&a, &b)| a < 5 && b < 10).count() as i64
    }
    fn oracle_b(&self) -> i64 { self.b.iter().filter(|&&b| b < 10).count() as i64 }
}

fn exec(fx: &Fx, plan: &Prepared, data: &Data, ids: &[u32], now: (Focus, Scope), lanes_map: &[Option<u16>]) -> Result<Receipt, ExecRefusal> {
    let lanes = [LaneRef::I32(&data.a), LaneRef::I32(&data.b), LaneRef::U32(&data.k)];
    let planes = Planes { n_rows: data.a.len(), masks: &[], lanes: &lanes };
    let foreign = Foreign::NONE;
    let mut d = FoldDialect::new(&planes, &foreign);
    let permit = ExecPermit { methods: ids.iter().map(|&i| fx.mref(i)).collect() };
    run(plan, &fx.store, &fx.frames, now, &permit, FieldLanes { lanes: lanes_map }, &mut d)
}

struct Tally(u32, u32);
impl Tally {
    fn check(&mut self, id: &str, ok: bool, detail: String) {
        if ok { self.0 += 1 } else { self.1 += 1 }
        println!("{} {:<6} {}", if ok { "PASS" } else { "FAIL" }, id, detail);
    }
}

fn ids<'a>(it: impl Iterator<Item = &'a MethodBinding>) -> Vec<u32> {
    it.map(|m| m.id.0).collect()
}

fn main() {
    let mut t = Tally(0, 0);
    let fx = Fx::new();
    let data = Data::new(1_000);

    // T1: trigger (ON) vs dependency footprint, on the change of field m1.
    let res = fx.store.resolve(&fx.views, CLASS, REV).expect("resolve");
    let on = ids(res.methods_on(F_M1));
    let mask = WideFieldMask::from_positions(&[F_M1]);
    let touching = ids(res.methods_touching(&mask));
    t.check("T1a", on == vec![M2, M2_WHEN], format!("methods_on(m1) = {on:?}: only methods whose ON names m1"));
    t.check("T1b", touching.contains(&M1) && !on.contains(&M1),
        format!("methods_touching(m1) = {touching:?}: includes M1, which WRITES m1 but is not triggered by it"));
    t.check("T1c", !on.contains(&UNRELATED) && !touching.contains(&UNRELATED), "UNRELATED (ON k) in neither".into());

    // T2: the receipt boundary. M1's masked result leaves only its field position.
    let p1 = fx.prepare(M1).expect("M1 prepares");
    let r1 = exec(&fx, &p1, &data, &[M1], NOW, &LANES).expect("M1 runs");
    t.check("T2a", r1.output.values.is_empty() && r1.output.intermediates == vec![F_M1] && r1.programs_run == 0,
        format!("M1 receipt: values {:?}, intermediates {:?}, programs {}", r1.output.values, r1.output.intermediates, r1.programs_run));
    let p1c = fx.prepare(M1_COUNT).expect("M1_COUNT prepares");
    let r1c = exec(&fx, &p1c, &data, &[M1_COUNT], NOW, &LANES).expect("runs");
    t.check("T2b", r1c.output.values == vec![(F_S, data.oracle_m1())] && r1c.output.intermediates == vec![F_M1],
        format!("M1_COUNT receipt: values {:?} (oracle {}), mask still only {:?}", r1c.output.values, data.oracle_m1(), r1c.output.intermediates));

    // T3: M2 cannot be handed M1's result. `run` takes no receipt; a field
    // with no earlier value in the SAME run is read from a resident lane.
    let p2 = fx.prepare(M2).expect("M2 prepares (preparation runs nothing)");
    let r2 = exec(&fx, &p2, &data, &[M2], NOW, &LANES);
    t.check("T3a", r2 == Err(ExecRefusal::FieldUnbound { effect: 1, field: F_M1 }), format!("M2 alone: {r2:?}"));
    // Binding m1 to a lane index that names no I32/U32/U64 lane: FieldLanes
    // addresses value lanes only, never a mask.
    let mut lm = LANES;
    lm[F_M1 as usize] = Some(3);
    let r2b = exec(&fx, &p2, &data, &[M2], NOW, &lm);
    t.check("T3b", r2b == Err(ExecRefusal::LaneUnusable { effect: 1, field: F_M1, lane: 3 }), format!("M2 with m1 bound to a non-lane: {r2b:?}"));

    // T4: a mask slot does not outlive the run that minted it.
    let lanes = [LaneRef::I32(&data.a), LaneRef::I32(&data.b), LaneRef::U32(&data.k)];
    let planes = Planes { n_rows: data.a.len(), masks: &[], lanes: &lanes };
    let foreign = Foreign::NONE;
    let mut d = FoldDialect::new(&planes, &foreign);
    let mut st = vec![Val::Scalar(0)];
    d.call(LOAD, [1, 0, 0], &mut st).expect("load a");
    st.push(Val::Scalar(5));
    d.call(INT_S_LESS, [0, 0, 0], &mut st).expect("a < 5");
    let slot = st.pop().expect("slot");
    let mut live = vec![slot];
    let ok = d.call(POP_COUNT, [0, 0, 0], &mut live).map(|_| live.clone());
    t.check("T4a", ok == Ok(vec![Val::Scalar(data.oracle_m1())]), format!("same epoch: {ok:?}"));
    let mut st = vec![Val::Scalar(0)];
    d.call(LOAD, [1, 0, 0], &mut st).expect("load a");
    st.push(Val::Scalar(5));
    d.call(INT_S_LESS, [0, 0, 0], &mut st).expect("a < 5");
    let slot = st.pop().expect("slot");
    d.discard_pending(); // what `run` does after every run
    let mut stale = vec![slot];
    let e = d.call(POP_COUNT, [0, 0, 0], &mut stale);
    t.check("T4b", e == Err(FoldError::StaleSlot), format!("after discard_pending (end of a run): {e:?}"));

    // T5: WHEN cannot carry M1's result as M2's scope.
    let pw = fx.prepare(M2_WHEN);
    t.check("T5", pw.as_ref().map(|_| ()) == Err(&ExecRefusal::GuardNotExecutable), format!("M2 with WHEN over m1: {:?}", pw.map(|_| ())));

    // T6: the cross-row XY hop is not an executable method effect.
    let pv = fx.prepare(XY_VIA);
    t.check("T6", matches!(pv, Err(ExecRefusal::NotExecutable { effect: 0, opcode }) if opcode == VIA), format!("VIA as an effect: {:?}", pv.map(|_| ())));

    // T7: Scope is an id compared for equality; a different scope refuses.
    let pf = fx.prepare(FUSED).expect("FUSED prepares");
    let narrower = (NOW.0, Scope { id: 2 });
    let rs = exec(&fx, &pf, &data, &[FUSED], narrower, &LANES);
    t.check("T7", rs == Err(ExecRefusal::Guards(GuardRefusal::ScopeChanged)), format!("FUSED under another scope: {rs:?}"));

    // T8: the furthest lawful route: M1's mask feeds M2's fold INSIDE one
    // binding (a Data-origin mask into a fold; no FoldResultFeedsFold).
    let rf = exec(&fx, &pf, &data, &[FUSED], NOW, &LANES).expect("FUSED runs");
    t.check("T8a", rf.output.values == vec![(F_S, data.oracle_both())],
        format!("FUSED = {:?}, oracle |a<5 ∧ b<10| = {}", rf.output.values, data.oracle_both()));
    let pc = fx.prepare(FUSED_CUT).expect("prepares");
    let rc = exec(&fx, &pc, &data, &[FUSED_CUT], NOW, &LANES).expect("runs");
    t.check("T8b", rc.output.values == vec![(F_S, data.oracle_b())] && data.oracle_b() != data.oracle_both(),
        format!("M1 operand removed: {:?} = |b<10| {} ≠ {}", rc.output.values, data.oracle_b(), data.oracle_both()));
    let mut d2 = Data::new(1_000);
    let i = (0..d2.a.len()).find(|&i| d2.a[i] < 5 && d2.b[i] < 10).expect("a supporting row");
    d2.a[i] = 30; // delete one support of M1
    let rd = exec(&fx, &pf, &d2, &[FUSED], NOW, &LANES).expect("runs");
    t.check("T8c", rd.output.values == vec![(F_S, data.oracle_both() - 1)], format!("one M1 support removed: {:?}", rd.output.values));
    let mut d3 = Data::new(1_000);
    for k in d3.k.iter_mut() { *k = (*k + 1) % 5; }
    let ru = exec(&fx, &pf, &d3, &[FUSED], NOW, &LANES).expect("runs");
    t.check("T8d", ru.output.values == rf.output.values, "unrelated lane k changed: answer unchanged".into());
    let mut frames2 = FrameStore::new(REVS);
    let g_old = frames2.seal();
    frames2.advance(FrameRevisions { source: 2, ..REVS });
    t.check("T8e", Guards { frame: g_old, focus: NOW.0, scope: NOW.1 }.check(&frames2, NOW).is_err(),
        "a frame sealed before the source moved is refused".into());

    // T9: fusing is not M1 → M2: the class's behaviour is immutable, so the
    // fused binding cannot be added beside M1 and M2 later.
    let mut s2 = Fx::new().store;
    let schema = ClassSchema { schema_rev: REV, field_count: FIELDS.len(), extent: Extent::DR64_CUBE };
    let mut more = methods();
    more.push(method(99, &[F_A], &[F_A], vec![step(F_M1, W_LT, vec![Arg::Field(F_A), Arg::Const(6)])]));
    let reg = s2.register(CLASS, schema, more);
    t.check("T9", matches!(reg, Err(lance_graph_dt64::Refusal::BehaviourImmutable(c)) if c == CLASS),
        format!("adding a composed method to a registered class: {reg:?}"));

    println!("== checks: {} passed, {} failed", t.0, t.1);
    if t.1 > 0 { std::process::exit(1); }
}
