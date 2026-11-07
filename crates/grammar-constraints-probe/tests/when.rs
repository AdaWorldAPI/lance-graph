//! D-GRAMMAR-WHEN-1 falsifier. Each control states its predicted outcome in
//! its doc comment, written before it was run.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use grammar_constraints_probe::fixtures::{self, Gold, A, B, C, GOLD};
use grammar_constraints_probe::when::*;
use grammar_constraints_probe::{
    has, tile, Ids, Operands, Refusal as PopRefusal, Rules, Tok, Vocab, TILE,
};
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_dt64::dr::{Frame, MapRead, MapRefusal, MapTable};
use lance_graph_dt64::{
    AlgebraRef, Arg, BindStore, Effect, Extent, Guard, MethodRef, Refusal, SchemaRev,
};

thread_local! { static BYTES: Cell<usize> = const { Cell::new(0) }; }
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = BYTES.try_with(|b| b.set(b.get() + l.size()));
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;

// --- harness ------------------------------------------------------------------

struct W {
    v: Vocab,
    table: MapTable,
    frame: Frame,
    pop: FramedPopulation,
    store: BindStore,
    lex: Lexicon,
}

fn world_with(toks: &[Tok], lex: Lexicon) -> W {
    let mut v = Vocab::new();
    let _ = Ids::bind(&mut v);
    let (table, frame) = tile(9, &fixtures::gids(toks));
    let pop = FramedPopulation::produce(toks, &mut v, &table, &frame, &lex).unwrap();
    let mut store = BindStore::new();
    let rules = [RuleDesc::recipient(), RuleDesc::experiencer()];
    register(&mut store, CLASS_VIA, Physical::Via, &rules, &v, &lex).unwrap();
    register(
        &mut store,
        CLASS_COADDR,
        Physical::CoAddressed,
        &rules,
        &v,
        &lex,
    )
    .unwrap();
    W {
        v,
        table,
        frame,
        pop,
        store,
        lex,
    }
}

fn world(toks: &[Tok]) -> W {
    world_with(toks, Lexicon::v0())
}

fn class_of(p: Physical) -> u32 {
    match p {
        Physical::Via => CLASS_VIA,
        Physical::CoAddressed => CLASS_COADDR,
    }
}

fn mref(w: &W, class: u32, desc: &RuleDesc, p: Physical) -> MethodRef {
    MethodRef::of(class, &desc.binding(p, &w.v, &w.lex))
}

fn all_plane(w: &W) -> Vec<u64> {
    let n = w.pop.base.n;
    let mut m = vec![0u64; n.div_ceil(64)];
    for i in 0..n {
        m[i / 64] |= 1 << (i % 64);
    }
    m
}

fn plane_of(w: &W, gids: &[u64]) -> Vec<u64> {
    let map = w.table.map(&w.frame, TILE).unwrap();
    let mut m = vec![0u64; w.pop.base.n.div_ceil(64)];
    for &g in gids {
        let l = usize::from(map.local(g).unwrap());
        m[l / 64] |= 1 << (l % 64);
    }
    m
}

fn gids_of(w: &W, mask: &[u64; 4]) -> Vec<u64> {
    let map = w.table.map(&w.frame, TILE).unwrap();
    let mut v: Vec<u64> = (0..w.pop.base.n as u8)
        .filter(|&l| has(mask, l))
        .map(|l| map.global(l).unwrap())
        .collect();
    v.sort_unstable();
    v
}

/// Compose and run `desc` under its own registered binding.
fn answer(w: &W, desc: &RuleDesc, p: Physical, plane: &[u64]) -> Vec<u64> {
    let r = mref(w, class_of(p), desc, p);
    let mut c = compose(&w.store, &r, desc, p, &w.v, &w.lex).unwrap();
    let m = c.run(&w.pop, &w.table, &w.frame, plane).unwrap();
    gids_of(w, &m)
}

/// Independent oracle: string-level, over the token structs, no ids, no
/// programs.
fn oracle(toks: &[Tok], lex: &Lexicon, frame: u32) -> Vec<u64> {
    let lemmas = lex.lemmas(frame);
    let mut v: Vec<u64> = toks
        .iter()
        .filter(|t| t.deprel == "obl:arg" && t.case == "Dat")
        .filter(|t| {
            toks.iter()
                .find(|h| h.gid == t.head)
                .is_some_and(|h| lemmas.contains(&h.lemma))
        })
        .map(|t| t.gid)
        .collect();
    v.sort_unstable();
    v
}

const BOTH: [Physical; 2] = [Physical::Via, Physical::CoAddressed];

// --- P0 -------------------------------------------------------------------------

#[test]
fn p0_both_forms_match_oracle_and_gold() {
    let corpus = fixtures::corpus();
    let w = world(&corpus);
    let all = all_plane(&w);
    for p in BOTH {
        let rec = answer(&w, &RuleDesc::recipient(), p, &all);
        let exp = answer(&w, &RuleDesc::experiencer(), p, &all);
        assert_eq!(rec, oracle(&corpus, &w.lex, FRAME_GIVE), "{p:?}");
        assert_eq!(exp, oracle(&corpus, &w.lex, FRAME_EXPERIENCE), "{p:?}");
        assert_eq!(rec, vec![105, 405], "{p:?}");
        assert_eq!(exp, vec![302], "{p:?}");
    }
    // gold agreement (independent annotation)
    for (gid, g) in GOLD {
        let rec = answer(&w, &RuleDesc::recipient(), Physical::CoAddressed, &all);
        let exp = answer(&w, &RuleDesc::experiencer(), Physical::CoAddressed, &all);
        assert_eq!(rec.contains(&gid), g == Gold::Recipient, "{gid}");
        assert_eq!(exp.contains(&gid), g == Gold::Experiencer, "{gid}");
    }
}

#[test]
fn guards_keep_distinct_witnesses_in_both_forms() {
    let w = world(&fixtures::corpus());
    for p in BOTH {
        let r = RuleDesc::recipient().guard(p, &w.v, &w.lex).witness();
        let e = RuleDesc::experiencer().guard(p, &w.v, &w.lex).witness();
        assert_ne!(r, e, "{p:?}");
    }
    // ...but without the frame part, they coincide (the #1947 finding).
    let strip = |mut d: RuleDesc| {
        d.frame = None;
        d.guard(Physical::CoAddressed, &w.v, &w.lex).witness()
    };
    assert_eq!(strip(RuleDesc::recipient()), strip(RuleDesc::experiencer()));
}

#[test]
fn unknown_stays_unknown_and_ambiguity_is_kept() {
    // helfen (E): no frame entry → no candidate in either method, and the
    // #1947 abstention rule still reports it UNKNOWN.
    let corpus = fixtures::corpus();
    let w = world(&corpus);
    let all = all_plane(&w);
    for p in BOTH {
        assert!(!answer(&w, &RuleDesc::recipient(), p, &all).contains(&505));
        assert!(!answer(&w, &RuleDesc::experiencer(), p, &all).contains(&505));
    }
    assert_eq!(w.pop.head_frame[usize::from(local(&w, 505))], FRAME_NONE);
    let mut v = Vocab::new();
    let ids = Ids::bind(&mut v);
    let pop =
        grammar_constraints_probe::Population::bind(&corpus, &mut v, &w.table, &w.frame).unwrap();
    let mut rules = Rules::prepare(&ids, Operands::ALL);
    let r = rules.run(&pop, &w.table, &w.frame).unwrap();
    assert!(has(&r.abstain_frame, local(&w, 505)));
    // control 9: Direction AND Purpose both kept for Schule
    assert!(has(&r.direction_or_purpose, local(&w, 208)));
}

fn local(w: &W, gid: u64) -> u8 {
    w.table.map(&w.frame, TILE).unwrap().local(gid).unwrap()
}

#[test]
fn candidates_are_read_only() {
    let w = world(&fixtures::corpus());
    for p in BOTH {
        for d in [RuleDesc::recipient(), RuleDesc::experiencer()] {
            let b = w.store.admit(&mref(&w, class_of(p), &d, p)).unwrap();
            assert!(
                b.effects.is_empty()
                    && b.writes() == lance_graph_contract::class_view::WideFieldMask::EMPTY
            );
        }
    }
    // A method with an effect is refused by the composer.
    let mut store = BindStore::new();
    store
        .declare_algebra(0x7F30_0000, SchemaRev(1), Extent::DR64_CUBE, vec![1])
        .unwrap();
    let mut b = RuleDesc::recipient().binding(Physical::CoAddressed, &w.v, &w.lex);
    b.effects = vec![Effect {
        target: 4,
        algebra: AlgebraRef {
            lens: 0x7F30_0000,
            word: 0,
        },
        args: vec![Arg::Field(1)],
    }];
    store
        .register(&LabView::new(), CLASS_COADDR, schema(), vec![b.clone()])
        .unwrap();
    let r = MethodRef::of(CLASS_COADDR, &b);
    assert_eq!(
        compose(
            &store,
            &r,
            &RuleDesc::recipient(),
            Physical::CoAddressed,
            &w.v,
            &w.lex
        )
        .err(),
        Some(ComposeRefusal::NotReadOnly)
    );
}

#[test]
fn dt64_does_not_validate_guard_lanes() {
    // FINDING: `validate` checks ON / reads / effects positions against the
    // basis, never the WHEN guard's lanes. A guard over lane 99 of a
    // five-field class registers.
    let w = world(&fixtures::corpus());
    let mut b = RuleDesc::recipient().binding(Physical::CoAddressed, &w.v, &w.lex);
    b.when = Some(Guard::new(SetExpr::eq_u32(99, 1), Logic::TwoValued));
    let mut store = BindStore::new();
    assert!(store
        .register(&LabView::new(), CLASS_COADDR, schema(), vec![b])
        .is_ok());
}

// --- negative controls ---------------------------------------------------------

/// 1. Head-frame condition removed. PREDICTED: recipient fires on 105, 302,
///    405, 505 (both rules collapse onto obl:arg ∧ Dat).
#[test]
fn n01_remove_head_frame() {
    let corpus = fixtures::corpus();
    let w = world(&corpus);
    let mut d = RuleDesc::recipient();
    d.frame = None;
    let mut store = BindStore::new();
    register(
        &mut store,
        CLASS_COADDR,
        Physical::CoAddressed,
        std::slice::from_ref(&d),
        &w.v,
        &w.lex,
    )
    .unwrap();
    let r = MethodRef::of(
        CLASS_COADDR,
        &d.binding(Physical::CoAddressed, &w.v, &w.lex),
    );
    let mut c = compose(&store, &r, &d, Physical::CoAddressed, &w.v, &w.lex).unwrap();
    let m = c.run(&w.pop, &w.table, &w.frame, &all_plane(&w)).unwrap();
    assert_eq!(gids_of(&w, &m), vec![105, 302, 405, 505]);
    // and the stripped description no longer matches the bound guard
    let bound = mref(
        &w,
        CLASS_COADDR,
        &RuleDesc::recipient(),
        Physical::CoAddressed,
    );
    assert_eq!(
        compose(&w.store, &bound, &d, Physical::CoAddressed, &w.v, &w.lex).err(),
        Some(ComposeRefusal::DescriptionMismatch)
    );
}

/// 2. GIVE and EXPERIENCE frames swapped in the lexicon. PREDICTED:
///    Via — the bound Sym(digest) no longer matches → DescriptionMismatch;
///    CoAddressed — a population produced under the swapped lexicon (rev 2)
///    is refused by a method composed under rev 1; recomposed honestly under
///    rev 2, the answers swap (105 → experiencer).
#[test]
fn n02_swap_frames() {
    let corpus = fixtures::corpus();
    let w = world(&corpus);
    let swapped = Lexicon::swapped();
    let bound = mref(&w, CLASS_VIA, &RuleDesc::recipient(), Physical::Via);
    assert_eq!(
        compose(
            &w.store,
            &bound,
            &RuleDesc::recipient(),
            Physical::Via,
            &w.v,
            &swapped
        )
        .err(),
        Some(ComposeRefusal::DescriptionMismatch)
    );
    let ws = world_with(&corpus, swapped);
    let r = mref(
        &w,
        CLASS_COADDR,
        &RuleDesc::recipient(),
        Physical::CoAddressed,
    );
    let mut c = compose(
        &w.store,
        &r,
        &RuleDesc::recipient(),
        Physical::CoAddressed,
        &w.v,
        &w.lex,
    )
    .unwrap();
    assert_eq!(
        c.run(&ws.pop, &ws.table, &ws.frame, &all_plane(&ws)).err(),
        Some(RunRefusal::LexiconRevision {
            bound: 1,
            population: 2
        })
    );
    assert_eq!(
        answer(
            &ws,
            &RuleDesc::experiencer(),
            Physical::CoAddressed,
            &all_plane(&ws)
        ),
        vec![105, 405]
    );
    assert_eq!(
        answer(
            &ws,
            &RuleDesc::recipient(),
            Physical::CoAddressed,
            &all_plane(&ws)
        ),
        vec![302]
    );
}

/// 3. Dative → accusative on Kind (A). PREDICTED: no candidate for 105.
#[test]
fn n03_dative_to_accusative() {
    let mut a = A;
    a[4].case = "Acc";
    let w = world(&a);
    for p in BOTH {
        assert!(answer(&w, &RuleDesc::recipient(), p, &all_plane(&w)).is_empty());
    }
}

/// 4. Dependency reversed (Kind heads the verb). PREDICTED: no candidate.
#[test]
fn n04_reversed_dependency() {
    let mut a = A;
    a[2].head = 105;
    a[2].deprel = "obl:arg";
    a[2].case = "Dat";
    a[4].head = 0;
    a[4].deprel = "root";
    let w = world(&a);
    for p in BOTH {
        assert!(
            answer(&w, &RuleDesc::recipient(), p, &all_plane(&w)).is_empty(),
            "{p:?}"
        );
    }
}

/// 5. The referenced head removed from the population. PREDICTED: a typed
///    refusal at binding (Unmapped), never a FALSE role.
#[test]
fn n05_missing_head() {
    let a: Vec<Tok> = A.iter().copied().filter(|t| t.gid != 103).collect();
    let mut v = Vocab::new();
    let (table, frame) = tile(9, &fixtures::gids(&a));
    let r = FramedPopulation::produce(&a, &mut v, &table, &frame, &Lexicon::v0());
    assert!(matches!(r, Err(PopRefusal::Map(MapRefusal::Unmapped))));
}

/// 6. Stale (foreign) frame. PREDICTED: Stale refusal.
#[test]
fn n06_foreign_frame() {
    let w = world(&A);
    let (_t, foreign) = tile(9, &fixtures::gids(&A));
    let r = mref(
        &w,
        CLASS_COADDR,
        &RuleDesc::recipient(),
        Physical::CoAddressed,
    );
    let mut c = compose(
        &w.store,
        &r,
        &RuleDesc::recipient(),
        Physical::CoAddressed,
        &w.v,
        &w.lex,
    )
    .unwrap();
    assert!(matches!(
        c.run(&w.pop, &w.table, &foreign, &all_plane(&w)),
        Err(RunRefusal::Frame(PopRefusal::Read(MapRead::Stale(_))))
    ));
}

/// 6b. Same table, re-sealed under a permuted layout of the same size
///     (Codex #1948). PREDICTED: refused as `PopulationFrame`, never answered
///     with bits that name other gids.
#[test]
fn n06b_resealed_same_size_layout() {
    let mut w = world(&A);
    let mut gids = fixtures::gids(&A);
    gids.reverse();
    let map =
        lance_graph_dt64::dr::LocalMap::prepare(grammar_constraints_probe::WIDTH, &gids).unwrap();
    w.table.advance(9, 0, vec![(TILE, Some(map))]);
    let resealed = w.table.seal();
    assert_eq!(w.table.map(&resealed, TILE).unwrap().len(), w.pop.base.n);
    let r = mref(
        &w,
        CLASS_COADDR,
        &RuleDesc::recipient(),
        Physical::CoAddressed,
    );
    let mut c = compose(
        &w.store,
        &r,
        &RuleDesc::recipient(),
        Physical::CoAddressed,
        &w.v,
        &w.lex,
    )
    .unwrap();
    let plane = all_plane(&w);
    assert_eq!(
        c.run(&w.pop, &w.table, &resealed, &plane).err(),
        Some(RunRefusal::Frame(PopRefusal::PopulationFrame))
    );
    // control: the producing frame still runs
    assert!(c.run(&w.pop, &w.table, &w.frame, &plane).is_ok());
}

/// 7. Foreign classid. PREDICTED: UnknownClass from admit.
#[test]
fn n07_foreign_class() {
    let w = world(&A);
    let r = mref(
        &w,
        CLASS_FOREIGN,
        &RuleDesc::recipient(),
        Physical::CoAddressed,
    );
    assert_eq!(
        compose(
            &w.store,
            &r,
            &RuleDesc::recipient(),
            Physical::CoAddressed,
            &w.v,
            &w.lex
        )
        .err(),
        Some(ComposeRefusal::Binding(Refusal::UnknownClass(
            CLASS_FOREIGN
        )))
    );
}

/// 8. One supporting operand (deprel) removed. PREDICTED: against the bound
///    guard → DescriptionMismatch; under its own binding the instrument
///    Löffel (408, obl, Dat, under geben) becomes a recipient.
#[test]
fn n08_remove_operand() {
    let corpus = fixtures::corpus();
    let w = world(&corpus);
    let mut d = RuleDesc::recipient();
    d.deprel = None;
    let bound = mref(
        &w,
        CLASS_COADDR,
        &RuleDesc::recipient(),
        Physical::CoAddressed,
    );
    assert_eq!(
        compose(&w.store, &bound, &d, Physical::CoAddressed, &w.v, &w.lex).err(),
        Some(ComposeRefusal::DescriptionMismatch)
    );
    let mut store = BindStore::new();
    register(
        &mut store,
        CLASS_COADDR,
        Physical::CoAddressed,
        std::slice::from_ref(&d),
        &w.v,
        &w.lex,
    )
    .unwrap();
    let r = MethodRef::of(
        CLASS_COADDR,
        &d.binding(Physical::CoAddressed, &w.v, &w.lex),
    );
    let mut c = compose(&store, &r, &d, Physical::CoAddressed, &w.v, &w.lex).unwrap();
    let got = gids_of(
        &w,
        &c.run(&w.pop, &w.table, &w.frame, &all_plane(&w)).unwrap(),
    );
    assert!(got.contains(&408), "{got:?}");
}

/// 10. One global WHERE plane reused by both methods. PREDICTED: with W =
///     tokens of A and C only, recipient = {105}, experiencer = {302}; equal
///     to the unscoped answer ∩ W; W = all gives the unscoped answer.
#[test]
fn n10_shared_where_plane() {
    let corpus = fixtures::corpus();
    let w = world(&corpus);
    let ac: Vec<u64> = A.iter().chain(C.iter()).map(|t| t.gid).collect();
    let scoped = plane_of(&w, &ac);
    for p in BOTH {
        let rec = answer(&w, &RuleDesc::recipient(), p, &scoped);
        let exp = answer(&w, &RuleDesc::experiencer(), p, &scoped);
        assert_eq!(rec, vec![105]);
        assert_eq!(exp, vec![302]);
        let full = answer(&w, &RuleDesc::recipient(), p, &all_plane(&w));
        let inter: Vec<u64> = full.into_iter().filter(|g| ac.contains(g)).collect();
        assert_eq!(rec, inter);
    }
    // B's Schule is outside W: no rule here can name it
    let _ = B;
}

// --- physical form ---------------------------------------------------------------

#[test]
fn coaddressed_query_has_no_via_and_via_query_does() {
    let w = world(&fixtures::corpus());
    for (p, want_via) in [(Physical::Via, true), (Physical::CoAddressed, false)] {
        let f = RuleDesc::recipient().filter(p, &w.v, &w.lex);
        let dbg = format!("{f:?}");
        assert_eq!(dbg.contains("EqU32Via"), want_via, "{p:?}");
    }
    // the producer did the cross-row reads once, counted: one per non-root token
    let non_root = fixtures::corpus().iter().filter(|t| t.head != 0).count();
    assert_eq!(w.pop.producer_lookups, non_root);
}

#[test]
fn hot_run_allocates_nothing() {
    let w = world(&fixtures::corpus());
    let all = all_plane(&w);
    for p in BOTH {
        let r = mref(&w, class_of(p), &RuleDesc::recipient(), p);
        let mut c = compose(&w.store, &r, &RuleDesc::recipient(), p, &w.v, &w.lex).unwrap();
        let _ = c.run(&w.pop, &w.table, &w.frame, &all).unwrap();
        let before = BYTES.with(|b| b.get());
        for _ in 0..1000 {
            let _ = c.run(&w.pop, &w.table, &w.frame, &all).unwrap();
        }
        assert_eq!(BYTES.with(|b| b.get()) - before, 0, "{p:?}");
    }
}
