//! D-GRAMMAR-CONSTRAINTS-0 falsifier: German A/B/C (+ controls D, E) and the
//! eight negative controls of the brief, each with a predicted outcome.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use grammar_constraints_probe::fixtures::{self, Gold, A, B, C, GOLD};
use grammar_constraints_probe::*;
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_dt64::dr::{Frame, MapRead, MapTable};
use lance_graph_dt64::Guard;

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

// --- harness ----------------------------------------------------------------

struct World {
    table: MapTable,
    frame: Frame,
    pop: Population,
    rules: Rules,
}

fn world_with(toks: &[Tok], ops: Operands) -> World {
    let (table, frame) = tile(7, &fixtures::gids(toks));
    let mut v = Vocab::new();
    let ids = Ids::bind(&mut v);
    let pop = Population::bind(toks, &mut v, &table, &frame).unwrap();
    World {
        table,
        frame,
        pop,
        rules: Rules::prepare(&ids, ops),
    }
}

fn world(toks: &[Tok]) -> World {
    world_with(toks, Operands::ALL)
}

/// What the rules said about one global id.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
struct Said {
    recipient: bool,
    experiencer: bool,
    dir_or_purp: bool,
    abstain: bool,
    unknown: bool,
}

impl Said {
    fn any_role(&self) -> bool {
        self.recipient || self.experiencer || self.dir_or_purp
    }
}

fn said(w: &mut World, gid: u64) -> Said {
    let r = w.rules.run(&w.pop, &w.table, &w.frame).unwrap();
    let local = w.table.map(&w.frame, TILE).unwrap().local(gid).unwrap();
    Said {
        recipient: has(&r.recipient, local),
        experiencer: has(&r.experiencer, local),
        dir_or_purp: has(&r.direction_or_purpose, local),
        abstain: has(&r.abstain_frame, local),
        unknown: has(&r.unknown_case, local),
    }
}

/// Every gid that got any role.
fn roled(w: &mut World, toks: &[Tok]) -> Vec<u64> {
    toks.iter()
        .map(|t| t.gid)
        .filter(|&g| said(w, g).any_role())
        .collect()
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Score {
    correct: u32,
    false_pos: u32,
    false_neg: u32,
    abstained: u32,
    ambiguous_kept: u32,
}

fn score(w: &mut World, toks: &[Tok]) -> Score {
    let mut s = Score::default();
    for t in toks {
        let got = said(w, t.gid);
        let gold = GOLD.iter().find(|(g, _)| *g == t.gid).map(|&(_, r)| r);
        let hit = match gold {
            Some(Gold::Recipient) => got.recipient,
            Some(Gold::Experiencer) => got.experiencer,
            Some(Gold::Direction) => got.dir_or_purp,
            _ => false,
        };
        let wrong = (got.recipient && gold != Some(Gold::Recipient))
            || (got.experiencer && gold != Some(Gold::Experiencer))
            || (got.dir_or_purp && gold != Some(Gold::Direction));
        if hit {
            s.correct += 1;
        }
        if wrong {
            s.false_pos += 1;
        }
        if matches!(
            gold,
            Some(Gold::Recipient | Gold::Experiencer | Gold::Direction)
        ) && !got.any_role()
            && !(got.abstain || got.unknown)
        {
            s.false_neg += 1;
        }
        if got.abstain || got.unknown {
            s.abstained += 1;
        }
        if got.dir_or_purp {
            s.ambiguous_kept += 1; // Direction AND Purpose both kept
        }
    }
    s
}

// --- P0: the three sentences plus D, E ---------------------------------------

#[test]
fn p0_scores_against_independent_gold() {
    let corpus = fixtures::corpus();
    let mut w = world(&corpus);
    let s = score(&mut w, &corpus);
    assert_eq!(
        s,
        Score {
            correct: 4,
            false_pos: 0,
            false_neg: 0,
            abstained: 1,
            ambiguous_kept: 1
        }
    );
    // A: recipient; C: experiencer; B: direction-or-purpose, never recipient.
    assert!(said(&mut w, 105).recipient && !said(&mut w, 105).experiencer);
    assert!(said(&mut w, 302).experiencer && !said(&mut w, 302).recipient);
    let b = said(&mut w, 208);
    assert!(b.dir_or_purp && !b.recipient && !b.experiencer);
    // D: the dative instrument under GIVE gets nothing.
    assert_eq!(said(&mut w, 408), Said::default());
    // E: helfen has no frame: UNKNOWN (abstain), not FALSE, not a guess.
    let e = said(&mut w, 505);
    assert!(e.abstain && !e.any_role());
}

#[test]
fn surface_case_alone_does_not_determine_the_role() {
    // Kind in A and Kind in C have identical own lanes (obl:arg, Dat).
    let (a, c) = (A[4], C[1]);
    assert_eq!((a.deprel, a.case), (c.deprel, c.case));
    let corpus = fixtures::corpus();
    let mut w = world(&corpus);
    assert_ne!(said(&mut w, a.gid), said(&mut w, c.gid));
}

// --- negative controls (brief §5) --------------------------------------------

/// Control 1: GIVE → LIKE with the dative marking unchanged.
#[test]
fn c1_give_replaced_by_like() {
    let mut a = A;
    a[2].lemma = "gefallen";
    let mut w = world(&a);
    let k = said(&mut w, 105);
    assert!(!k.recipient && k.experiencer, "{k:?}");
}

/// Control 2: The verb-frame constraint removed: predicted false positives in C and E.
#[test]
fn c2_without_the_frame_constraint() {
    let corpus = fixtures::corpus();
    let mut w = world_with(
        &corpus,
        Operands {
            frame: false,
            ..Operands::ALL
        },
    );
    let s = score(&mut w, &corpus);
    // Recipient fires on 302 and 505, experiencer on 105, 405, 505.
    assert!(said(&mut w, 302).recipient && said(&mut w, 105).experiencer);
    assert_eq!(s.false_pos, 4, "{s:?}"); // 105, 302, 405, 505
}

/// Control 3: Dependency direction swapped (Kind heads the verb).
#[test]
fn c3_swapped_dependency_direction() {
    let mut a = A;
    a[2].head = 105;
    a[2].deprel = "obl:arg";
    a[2].case = "";
    a[4].head = 0;
    a[4].deprel = "root";
    let mut w = world(&a);
    assert!(roled(&mut w, &a).is_empty());
}

/// Control 4: Only surface word order changes: "Das Buch gibt die Mutter dem Kind."
#[test]
fn c4_word_order_changes_nothing() {
    let reordered = [A[5], A[6], A[2], A[0], A[1], A[3], A[4], A[7]];
    let mut w1 = world(&A);
    let mut w2 = world(&reordered);
    for t in &A {
        assert_eq!(said(&mut w1, t.gid), said(&mut w2, t.gid), "{}", t.form);
    }
    // anti-vacuity: the local position of Kind really moved
    let l1 = w1.table.map(&w1.frame, TILE).unwrap().local(105).unwrap();
    let l2 = w2.table.map(&w2.frame, TILE).unwrap().local(105).unwrap();
    assert_ne!(l1, l2);
}

/// Control 5: ambiguous attachment: "zur Schule" under the verb (obl) or under
/// "Buch" (nmod). Two readings, two populations; neither is merged away.
#[test]
fn c5_ambiguous_attachment_keeps_both_readings() {
    let verb_reading = B;
    let mut noun_reading = B;
    noun_reading[7].deprel = "nmod";
    noun_reading[7].head = 205;
    let mut w1 = world(&verb_reading);
    let mut w2 = world(&noun_reading);
    assert!(said(&mut w1, 208).dir_or_purp);
    assert!(!said(&mut w2, 208).dir_or_purp);
}

/// Control 6: A supporting observation deleted: Kind's Case feature removed.
#[test]
fn c6_missing_case_is_unknown_not_false() {
    let mut a = A;
    a[4].case = "";
    let mut w = world(&a);
    let k = said(&mut w, 105);
    assert!(!k.recipient && k.unknown && !k.abstain, "{k:?}");
    // and deleting the zu case marker removes the goal candidate
    let b: Vec<Tok> = B.iter().copied().filter(|t| t.gid != 206).collect();
    let mut wb = world(&b);
    assert!(!said(&mut wb, 208).dir_or_purp);
    // and Schule without its Case feature is not asserted a goal either
    let mut b2 = B;
    b2[7].case = "";
    let mut wb2 = world(&b2);
    assert!(!said(&mut wb2, 208).dir_or_purp);
}

/// Control 7: A stale (foreign) coordinate frame is refused.
#[test]
fn c7_foreign_frame_refuses() {
    let mut w = world(&A);
    let (_other, foreign) = tile(7, &fixtures::gids(&A));
    assert!(matches!(
        w.rules.run(&w.pop, &w.table, &foreign),
        Err(Refusal::Read(MapRead::Stale(_)))
    ));
}

/// Control 8: one required relational operand disabled: deprel. The dative
/// instrument `mit dem Löffel` (obl, Dat, under geben) becomes a recipient.
#[test]
fn c8_disabled_deprel_operand() {
    let corpus = fixtures::corpus();
    let mut w = world_with(
        &corpus,
        Operands {
            deprel: false,
            ..Operands::ALL
        },
    );
    assert!(said(&mut w, 408).recipient);
    // and the article "dem" (det, Dat, head Kind) is NOT, because its head is
    // not a GIVE verb: the frame hop still holds
    assert!(!said(&mut w, 404).recipient);
}

// --- WHEN expressibility (DT64 Guard / SetExpr) -------------------------------

#[test]
fn when_guard_cannot_state_the_head_frame() {
    // The dependent's own conditions are a SetExpr over two lanes.
    let own = SetExpr::And(vec![SetExpr::eq_u32(DEPREL, 1), SetExpr::eq_u32(CASE, 2)]);
    let recipient = Guard::new(own.clone(), Logic::TwoValued);
    let experiencer = Guard::new(own, Logic::TwoValued);
    // With only single-population leaves, the two rules have the SAME guard
    // witness: WHEN cannot tell A from C. The frame lives one hop away (the
    // head's lemma) and SetExpr has no hop leaf; only an opaque Sym(k), whose
    // meaning the consumer supplies, could stand in for it.
    assert_eq!(recipient.witness(), experiencer.witness());
    let with_sym = |k| {
        Guard::new(
            SetExpr::And(vec![SetExpr::eq_u32(DEPREL, 1), SetExpr::Sym(k)]),
            Logic::TwoValued,
        )
    };
    assert_ne!(with_sym(1).witness(), with_sym(2).witness());
}

// --- cost ---------------------------------------------------------------------

#[test]
fn hot_run_allocates_nothing() {
    let corpus = fixtures::corpus();
    let mut w = world(&corpus);
    let _ = w.rules.run(&w.pop, &w.table, &w.frame).unwrap();
    let before = BYTES.with(|b| b.get());
    let mut n = 0u32;
    for _ in 0..1000 {
        let r = w.rules.run(&w.pop, &w.table, &w.frame).unwrap();
        n += r.recipient.iter().map(|x| x.count_ones()).sum::<u32>();
    }
    assert_eq!(n, 2000); // 105 and 405, every run
    assert_eq!(BYTES.with(|b| b.get()) - before, 0);
}
