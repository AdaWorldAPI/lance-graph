//! The D-RPR-1 handoff probe (`chain`) against oracles that use no index, no
//! focus and no prepared operand.

use std::collections::BTreeSet;

use lance_graph_mask_risc::{MaskOp, Operand, Program, Terminal};
use mork_pathmap_probe::chain::*;
use mork_pathmap_probe::rmap::*;

const PREDS: [u32; 3] = [0, 1, 2];
/// The three questions: predicate and hop budget. Each is seeded with the
/// previous answer.
const QUESTIONS: [(u32, usize); 3] = [(0, 3), (1, 1), (2, 1)];
const SEED: u32 = 17;
/// A seed in another community (communities are 512 identities wide). The
/// filler around it holds nothing the chain needs, so every question must
/// bring in what it asks about.
const FAR: u32 = 40_000;

/// Global reachability from `seeds` over `preds` within `hops`, by scanning
/// every fact.
fn oracle(facts: &[Fact], preds: &[u32], seeds: &BTreeSet<u32>, hops: usize) -> BTreeSet<u32> {
    let mut reached = seeds.clone();
    let mut frontier = seeds.clone();
    for _ in 0..hops {
        let next: BTreeSet<u32> = facts
            .iter()
            .filter(|f| frontier.contains(&f[0]) && preds.contains(&f[1]))
            .map(|f| f[2])
            .filter(|o| !reached.contains(o))
            .collect();
        if next.is_empty() {
            break;
        }
        reached.extend(next.iter().copied());
        frontier = next;
    }
    reached
}

fn facts() -> Vec<Fact> {
    graph(65_536, 3, 2, 512, 0.9, 0x5EED)
}

fn cur() -> Witness {
    Witness {
        facts: 1,
        seal: 1,
        focus: 0,
        scope: 7,
    }
}

/// The seed plus 255 identities around `FAR`.
fn first_focus(run: &SortedRun<'_>) -> Focus {
    let filler = discover(run, &[FAR], &PREDS, SLOTS);
    let mut ids = vec![SEED];
    ids.extend(
        filler
            .ids()
            .iter()
            .copied()
            .filter(|&g| g != SEED)
            .take(SLOTS - 1),
    );
    Focus::bind(&ids, 1)
}

struct Chain {
    answers: Vec<Answer>,
    frame: Frame,
    initial_rows: BTreeSet<u32>,
}

fn run_chain(facts: &[Fact], how: Rebuild) -> Chain {
    let run = SortedRun::build(facts, false, 1);
    let mut frame = Frame::prepare(&run, first_focus(&run), &PREDS, &cur());
    let initial_rows = frame.csr.edge_set().into_iter().map(|e| e.3).collect();
    let mut exec = TileExec::new();
    let mut keep = BTreeSet::new();
    let mut seeds = BTreeSet::from([SEED]);
    let mut answers = Vec::new();
    for (p, hops) in QUESTIONS {
        let a = ask(
            &mut frame,
            &mut exec,
            &run,
            &cur(),
            &[p],
            &seeds,
            hops,
            &mut keep,
            how,
        )
        .unwrap();
        seeds = a.reached.clone();
        answers.push(a);
    }
    Chain {
        answers,
        frame,
        initial_rows,
    }
}

#[test]
fn every_step_equals_the_global_oracle_and_needed_a_lookup() {
    let facts = facts();
    let c = run_chain(&facts, Rebuild::Incremental);
    let mut seeds = BTreeSet::from([SEED]);
    for (i, ((p, hops), a)) in QUESTIONS.iter().zip(&c.answers).enumerate() {
        assert_eq!(a.reached, oracle(&facts, &[*p], &seeds, *hops), "step {i}");
        // Anti-vacuity, per step: the frame did not already hold the answer.
        // At least one fold was refused as incomplete, and every admission
        // brought in identities the frame lacked.
        assert!(!a.admissions.is_empty(), "step {i} needed no lookup");
        assert!(a.folds >= 2, "step {i} was never refused as incomplete");
        assert!(a.admissions.iter().all(|x| !x.demanded.is_empty()));
        assert!(
            a.reached.len() > seeds.len(),
            "step {i} reached nothing new"
        );
        seeds = a.reached.clone();
    }
}

/// The previous answer's addresses are the next question's seeds, and they
/// are already co-addressed: no lookup is made for them. Every lookup a later
/// step makes is for identities its own fold produced as escapes, never for
/// its seeds.
#[test]
fn the_previous_answer_seeds_the_next_step_without_a_lookup() {
    let facts = facts();
    let c = run_chain(&facts, Rebuild::Incremental);
    for i in 1..c.answers.len() {
        let seeds = &c.answers[i - 1].reached;
        for x in &c.answers[i].admissions {
            assert!(
                x.demanded.is_disjoint(seeds),
                "step {i} looked up one of its own seeds"
            );
        }
    }
}

/// Incremental and full rebuilds give the same answers and the same prepared
/// geometry; the incremental one keeps canonical rows from the first
/// preparation and seeks less.
#[test]
fn incremental_equals_full_preparation_and_seeks_less() {
    let facts = facts();
    let inc = run_chain(&facts, Rebuild::Incremental);
    let full = run_chain(&facts, Rebuild::Full);
    assert_eq!(
        inc.answers.iter().map(|a| &a.reached).collect::<Vec<_>>(),
        full.answers.iter().map(|a| &a.reached).collect::<Vec<_>>()
    );
    assert_eq!(inc.frame.focus.ids(), full.frame.focus.ids());
    assert_eq!(inc.frame.csr.edge_set(), full.frame.csr.edge_set());
    assert_eq!(inc.frame.csr.escape_set(), full.frame.csr.escape_set());
    for &p in &PREDS {
        assert_eq!(inc.frame.tile.plane(p), full.frame.tile.plane(p));
    }
    let rows: BTreeSet<u32> = inc.frame.csr.edge_set().into_iter().map(|e| e.3).collect();
    let reused = rows.intersection(&inc.initial_rows).count();
    let resolved = rows.difference(&inc.initial_rows).count();
    assert!(
        reused > 0 && resolved > 0,
        "reused {reused}, resolved {resolved}"
    );
    let seeks = |c: &Chain| -> usize {
        c.answers
            .iter()
            .flat_map(|a| &a.admissions)
            .map(|x| x.seeks)
            .sum()
    };
    // Full counts its seeks through the index; incremental's are the
    // documented `changed × preds`. Equal changes, so the ratio is exact.
    assert!(
        seeks(&inc) < seeks(&full),
        "{} vs {}",
        seeks(&inc),
        seeks(&full)
    );
    for a in &full.answers {
        for x in &a.admissions {
            assert_eq!(x.seeks, x.full_seeks, "a full prepare seeks every slot");
        }
    }
}

/// Each answer still holds on the frame the whole chain left: re-asking the
/// first question at the end admits nothing and folds once.
#[test]
fn an_earlier_answer_survives_the_later_steps() {
    let facts = facts();
    let mut c = run_chain(&facts, Rebuild::Incremental);
    let run = SortedRun::build(&facts, false, 1);
    let mut exec = TileExec::new();
    let again = ask(
        &mut c.frame,
        &mut exec,
        &run,
        &cur(),
        &[0],
        &BTreeSet::from([SEED]),
        3,
        &mut BTreeSet::new(),
        Rebuild::Incremental,
    )
    .unwrap();
    assert_eq!(again.reached, c.answers[0].reached);
    assert!(again.admissions.is_empty());
    assert_eq!(again.folds, 1);
}

/// The fold is co-addressed: every program the tile executor runs reads planes
/// of one `(target, source)` address space. Silence twin: the check fires on
/// a program with a `Gather`.
#[test]
fn the_fold_program_has_no_gather_and_no_scatter() {
    for n in 1..=PREDS.len() {
        for delta in [false, true] {
            assert!(
                fold_program_is_co_addressed(n, delta),
                "n {n} delta {delta}"
            );
        }
    }
    let gather = Program::new(
        vec![MaskOp::Gather {
            lane: 0,
            foreign: 0,
            under: None,
            dst: 0,
        }],
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    );
    assert!(!is_co_addressed(&gather), "the check must see a Gather");
}

/// A frame built before a change of the facts or the scope is refused, not
/// answered.
#[test]
fn a_changed_world_or_scope_refuses_the_frame() {
    let facts = facts();
    let run = SortedRun::build(&facts, false, 1);
    for (changed, want) in [
        (Witness { facts: 2, ..cur() }, Refusal::FactsChanged),
        (Witness { seal: 2, ..cur() }, Refusal::IndexResealed),
        (Witness { scope: 8, ..cur() }, Refusal::ScopeChanged),
    ] {
        let mut c = run_chain(&facts, Rebuild::Incremental);
        let mut exec = TileExec::new();
        let r = ask(
            &mut c.frame,
            &mut exec,
            &run,
            &changed,
            &[0],
            &BTreeSet::from([SEED]),
            3,
            &mut BTreeSet::new(),
            Rebuild::Incremental,
        );
        assert_eq!(r, Err(ChainRefusal::Stale(want)));
    }
}

/// The demanded identities must fit beside the kept ones; otherwise the
/// question is refused. Silence twin: one evictable slot is enough.
#[test]
fn a_focus_too_small_for_the_answer_is_refused() {
    // 0 -> 1 -> 2: from 0, two hops need {0, 1, 2}.
    let facts: Vec<Fact> = vec![[0, 0, 1], [1, 0, 2]];
    let run = SortedRun::build(&facts, false, 1);
    let mut exec = TileExec::new();
    let mut frame = Frame::prepare(&run, Focus::bind(&[0, 1], 1), &[0], &cur());
    let r = ask(
        &mut frame,
        &mut exec,
        &run,
        &cur(),
        &[0],
        &BTreeSet::from([0]),
        2,
        &mut BTreeSet::new(),
        Rebuild::Incremental,
    );
    assert_eq!(r, Err(ChainRefusal::FocusExhausted { needed: 1, free: 0 }));
    let mut frame = Frame::prepare(&run, Focus::bind(&[0, 1, 50], 1), &[0], &cur());
    let a = ask(
        &mut frame,
        &mut exec,
        &run,
        &cur(),
        &[0],
        &BTreeSet::from([0]),
        2,
        &mut BTreeSet::new(),
        Rebuild::Incremental,
    )
    .unwrap();
    assert_eq!(a.reached, BTreeSet::from([0, 1, 2]));
    assert_eq!(a.admissions.len(), 1);
    assert_eq!(a.admissions[0].demanded, BTreeSet::from([2]));
}

/// 64 thoughts over the frame the chain produced: each retracts the edges out
/// of one slot the first answer reached. Each answer equals the oracle over
/// its edited facts inside the focus; another thought id is refused; the
/// shared tile is unchanged.
#[test]
fn thoughts_run_over_the_frame_the_chain_produced() {
    let facts = facts();
    let c = run_chain(&facts, Rebuild::Incremental);
    let f = &c.frame;
    let w = Witness {
        focus: f.focus.generation,
        ..cur()
    };
    let seeds = slot_set([f.focus.slot_of(SEED).unwrap()]);
    let mut ex = TileExec::new();
    let base = ex
        .reach(&f.tile, &f.csr, &w, None, None, &[0], &seeds, 3)
        .unwrap();
    let reached: Vec<u8> = slots_of(&base.reached);
    let inf: BTreeSet<u32> = f.focus.ids().iter().copied().collect();
    let mut changed = 0;
    for t in 0..64u64 {
        let victim = reached[t as usize % reached.len()];
        let del: Vec<(u8, u8)> = f
            .csr
            .edge_set()
            .into_iter()
            .filter(|e| e.0 == 0 && e.1 == victim)
            .map(|e| (victim, e.2))
            .collect();
        let delta = ThoughtDelta::new(t, 0, &[], &del);
        let got = ex
            .reach(&f.tile, &f.csr, &w, Some(&delta), Some(t), &[0], &seeds, 3)
            .unwrap();
        let g = f.focus.id(victim);
        let edited: Vec<Fact> = facts
            .iter()
            .copied()
            .filter(|x| !(x[0] == g && x[1] == 0 && inf.contains(&x[2])))
            .filter(|x| inf.contains(&x[0]) && inf.contains(&x[2]))
            .collect();
        let want = oracle(&edited, &[0], &BTreeSet::from([SEED]), 3);
        let got_ids: BTreeSet<u32> = slots_of(&got.reached)
            .into_iter()
            .map(|s| f.focus.id(s))
            .collect();
        assert_eq!(got_ids, want, "thought {t}");
        changed += usize::from(got.reached != base.reached);
        assert_eq!(
            ex.reach(
                &f.tile,
                &f.csr,
                &w,
                Some(&delta),
                Some(t + 1),
                &[0],
                &seeds,
                3
            )
            .err(),
            Some(Refusal::ThoughtMismatch)
        );
    }
    assert!(
        changed > 0,
        "some retraction must change its thought's answer"
    );
    let again = ex
        .reach(&f.tile, &f.csr, &w, None, None, &[0], &seeds, 3)
        .unwrap();
    assert_eq!(again.reached, base.reached, "the shared tile is unchanged");
}
