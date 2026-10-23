//! D-TGA-5D / Probe D: protected readers of a squash.
//!
//! Pre-registered in
//! `.claude/board/entries/2026-10-23-d-tga-5d-protected-readers-preregistration.md`
//! before this file was written. Predictions D1..D5 are the tests below; each
//! disable is named on the test it must turn red.

use thought_git_probe::fixtures::{Cell, Outcome, Thought, cell, givens, main_version};
use thought_git_probe::oracle::{Givens, Version, next_version};
use thought_git_probe::squash::{
    Lane, ProtectedRefusal, Reader, bytes, history_contradictions, history_divergences,
    history_multiplicity, land_interleaved, protected_squash, squashed_multiplicity,
};

fn commit(id: u32, writes: Vec<(Cell, u64)>) -> Thought {
    Thought::new(id, writes, Outcome::Commit)
}

/// (a) Prune, Veto and Plan writes between commits.
fn fixture_a() -> Vec<Lane> {
    vec![Lane::new(
        3,
        vec![
            commit(1, vec![(cell(4, 1), 41)]),
            Thought::new(2, vec![(cell(5, 1), 51)], Outcome::Prune),
            Thought::new(3, vec![(cell(4, 1), 99)], Outcome::Veto),
            Thought::new(4, vec![(cell(6, 1), 61)], Outcome::Plan),
            commit(5, vec![(cell(7, 2), 72)]),
        ],
    )]
}

/// (b) A committed write contradicting the given (1, 0) = 101, then a write
/// agreeing with it.
fn fixture_b() -> Vec<Lane> {
    vec![Lane::new(
        5,
        vec![
            commit(1, vec![(cell(1, 0), 999), (cell(4, 2), 42)]),
            commit(2, vec![(cell(1, 0), 101)]),
        ],
    )]
}

/// (c) Two lanes diverge on (5, 2) at step 1 and agree at step 2.
fn fixture_c() -> Vec<Lane> {
    vec![
        Lane::new(
            4,
            vec![
                commit(1, vec![(cell(5, 2), 7)]),
                commit(3, vec![(cell(5, 2), 9)]),
            ],
        ),
        Lane::new(
            9,
            vec![
                commit(2, vec![(cell(5, 2), 8)]),
                commit(4, vec![(cell(5, 2), 9)]),
            ],
        ),
    ]
}

/// (d) Lane 2 writes (6, 3) = 63 three times; lane 8 writes it once.
fn fixture_d() -> Vec<Lane> {
    vec![
        Lane::new(
            2,
            vec![
                commit(1, vec![(cell(6, 3), 63)]),
                commit(2, vec![(cell(6, 3), 63)]),
                commit(3, vec![(cell(6, 3), 63)]),
            ],
        ),
        Lane::new(8, vec![commit(4, vec![(cell(6, 3), 63)])]),
    ]
}

/// (e) Clean lanes over a non-empty prior version.
fn fixture_e() -> (Version, Vec<Lane>) {
    let prior = next_version(
        &givens(),
        &main_version(),
        &[commit(90, vec![(cell(11, 2), 112), (cell(4, 1), 1)])],
    );
    let lanes = vec![
        Lane::new(
            3,
            vec![
                commit(1, vec![(cell(4, 1), 41)]),
                commit(2, vec![(cell(6, 2), 62)]),
            ],
        ),
        Lane::new(
            7,
            vec![
                commit(3, vec![(cell(5, 1), 50)]),
                commit(4, vec![(cell(5, 1), 52), (cell(6, 2), 62)]),
            ],
        ),
    ];
    (prior, lanes)
}

fn all_fixtures() -> Vec<(&'static str, Version, Vec<Lane>)> {
    let (pe, le) = fixture_e();
    vec![
        ("a", main_version(), fixture_a()),
        ("b", main_version(), fixture_b()),
        ("c", main_version(), fixture_c()),
        ("d", main_version(), fixture_d()),
        ("e", pe, le),
    ]
}

const LOGS: [Reader; 3] = [
    Reader::Contradictions,
    Reader::Divergences,
    Reader::Multiplicity,
];

fn g() -> Givens {
    givens()
}

/// D1. Final reads survive on every fixture, with an empty witness.
/// Disable: land the squash on `main_version()` instead of `prior` in
/// `protected_squash` -> (e) refuses with `Unretainable(Final)`.
#[test]
fn d1_final_reads_survive_with_an_empty_witness() {
    for (name, prior, lanes) in all_fixtures() {
        let (s, w) = protected_squash(&g(), &prior, &lanes, &[Reader::Final])
            .unwrap_or_else(|e| panic!("fixture {name}: {e:?}"));
        assert_eq!(w, Default::default(), "fixture {name}: witness not empty");
        // Independent of the gate's own check: land and compare here.
        let history = land_interleaved(&g(), &prior, &lanes);
        let last = history.last().expect("non-empty history");
        let landed = next_version(&g(), &prior, &[s.as_thought(7777)]);
        for r in 0..12 {
            for l in 0..4 {
                let c = cell(r, l);
                assert_eq!(landed.read(&g(), c), last.read(&g(), c), "{name} {c:?}");
            }
        }
    }
    // Anti-vacuity: the prior's untouched cell survives in (e).
    let (prior, lanes) = fixture_e();
    let (s, _) = protected_squash(&g(), &prior, &lanes, &[Reader::Final]).expect("e");
    assert_eq!(
        next_version(&g(), &prior, &[s.as_thought(1)]).read(&g(), cell(11, 2)),
        Some(112)
    );
}

/// D2. Without a witness the squash loses the logs. The loss is real: the
/// history has a contradiction in (b), a divergence in (c) and a 4-write
/// cell in (d), and the bare squash has none of them.
#[test]
fn d2_a_bare_squash_loses_contradictions_divergences_and_multiplicity() {
    let (s, w) = protected_squash(&g(), &main_version(), &fixture_b(), &[Reader::Final])
        .expect("b squashes");
    let h = history_contradictions(&land_interleaved(&g(), &main_version(), &fixture_b()));
    assert_eq!(h.len(), 1, "fixture b must record a contradiction");
    assert!(w.contradictions.is_empty());
    assert_ne!(w.contradictions, h, "the loss of b's contradiction");
    let _ = s;

    let (_, w) = protected_squash(&g(), &main_version(), &fixture_c(), &[Reader::Final])
        .expect("c squashes");
    let h = history_divergences(&land_interleaved(&g(), &main_version(), &fixture_c()));
    assert_eq!(h.len(), 1, "fixture c must record a divergence");
    assert_ne!(w.divergences, h, "the loss of c's divergence");

    let (s, _) = protected_squash(&g(), &main_version(), &fixture_d(), &[Reader::Final])
        .expect("d squashes");
    let h = history_multiplicity(&fixture_d());
    assert_eq!(h[&cell(6, 3)], 4, "fixture d writes (6, 3) four times");
    assert_eq!(
        squashed_multiplicity(&s)[&cell(6, 3)],
        2,
        "blame counts two lanes"
    );
}

/// D3. With the logs protected the witness keeps them and every retained
/// answer equals the history's, on every fixture. Disables, each turning one
/// fixture red (the gate refuses with `Unretainable`): skip filling
/// `contradictions` (b), `divergences` (c), `multiplicity` (d).
#[test]
fn d3_a_protected_squash_keeps_every_log_it_promises() {
    for (name, prior, lanes) in all_fixtures() {
        let mut protect = vec![Reader::Final];
        protect.extend(LOGS);
        let (_, w) = protected_squash(&g(), &prior, &lanes, &protect)
            .unwrap_or_else(|e| panic!("fixture {name}: {e:?}"));
        let history = land_interleaved(&g(), &prior, &lanes);
        assert_eq!(w.contradictions, history_contradictions(&history), "{name}");
        assert_eq!(w.divergences, history_divergences(&history), "{name}");
        assert_eq!(w.multiplicity, history_multiplicity(&lanes), "{name}");
    }
    // Anti-vacuity: the witness is non-empty exactly where the history is.
    let protect = [Reader::Contradictions, Reader::Divergences];
    let (_, wb) = protected_squash(&g(), &main_version(), &fixture_b(), &protect).expect("b");
    let (_, wc) = protected_squash(&g(), &main_version(), &fixture_c(), &protect).expect("c");
    assert_eq!((wb.contradictions.len(), wb.divergences.len()), (1, 0));
    assert_eq!((wc.contradictions.len(), wc.divergences.len()), (0, 1));
}

/// D4. Intermediate reads cannot be retained: when one differs from the
/// final read and `Intermediate` is protected, the gate refuses, and only
/// then. The expectation is computed here, independently of the gate.
/// Disable: remove the intermediate check -> this test is red.
#[test]
fn d4_a_protected_intermediate_read_that_differs_refuses_the_squash() {
    let mut refused = 0;
    for (name, prior, lanes) in all_fixtures() {
        let history = land_interleaved(&g(), &prior, &lanes);
        let last = history.last().expect("non-empty history");
        let differs = history[..history.len() - 1].iter().any(|v| {
            (0..12).any(|r| (0..4).any(|l| v.read(&g(), cell(r, l)) != last.read(&g(), cell(r, l))))
        });
        let got = protected_squash(&g(), &prior, &lanes, &[Reader::Intermediate]).err();
        if differs {
            refused += 1;
            assert_eq!(
                got,
                Some(ProtectedRefusal::Unretainable(Reader::Intermediate)),
                "fixture {name}"
            );
        } else {
            assert_eq!(
                got, None,
                "fixture {name} has no differing intermediate read"
            );
        }
    }
    // Anti-vacuity: most fixtures do lose an intermediate read.
    assert!(refused >= 3, "only {refused} fixtures exercise the refusal");
}

/// D4 silent twins: a one-step history, and a multi-step history whose
/// intermediate reads all equal the final, are accepted with `Intermediate`
/// protected.
#[test]
fn d4_silent_twins_accept_when_no_intermediate_read_differs() {
    let one = vec![
        Lane::new(3, vec![commit(1, vec![(cell(4, 1), 41)])]),
        Lane::new(7, vec![commit(2, vec![(cell(5, 1), 51)])]),
    ];
    assert!(protected_squash(&g(), &main_version(), &one, &[Reader::Intermediate]).is_ok());

    let steady = vec![Lane::new(
        3,
        vec![
            commit(1, vec![(cell(4, 1), 41)]),
            commit(2, vec![(cell(4, 1), 41)]),
            commit(3, vec![(cell(4, 1), 41)]),
        ],
    )];
    assert_eq!(land_interleaved(&g(), &main_version(), &steady).len(), 3);
    assert!(protected_squash(&g(), &main_version(), &steady, &[Reader::Intermediate]).is_ok());
}

/// R1 and R2 are unchanged underneath: a protected squash still refuses
/// what the plain gate refuses.
#[test]
fn the_gate_refusals_pass_through() {
    let div = vec![
        Lane::new(4, vec![commit(1, vec![(cell(5, 2), 7)])]),
        Lane::new(9, vec![commit(2, vec![(cell(5, 2), 8)])]),
    ];
    assert!(matches!(
        protected_squash(&g(), &main_version(), &div, &[Reader::Final]),
        Err(ProtectedRefusal::Gate(_))
    ));
    let contra = vec![Lane::new(5, vec![commit(1, vec![(cell(1, 0), 999)])])];
    assert!(matches!(
        protected_squash(&g(), &main_version(), &contra, &[Reader::Final]),
        Err(ProtectedRefusal::Gate(_))
    ));
}

/// D5. Bytes, reported not gated: transient grows with the writes; the
/// squash grows with the order list (one entry per thought) and the distinct
/// cells; the witness grows with the anomalies and the multiplicity cells.
#[test]
fn d5_retained_and_transient_bytes() {
    const N: u32 = 8;
    let mut rows = Vec::new();
    for k in [1u32, 4, 16, 64] {
        let chain: Vec<Thought> = (0..k)
            .map(|round| {
                commit(
                    round,
                    (0..N)
                        .map(|i| (cell(4 + i, 1), u64::from(round * 100 + i)))
                        .collect(),
                )
            })
            .collect();
        let lanes = vec![Lane::new(1, chain)];
        let mut protect = vec![Reader::Final];
        protect.extend(LOGS);
        let (s, w) = protected_squash(&g(), &main_version(), &lanes, &protect).expect("squash");
        let b = bytes(&lanes, &s, &w);
        assert_eq!(b.transient, (k * N) as usize * 13, "transient at k = {k}");
        assert_eq!(
            b.witness,
            N as usize * 9,
            "witness at k = {k}: N multiplicity cells"
        );
        rows.push((k, b.transient, b.squash, b.witness));
    }
    // The squash grows only through the order list: 5 bytes per thought.
    assert_eq!(rows[1].2 - rows[0].2, 3 * 5);
    println!("k, transient B, squash B, witness B: {rows:?}");
}
