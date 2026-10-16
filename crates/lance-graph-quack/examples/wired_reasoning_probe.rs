//! The field rules of `deepnsm_v2::wire` as Datalog over presence planes,
//! executed through quack and mask-risc, against a row oracle.
//!
//! One row = one wired triple. Every input fact and every output field is a
//! presence plane over the rows; a rule is a quack filter over those planes,
//! so the rule set is the reasoning AST and mask-risc executes it. Names are
//! labels used only to print.
//!
//! ```text
//! R1 pearl_po(E)    <-> kausal(E)                       (Pearl rung from TEKAMOLO)
//! R2 cert_rel(E)    <-> support2(E), press_pos(E)       (Related or better)
//! R3 candidate(E)    -> cert_rel(E)
//! R4 climbed(E)      -> support2(E)                     (a rung above Staunen needs evidence)
//! R5 direct(E)      <-> main(E)                         (topology from the clause path)
//! ```
//!
//! An equivalence holds when `Any(Or(And(A, Not B), And(B, Not A)))` is false,
//! an implication when `Any(And(A, Not B))` is false. Each side is checked to
//! be non-trivial (some rows, not all) so no rule passes on an empty plane.
//! The falsifier arm corrupts one output bit and expects exactly that rule to
//! fail.
//!
//! ```text
//! cargo run -p lance-graph-quack --example wired_reasoning_probe
//! ```

use causal_edge::CausalMask;
use deepnsm_v2::fsm::{Origin, Pos, PosSet, Reading};
use deepnsm_v2::haspelmath::FrozenOperandAbi;
use deepnsm_v2::tekamolo::{GrammarHeuristics, ReadParams};
use deepnsm_v2::wire::{wire, VerseIn, WiredRow, KAUSAL_LANE};
use lance_graph_contract::epistemic_state5::{Certification3, Epi5Gen, EpistemicState5, Topology2};
use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::{words_for, Foreign, Out, Planes, Value};
use lance_graph_quack::{lower, Agg, Filter, Mask, Query};

const NOUNS: [u16; 6] = [10, 11, 12, 13, 14, 15];
const VERBS: [u16; 3] = [20, 21, 22];
const THE: u16 = 30;
const WHO: u16 = 31;
const WEIL: u16 = 32;

fn one(id: u16, pos: Pos) -> Reading {
    Reading::new(id, PosSet::single(pos))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) % n
    }
}

/// A synthetic corpus: SVO and subject-relative clauses, a `weil` marker on
/// some verses, few objects per (S, P) so statements are both corroborated
/// and contested.
fn corpus(n: usize) -> (Vec<Vec<Reading>>, Vec<Vec<&'static str>>) {
    let mut g = Lcg(0x5EED);
    let (mut rs, mut ss) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let s = NOUNS[g.next(2) as usize];
        let v = VERBS[g.next(2) as usize];
        let o = NOUNS[2 + g.next(2) as usize];
        let mut r = vec![one(THE, Pos::Det), one(s, Pos::Noun)];
        let mut w = vec!["der", "x"];
        if g.next(3) == 0 {
            r.extend([
                one(WHO, Pos::Rel),
                one(v, Pos::Verb),
                one(THE, Pos::Det),
                one(o, Pos::Noun),
            ]);
            w.extend(["der", "v", "den", "y"]);
            let o2 = NOUNS[4 + g.next(2) as usize];
            r.extend([one(v, Pos::Verb), one(THE, Pos::Det), one(o2, Pos::Noun)]);
            w.extend(["v", "den", "z"]);
        } else {
            r.extend([one(v, Pos::Verb), one(THE, Pos::Det), one(o, Pos::Noun)]);
            w.extend(["v", "den", "y"]);
        }
        if g.next(2) == 0 {
            // `. weil <cause clause>`: the cause event the Kausal lane marks.
            let (cs, co) = (NOUNS[g.next(2) as usize], NOUNS[2 + g.next(2) as usize]);
            r.extend([
                Reading::stop(),
                one(WEIL, Pos::Other),
                one(THE, Pos::Det),
                one(cs, Pos::Noun),
                one(VERBS[2], Pos::Verb),
                one(THE, Pos::Det),
                one(co, Pos::Noun),
            ]);
            w.extend([".", "weil", "die", "x", "v", "den", "y"]);
        }
        rs.push(r);
        ss.push(w);
    }
    (rs, ss)
}

fn set(w: &mut [u64], i: usize) {
    w[i / 64] |= 1 << (i % 64);
}

fn get(w: &[u64], i: usize) -> bool {
    w[i / 64] >> (i % 64) & 1 == 1
}

const INPUTS: [&str; 5] = ["kausal", "support2", "press_pos", "main", "climbed"];
const OUTPUTS: [&str; 4] = ["pearl_po", "cert_rel", "candidate", "direct"];

/// Plane index by label: inputs first, then outputs.
fn ix(label: &str) -> usize {
    INPUTS
        .iter()
        .chain(OUTPUTS.iter())
        .position(|x| *x == label)
        .unwrap()
}

fn planes_of(rows: &[WiredRow]) -> Vec<Vec<u64>> {
    let n = rows.len();
    let mut p = vec![vec![0u64; words_for(n)]; INPUTS.len() + OUTPUTS.len()];
    for (i, r) in rows.iter().enumerate() {
        let e = r.edge;
        let st = EpistemicState5::decode(Epi5Gen::V1, e.epistemic_raw5()).unwrap();
        let cert = st.certification();
        let facts = [
            ("kausal", r.tekamolo[KAUSAL_LANE].is_some()),
            ("support2", r.support >= 2),
            ("press_pos", 2 * r.support > r.antecedent),
            ("main", r.origin == Some(Origin::Main)),
            ("climbed", e.novelty_wisdom() > 1),
            ("pearl_po", e.causal_mask() == CausalMask::PO),
            (
                "cert_rel",
                matches!(
                    cert,
                    Certification3::Related | Certification3::CausalCandidate
                ),
            ),
            ("candidate", cert == Certification3::CausalCandidate),
            ("direct", st.topology() == Topology2::Direct),
        ];
        for (label, on) in facts {
            if on {
                set(&mut p[ix(label)], i);
            }
        }
    }
    p
}

fn pl(label: &str) -> Filter {
    Filter::Plane(Mask(ix(label) as u16))
}

fn not(f: Filter) -> Filter {
    Filter::Not(Box::new(f))
}

fn and(a: Filter, b: Filter) -> Filter {
    Filter::And(vec![a, b])
}

/// The violation filter of each rule: rows that break it.
fn rules() -> Vec<(&'static str, Filter)> {
    let iff =
        |a: Filter, b: Filter| Filter::Or(vec![and(a.clone(), not(b.clone())), and(b, not(a))]);
    vec![
        ("R1 pearl_po <-> kausal", iff(pl("pearl_po"), pl("kausal"))),
        (
            "R2 cert_rel <-> support2, press_pos",
            iff(pl("cert_rel"), and(pl("support2"), pl("press_pos"))),
        ),
        (
            "R3 candidate -> cert_rel",
            and(pl("candidate"), not(pl("cert_rel"))),
        ),
        (
            "R4 climbed -> support2",
            and(pl("climbed"), not(pl("support2"))),
        ),
        ("R5 direct <-> main", iff(pl("direct"), pl("main"))),
    ]
}

fn run(filter: &Filter, agg: Agg, planes: &[Vec<u64>], n: usize) -> Value {
    let program = lower(&Query {
        filter: filter.clone(),
        agg,
    })
    .expect("lowers");
    let masks: Vec<&[u64]> = planes.iter().map(Vec::as_slice).collect();
    let p = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let mut s = Scratch::for_program(&program, n).expect("scratch");
    execute_extent(&program, &p, &Foreign::NONE, &mut s, Out::None, 0..n).expect("runs")
}

/// Row oracle: the violation count of each rule, by a plain loop.
fn oracle(planes: &[Vec<u64>], n: usize) -> Vec<usize> {
    let b = |l: &str, i: usize| get(&planes[ix(l)], i);
    let mut v = vec![0; 5];
    for i in 0..n {
        v[0] += usize::from(b("pearl_po", i) != b("kausal", i));
        v[1] += usize::from(b("cert_rel", i) != (b("support2", i) && b("press_pos", i)));
        v[2] += usize::from(b("candidate", i) && !b("cert_rel", i));
        v[3] += usize::from(b("climbed", i) && !b("support2", i));
        v[4] += usize::from(b("direct", i) != b("main", i));
    }
    v
}

fn check(planes: &[Vec<u64>], n: usize) -> Vec<usize> {
    let want = oracle(planes, n);
    let mut got = Vec::new();
    for ((name, f), w) in rules().iter().zip(&want) {
        let Value::Count(c) = run(f, Agg::Count, planes, n) else {
            panic!("{name}: count")
        };
        let any = matches!(run(f, Agg::Any, planes, n), Value::Bool(true));
        assert_eq!(c, *w, "{name}: quack vs row oracle");
        assert_eq!(any, *w > 0, "{name}: Any vs Count");
        got.push(c);
    }
    got
}

fn main() {
    let n_verses = 400;
    let (readings, surfaces) = corpus(n_verses);
    let markers: Vec<(u16, u16)> = (0..n_verses)
        .map(|i| (1 + (i / 50) as u16, 1 + (i % 50) as u16))
        .collect();
    let ins: Vec<VerseIn<'_>> = readings
        .iter()
        .zip(&surfaces)
        .map(|(r, s)| VerseIn {
            readings: r.clone(),
            surface: s.clone(),
            forms: vec![],
            qualia: vec![],
        })
        .collect();
    let mut ops: Vec<u16> = NOUNS.to_vec();
    ops.extend(VERBS);
    ops.extend([THE, WHO, WEIL]);
    let abi = FrozenOperandAbi::new(1, &ops).unwrap();
    let h = GrammarHeuristics::parse("# lane\tlemma\trelation\tcount\nKausal\tweil\tmark\t273\n");
    let w = wire(&markers, &ins, 1, &abi, Some(&h), ReadParams::RIGHT_CORNER).unwrap();
    let (rows, rep) = (w.rows, w.report);
    let n = rows.len();
    println!("rows {n}  report {rep:?}");

    let planes = planes_of(&rows);
    for (k, label) in INPUTS.iter().chain(OUTPUTS.iter()).enumerate() {
        let c = (0..n).filter(|&i| get(&planes[k], i)).count();
        println!("  {label:10} {c:5} / {n}");
        assert!(
            c > 0 && c < n,
            "{label}: plane must be non-trivial (got {c})"
        );
    }

    let v = check(&planes, n);
    for ((name, _), c) in rules().iter().zip(&v) {
        println!("{name:40} violations {c}");
    }
    assert!(
        v.iter().all(|&c| c == 0),
        "every rule holds on the wired rows"
    );

    // Falsifier: flip one output bit per rule and expect exactly that rule to
    // fail (R1..R5 read pearl_po, cert_rel, candidate, climbed, direct).
    let flips = ["pearl_po", "cert_rel", "candidate", "climbed", "direct"];
    for (r, label) in flips.iter().enumerate() {
        let mut bad = planes.clone();
        let p = &mut bad[ix(label)];
        // A row where flipping must violate: for implications, a row where
        // the consequent is false; otherwise any row.
        let target = match *label {
            "candidate" => (0..n).find(|&i| !get(&planes[ix("cert_rel")], i)),
            "climbed" => (0..n).find(|&i| !get(&planes[ix("support2")], i)),
            _ => Some(0),
        }
        .expect("a row to flip");
        p[target / 64] ^= 1 << (target % 64);
        let v = check(&bad, n);
        assert!(v[r] > 0, "flipping {label} must break rule {}", r + 1);
        println!("flip {label:10} -> rule {} violations {}", r + 1, v[r]);
    }
    println!("ok");
}
