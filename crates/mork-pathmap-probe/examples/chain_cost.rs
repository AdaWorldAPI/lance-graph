//! Cost of the D-RPR-1 handoff, transition by transition.
//!
//! `cargo run --release --example chain_cost`
//!
//! Three arms answer the same three-question chain (predicate 0 for 3 hops
//! from one seed, then predicate 1 for 1 hop from that answer, then predicate
//! 2 for 1 hop):
//!
//! - **direct**: every question walks the sorted run from scratch, one seek
//!   per visited identity and hop; no frame;
//! - **full**: a frame whose focus changes are rebuilt with `Csr::prepare` +
//!   `Tile::from_csr`;
//! - **incremental**: the same frame, focus changes applied with `rebind`.
//!
//! Every answer is checked against a full-scan oracle. Times are medians of 5
//! runs of the whole chain, in µs. Lab measurement only.

use std::collections::BTreeSet;
use std::time::Instant;

use mork_pathmap_probe::chain::*;
use mork_pathmap_probe::rmap::*;

const PREDS: [u32; 3] = [0, 1, 2];
const QUESTIONS: [(u32, usize); 3] = [(0, 3), (1, 1), (2, 1)];
const SEED: u32 = 17;
const FAR: u32 = 40_000;
const REPS: usize = 5;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn oracle(facts: &[Fact], p: u32, seeds: &BTreeSet<u32>, hops: usize) -> BTreeSet<u32> {
    let mut reached = seeds.clone();
    let mut frontier = seeds.clone();
    for _ in 0..hops {
        let next: BTreeSet<u32> = facts
            .iter()
            .filter(|f| frontier.contains(&f[0]) && f[1] == p)
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

/// The direct arm: breadth-first over the sorted run, counting seeks.
fn direct(
    run: &SortedRun<'_>,
    p: u32,
    seeds: &BTreeSet<u32>,
    hops: usize,
) -> (BTreeSet<u32>, usize) {
    let mut reached = seeds.clone();
    let mut frontier: Vec<u32> = seeds.iter().copied().collect();
    let mut seeks = 0;
    for _ in 0..hops {
        let mut next = Vec::new();
        for &s in &frontier {
            seeks += 1;
            run.neighbours(p, s, |o, _| {
                if reached.insert(o) {
                    next.push(o);
                }
            });
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    (reached, seeks)
}

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

fn cur() -> Witness {
    Witness {
        facts: 1,
        seal: 1,
        focus: 0,
        scope: 7,
    }
}

/// One transition of a framed arm: an admission (lookup + translation + tile
/// rebuild) or a fold.
#[derive(Clone, Debug)]
enum Event {
    Admit {
        step: usize,
        demanded: usize,
        changed: usize,
        seeks: usize,
        full_seeks: usize,
        us: f64,
        tile_us: f64,
    },
    Fold {
        step: usize,
        escapes: usize,
        us: f64,
    },
    /// The whole chain asked again on the final frame: folds only, 0 seeks.
    Reask { us: f64 },
}

fn event_us(e: &Event) -> f64 {
    match e {
        Event::Admit { us, .. } | Event::Fold { us, .. } | Event::Reask { us } => *us,
    }
}

/// The chain, inlined from `chain::ask` so each transition can be timed.
fn framed(facts: &[Fact], how: Rebuild) -> (Vec<BTreeSet<u32>>, Vec<Event>, usize, f64) {
    let run = SortedRun::build(facts, false, 1);
    let t = Instant::now();
    let mut frame = Frame::prepare(&run, first_focus(&run), &PREDS, &cur());
    let prep_us = t.elapsed().as_secs_f64() * 1e6;
    let prep_seeks = frame.prepare_seeks;
    let mut exec = TileExec::new();
    let mut keep = BTreeSet::new();
    let mut seeds = BTreeSet::from([SEED]);
    let mut answers = Vec::new();
    let mut ev = Vec::new();
    for (step, (p, hops)) in QUESTIONS.into_iter().enumerate() {
        keep.extend(seeds.iter().copied());
        let mut pending: BTreeSet<u32> = seeds.clone();
        loop {
            let t = Instant::now();
            let adm = frame.admit(&run, &cur(), &keep, &pending, how).unwrap();
            let us = t.elapsed().as_secs_f64() * 1e6;
            if let Some(a) = adm {
                let t = Instant::now();
                let _ = Tile::from_csr(&frame.csr);
                let tile_us = t.elapsed().as_secs_f64() * 1e6;
                ev.push(Event::Admit {
                    step,
                    demanded: a.demanded.len(),
                    changed: a.changed,
                    seeks: a.seeks,
                    full_seeks: a.full_seeks,
                    us,
                    tile_us,
                });
            }
            let w = Witness {
                focus: frame.focus.generation,
                ..cur()
            };
            let ss = frame.slots(&seeds).unwrap();
            let t = Instant::now();
            let r = exec
                .reach(&frame.tile, &frame.csr, &w, None, None, &[p], &ss, hops)
                .unwrap();
            let us = t.elapsed().as_secs_f64() * 1e6;
            ev.push(Event::Fold {
                step,
                escapes: r.escaped.len(),
                us,
            });
            let reached: BTreeSet<u32> = slots_of(&r.reached)
                .into_iter()
                .map(|s| frame.focus.id(s))
                .collect();
            keep.extend(reached.iter().copied());
            if r.global().is_ok() {
                answers.push(reached.clone());
                seeds = reached;
                break;
            }
            pending = r.escaped.clone();
        }
    }
    // Ask the whole chain again on the retained frame: no admission may happen.
    let mut reask = 0.0;
    let mut seeds = BTreeSet::from([SEED]);
    for (i, (p, hops)) in QUESTIONS.into_iter().enumerate() {
        let w = Witness {
            focus: frame.focus.generation,
            ..cur()
        };
        let ss = frame.slots(&seeds).expect("re-ask seeds are focused");
        let t = Instant::now();
        let r = exec
            .reach(&frame.tile, &frame.csr, &w, None, None, &[p], &ss, hops)
            .unwrap();
        reask += t.elapsed().as_secs_f64() * 1e6;
        assert!(r.global().is_ok(), "re-ask of step {i} escaped");
        let got: BTreeSet<u32> = slots_of(&r.reached)
            .into_iter()
            .map(|s| frame.focus.id(s))
            .collect();
        assert_eq!(got, answers[i], "re-ask of step {i}");
        seeds = got;
    }
    ev.push(Event::Reask { us: reask });
    (answers, ev, prep_seeks, prep_us)
}

fn main() {
    let tier = if cfg!(target_feature = "avx512f") {
        "avx512f"
    } else if cfg!(target_feature = "avx2") {
        "avx2"
    } else {
        "scalar"
    };
    let facts = graph(65_536, 3, 2, 512, 0.9, 0x5EED);
    println!(
        "tier: {tier}; facts: {}; questions: {QUESTIONS:?}; reps: {REPS}",
        facts.len()
    );

    // Oracle and direct arm.
    let run = SortedRun::build(&facts, false, 1);
    let mut want = Vec::new();
    let mut seeds = BTreeSet::from([SEED]);
    let mut direct_seeks = Vec::new();
    let mut direct_us = vec![Vec::new(); QUESTIONS.len()];
    for (i, (p, hops)) in QUESTIONS.into_iter().enumerate() {
        let w = oracle(&facts, p, &seeds, hops);
        let (got, s) = direct(&run, p, &seeds, hops);
        assert_eq!(got, w, "direct, step {i}");
        for _ in 0..REPS {
            let t = Instant::now();
            let _ = direct(&run, p, &seeds, hops);
            direct_us[i].push(t.elapsed().as_secs_f64() * 1e6);
        }
        direct_seeks.push(s);
        seeds = w.clone();
        want.push(w);
    }

    let mut totals = Vec::new();
    for how in [Rebuild::Incremental, Rebuild::Full] {
        let mut runs = Vec::new();
        for _ in 0..REPS {
            runs.push(framed(&facts, how));
        }
        let (answers, events, prep_seeks, _) = &runs[0];
        assert_eq!(answers, &want, "{how:?} answers differ from the oracle");
        let prep_us = median(runs.iter().map(|r| r.3).collect());
        println!(
            "\n== {how:?}: initial prepare {prep_us:.1} us, {prep_seeks} seeks (256 slots x 3 preds)"
        );
        for (k, e) in events.iter().enumerate() {
            let us_med = median(runs.iter().map(|r| event_us(&r.1[k])).collect());
            match e {
                Event::Admit {
                    step,
                    demanded,
                    changed,
                    seeks,
                    full_seeks,
                    ..
                } => {
                    let tile = median(
                        runs.iter()
                            .map(|r| match &r.1[k] {
                                Event::Admit { tile_us, .. } => *tile_us,
                                _ => 0.0,
                            })
                            .collect(),
                    );
                    println!(
                        "  step {step} admit: demanded {demanded}, slots changed {changed}, seeks {seeks} (full would seek {full_seeks}), {us_med:.1} us incl. tile rebuild ~{tile:.1} us"
                    );
                }
                Event::Fold { step, escapes, .. } => {
                    println!("  step {step} fold : {us_med:.1} us, escapes {escapes}");
                }
                Event::Reask { .. } => {
                    println!(
                        "  re-ask of all {} steps on the final frame: {us_med:.1} us, 0 seeks, answers equal",
                        QUESTIONS.len()
                    );
                }
            }
        }
        let seeks: usize = events
            .iter()
            .map(|e| match e {
                Event::Admit { seeks, .. } => *seeks,
                _ => 0,
            })
            .sum();
        let full: usize = events
            .iter()
            .map(|e| match e {
                Event::Admit { full_seeks, .. } => *full_seeks,
                _ => 0,
            })
            .sum();
        println!(
            "  chain seeks after the initial prepare: {seeks} (a full rebuild per admission would seek {full}; avoided {})",
            full - seeks
        );
        let n = events.len();
        let first: f64 = prep_us
            + (0..n - 1)
                .map(|k| median(runs.iter().map(|r| event_us(&r.1[k])).collect()))
                .sum::<f64>();
        let reask = median(runs.iter().map(|r| event_us(&r.1[n - 1])).collect());
        totals.push((how, first, reask));
    }

    println!("\n== direct (no frame, every question from scratch)");
    for (i, s) in direct_seeks.iter().enumerate() {
        println!(
            "  step {i}: {s} seeks, {:.1} us, answer {} ids",
            median(direct_us[i].clone()),
            want[i].len()
        );
    }
    println!(
        "  direct seeks total: {}",
        direct_seeks.iter().sum::<usize>()
    );

    // Arithmetic, not a measurement: how many re-asks of the chain repay the
    // first framed pass against asking directly every time.
    let direct_pass: f64 = direct_us.iter().map(|v| median(v.clone())).sum();
    println!("\n== whole chain, first pass vs re-ask (us)");
    println!("  direct: {direct_pass:.1} every pass");
    for (how, first, reask) in totals {
        let saving = direct_pass - reask;
        let note = if saving > 0.0 {
            format!("repaid after {:.0} re-asks (arithmetic)", first / saving)
        } else {
            "a re-ask is not cheaper than asking directly: never repaid".to_string()
        };
        println!("  {how:?}: first pass {first:.1}, re-ask {reask:.1}; {note}");
    }
}
