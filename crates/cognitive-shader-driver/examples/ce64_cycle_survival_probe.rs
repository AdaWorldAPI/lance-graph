//! D-CE64-TIME-0: `CausalEdge64` as the register that survives a cycle.
//!
//! Claim under test:
//!
//! ```text
//! CE64_k ──folds of cycle k──▶ CE64_{k+1} ──▶ read by cycle k+1
//! ```
//!
//! A cycle's fold population is transient; only what the existing revision
//! writes into the register survives, and the next cycle measures its
//! affordances from that register alone.
//!
//! # Existing machinery used (unchanged)
//!
//! - **The surviving store:** one declared (`set_populated`) row of
//!   `MailboxSoA`'s `edges` column, written through the cycle-gated
//!   `write_row(row, cycle, cell)`, carried across `tick()`, and read back
//!   through the production `MailboxSoaView` lens (`n_rows` + `edges_raw`).
//!   Nothing else crosses a cycle boundary.
//! - **The fold:** `CausalEdge64::learn(observation, t)`, the shipped in-place
//!   NARS revision: `f`/`c` merge by evidence weight, plasticity follows
//!   confidence. Under the v2 layout it never touches bits 59..63.
//! - **The fold population:** `causal_audit::SupportLedger`, built fresh in
//!   each cycle and dropped at its end; `distinct_sources_for` counts sources,
//!   not receipts.
//! - **The next-cycle reading:** the D-GSO-AFF-0 recipe law (#1370), shared
//!   unchanged through `shared/affordance_law.rs`:
//!   `EligibleRecipes = STATE[raw5] & PEARL[pearl3]`.
//!
//! # The one probe-declared piece: the EpistemicState5 transition
//!
//! No production code writes bits 59..63 from evidence: `learn` moves only
//! F/C and plasticity, and the #1370 law reads only bits 59..63 and Pearl, so
//! F/C alone can never change eligibility. The missing step is declared here,
//! as small as possible, over the AFF-0 codebook:
//!
//! | from | to | when (after the cycle's folds) |
//! |---|---|---|
//! | 0 `Direct × Open` | 4 `Direct × Associated` | ≥ 2 distinct `DirectlyObserved` sources this cycle, `c ≥ 200`, `f ≥ 200` |
//! | 4 | 0 | `f < 180` (contradiction revised the carried evidence down) |
//!
//! Accumulation and certification stay separate: any number of folds from
//! one source moves F/C but never the code. Thresholds are policy pins.
//!
//! # What it does NOT prove
//!
//! - That this transition belongs in production; there is no production
//!   writer of bits 59..63 from evidence. That gap is the finding.
//! - Cross-cycle source identity: the ledger dies with its cycle, so two
//!   different sources seen in two different cycles never count as two. The
//!   register cannot carry a source set; a test pins this as missing state.
//! - Scale: the population here is a handful of folds. `learn` is a
//!   fixed-size, O(1) update per fold, so a larger population changes the
//!   number of updates, not what crosses the boundary (one 64-bit word).
//!
//! Run: `cargo run -p cognitive-shader-driver --example ce64_cycle_survival_probe`
//! Tests: `cargo test -p cognitive-shader-driver --example ce64_cycle_survival_probe`

#[path = "shared/affordance_law.rs"]
mod affordance_law;

use affordance_law::{measure, raw5, LawGen};
use causal_edge::edge::CausalEdge64;
use causal_edge::pearl::CausalMask;
use causal_edge::PlasticityState;
use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome};
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::causal_audit::{
    EvidenceSourceId, SupportBasis, SupportLedger, SupportReceipt,
};
use lance_graph_contract::scheduler::DatasetVersion;
use lance_graph_contract::soa_view::MailboxSoaView;

/// The row of the mailbox that holds the register.
const ROW: usize = 0;

/// EpistemicState5 codes of the AFF-0 law used here.
/// `Direct × Open` (was code 3 `Direct × Observed` under the #1370
/// probe-local codebook; observation is evidence, not a coordinate).
const CODE_OPEN: u8 = 0;
/// `Direct × Associated` (was code 7).
const CODE_ASSOCIATED: u8 = 4;

/// Policy pins of the probe-declared transition.
const C_MIN: u8 = 200;
const F_HI: u8 = 200;
const F_LO: u8 = 180;

/// One local contribution: a source reports `(f, c)` about the tracked
/// relation. `strength` and `at` are carried on the receipt but read by
/// nothing below, so they serve as irrelevant inputs.
#[derive(Debug, Clone, Copy)]
struct Fold {
    source: u64,
    f: u8,
    c: u8,
    strength: u8,
    at: u64,
}

const fn fold(source: u64, f: u8) -> Fold {
    Fold {
        source,
        f,
        c: 128,
        strength: 1,
        at: 1,
    }
}

/// Write a declared V1 EpistemicState5 code (bits 59..63, jointly).
fn with_code(edge: CausalEdge64, code: u8) -> CausalEdge64 {
    affordance_law::write_code(edge, code)
}

/// The tracked relation as an observation carrying `(f, c)`. S/P/O match the
/// register, so `learn` never re-targets it.
fn observation(f: u8, c: u8) -> CausalEdge64 {
    CausalEdge64::pack_v2(1, 2, 3, f, c, CausalMask::SO, 0, PlasticityState::ALL_HOT)
}

/// The register before any evidence: code `Direct × Open`, no confidence.
fn initial() -> CausalEdge64 {
    with_code(
        CausalEdge64::pack_v2(1, 2, 3, 128, 0, CausalMask::SO, 0, PlasticityState::ALL_HOT),
        CODE_OPEN,
    )
}

/// One cycle: fold the population into the register, then settle the code.
/// Takes the surviving register and this cycle's folds; nothing else.
fn cycle(register: CausalEdge64, folds: &[Fold]) -> CausalEdge64 {
    let mut reg = register;
    let mut ledger = SupportLedger::new();
    for x in folds {
        reg.learn(observation(x.f, x.c), 0);
        ledger.record(SupportReceipt {
            basis: SupportBasis::DirectlyObserved,
            source: EvidenceSourceId(x.source),
            at: DatasetVersion(x.at),
            strength: x.strength,
        });
    }
    let sources = ledger.distinct_sources_for(SupportBasis::DirectlyObserved);
    let code = match raw5(reg) {
        CODE_OPEN if sources >= 2 && reg.confidence_u8() >= C_MIN && reg.frequency_u8() >= F_HI => {
            CODE_ASSOCIATED
        }
        CODE_ASSOCIATED if reg.frequency_u8() < F_LO => CODE_OPEN,
        other => other,
    };
    with_code(reg, code)
    // `ledger` is dropped here: the population does not survive the cycle.
}

/// What a cycle can legally do, measured from the register alone.
fn eligible(register: CausalEdge64) -> u64 {
    measure(LawGen::V1, register, EdgeProvenance::V2Stamped).expect("declared code")
}

/// The live register, read through the production `MailboxSoaView` lens.
/// `None` when the row is not a declared (populated) row of the mailbox, so
/// an undeclared row can never pass for surviving state.
fn read_register(view: &dyn MailboxSoaView) -> Option<CausalEdge64> {
    (ROW < view.n_rows()).then(|| CausalEdge64(view.edges_raw()[ROW]))
}

/// A mailbox with the register row declared and seeded with `seed`.
fn mailbox_with(id: u32, seed: CausalEdge64) -> MailboxSoA<4> {
    let mut mb: MailboxSoA<4> = MailboxSoA::new(id, 0, 0.5);
    mb.set_populated(ROW + 1);
    let cell = WriteCell {
        edge: Some(seed),
        ..WriteCell::default()
    };
    assert_eq!(mb.write_row(ROW, mb.cycle(), &cell), WriteOutcome::Accepted);
    mb
}

/// Run a schedule of cycles through the mailbox row. With `persist == false`
/// the cycle's result is not written back (F1).
fn run(schedule: &[&[Fold]], persist: bool) -> (Vec<CausalEdge64>, Vec<u64>) {
    let mut mb = mailbox_with(1, initial());
    let mut registers = Vec::new();
    let mut eligibility = Vec::new();
    for folds in schedule {
        let now = read_register(&mb).expect("declared register row");
        registers.push(now);
        eligibility.push(eligible(now));
        let next = cycle(now, folds);
        if persist {
            let cell = WriteCell {
                edge: Some(next),
                ..WriteCell::default()
            };
            assert_eq!(mb.write_row(ROW, mb.cycle(), &cell), WriteOutcome::Accepted);
        }
        mb.tick();
    }
    let last = read_register(&mb).expect("declared register row");
    registers.push(last);
    eligibility.push(eligible(last));
    (registers, eligibility)
}

/// Cycle 0: one source, three times. Cycle 1: two new sources agree.
/// Cycle 2: two sources contradict.
const K0: &[Fold] = &[fold(1, 230), fold(1, 230), fold(1, 230)];
const K1: &[Fold] = &[fold(2, 230), fold(3, 230)];
const K2: &[Fold] = &[fold(4, 25), fold(5, 25)];
const SCHEDULE: [&[Fold]; 3] = [K0, K1, K2];

fn names(e: u64) -> String {
    let n = [
        "HYDRATE",
        "MECHANISM",
        "STRATIFY",
        "ROBUSTNESS",
        "CAUSAL_ID",
        "OBSERVE",
        "COUNTERFACTUAL",
    ];
    (0..7)
        .filter(|i| e & (1 << i) != 0)
        .map(|i| n[i])
        .collect::<Vec<_>>()
        .join("|")
}

fn main() {
    let (regs, elig) = run(&SCHEDULE, true);
    println!("D-CE64-TIME-0: the register survives the cycle");
    for (k, (r, e)) in regs.iter().zip(&elig).enumerate() {
        println!(
            "  now_{k}: f={:>3} c={:>3} code={:>2}  eligible={}",
            r.frequency_u8(),
            r.confidence_u8(),
            raw5(*r),
            names(*e)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::affordance_law::{OBSERVE_FOLD, STRATIFY};
    use super::*;

    const OBS: u64 = 1 << OBSERVE_FOLD;
    const STRAT: u64 = 1 << STRATIFY;

    /// The trajectory: accumulation in cycle 0 changes nothing legal, the
    /// certification in cycle 1 is read by cycle 2, and the contradiction in
    /// cycle 2 is read by cycle 3.
    #[test]
    fn the_register_carries_the_result_across_three_boundaries() {
        let (regs, elig) = run(&SCHEDULE, true);
        assert_eq!(elig, vec![OBS, OBS, OBS | STRAT, OBS]);
        // Cycle 0 moved F/C without moving the code.
        assert!(regs[1].confidence_u8() > regs[0].confidence_u8());
        assert_eq!(raw5(regs[1]), CODE_OPEN);
        // Cycle 1 certified association; cycle 2 revised it away.
        assert_eq!(raw5(regs[2]), CODE_ASSOCIATED);
        assert_eq!(raw5(regs[3]), CODE_OPEN);
        assert!(regs[3].frequency_u8() < regs[2].frequency_u8());
    }

    /// F1: without the write-back each cycle starts from the seed, and the
    /// accumulated result never reaches a later cycle.
    #[test]
    fn f1_no_survival_no_memory() {
        let (_, kept) = run(&SCHEDULE, true);
        let (regs, lost) = run(&SCHEDULE, false);
        assert!(
            regs.iter().all(|r| *r == initial()),
            "every cycle saw the seed"
        );
        assert_eq!(lost, vec![OBS; 4]);
        assert_ne!(lost, kept);
    }

    /// F2: the evidence carried out of cycle 0 is load-bearing for cycle 1.
    /// Without it, cycle 1's two sources do not reach the confidence bar.
    #[test]
    fn f2_a_carried_contribution_matters() {
        let (_, with) = run(&SCHEDULE, true);
        let (_, without) = run(&[&[], K1, K2], true);
        assert_eq!(with[2], OBS | STRAT);
        assert_eq!(without[2], OBS, "cycle 2 never gains STRATIFY");
        // And one source standing in for another breaks the certification.
        let (_, same_source) = run(&[K0, &[fold(2, 230), fold(2, 230)], K2], true);
        assert_eq!(same_source[2], OBS);
    }

    /// F3: inputs the transition does not read leave the surviving register
    /// bit-identical.
    #[test]
    fn f3_irrelevant_contributions_stay_silent() {
        let (base, _) = run(&SCHEDULE, true);
        let noisy = |fs: &[Fold]| -> Vec<Fold> {
            fs.iter()
                .enumerate()
                .map(|(i, x)| Fold {
                    strength: 200 - i as u8,
                    at: 99 + i as u64,
                    ..*x
                })
                .collect()
        };
        let (a, b, c) = (noisy(K0), noisy(K1), noisy(K2));
        let (other, _) = run(&[&a, &b, &c], true);
        assert_eq!(base, other);
    }

    /// F4: a change in cycle k alters what cycle k+1 may legally do.
    #[test]
    fn f4_the_next_cycle_reads_the_change() {
        let (_, elig) = run(&SCHEDULE, true);
        assert_ne!(elig[1], elig[2], "cycle 1's certification reaches cycle 2");
        assert_ne!(elig[2], elig[3], "cycle 2's contradiction reaches cycle 3");
    }

    /// F5: the same seed, folds, order and law replay to the same registers.
    #[test]
    fn f5_replay_is_deterministic() {
        assert_eq!(run(&SCHEDULE, true), run(&SCHEDULE, true));
    }

    /// F6 and F7: a fresh mailbox seeded with nothing but the surviving
    /// register continues exactly like the full run. No earlier population,
    /// event log or second object is needed.
    #[test]
    fn f6_f7_the_register_alone_continues_the_run() {
        let (regs, _) = run(&SCHEDULE, true);
        let mut fresh = mailbox_with(9, regs[2]);
        fresh.tick();
        let now = read_register(&fresh).expect("declared register row");
        assert_eq!(cycle(now, K2), regs[3]);
    }

    /// The register is only readable as a declared mailbox row: the same
    /// bytes in an undeclared row are not surviving state.
    #[test]
    fn an_undeclared_row_is_not_a_register() {
        let mut mb: MailboxSoA<4> = MailboxSoA::new(3, 0, 0.5);
        let cell = WriteCell {
            edge: Some(initial()),
            ..WriteCell::default()
        };
        assert_eq!(mb.write_row(ROW, mb.cycle(), &cell), WriteOutcome::Accepted);
        assert_eq!(read_register(&mb), None, "written but not declared");
        mb.set_populated(ROW + 1);
        assert_eq!(read_register(&mb), Some(initial()));
    }

    /// Accumulation is not certification: a thousand folds from one source
    /// saturate the evidence and leave the code where it was.
    #[test]
    fn repetition_from_one_source_never_certifies() {
        let many: Vec<Fold> = (0..1000).map(|_| fold(1, 230)).collect();
        let reg = cycle(initial(), &many);
        assert!(reg.confidence_u8() >= C_MIN && reg.frequency_u8() >= F_HI);
        assert_eq!(raw5(reg), CODE_OPEN);
        assert_eq!(eligible(reg), OBS);
    }

    /// Missing state, pinned: two different sources in two different cycles
    /// never count as two, because the register carries no source set.
    #[test]
    fn sources_split_across_cycles_do_not_combine() {
        let (_, elig) = run(&[K0, &[fold(2, 230)], &[fold(3, 230)]], true);
        assert!(elig.iter().all(|&e| e == OBS), "{elig:?}");
    }
}
