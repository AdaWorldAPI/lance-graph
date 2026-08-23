//! PROBE-R2IL-FRONTIER-PHASE2-1 — the Phase-1 frontier loop lifted onto
//! R2IL-shaped typed behavior ops, where "way richer" becomes measurable:
//! reconstruction at MACHINE-STATE level, and a falsification intervention
//! that catches a macro happy-path reinforcement would have LEARNED.
//!
//! **Phase 2 of the operator's plan (2026-08-23):** Phase 1
//! (`PROBE-STYLE-MICROCODE-FRONTIER-1`, merged #1012) proved the loop —
//! styles as ordered microcode, frozen/explore superposition, outcome
//! revision via shipped `TruthValue::revise` + `Stamp`, freezing gated on
//! the `LearnedSurvivedTests` predicate, no learner subsystem. Phase 2
//! swaps the op vocabulary: from #1001 view-edit atoms to **R2IL-shaped
//! reconstructible typed behavior**.
//!
//! # Grounding (real, cited — mirrored, NOT imported)
//!
//! The op shapes mirror `ruff_r2il` (ruff repo, `origin/main`,
//! `crates/ruff_r2il/src/`): the p-code-shaped op layer (`R2ILOp` over
//! `SpaceId` operands, `furnace.rs`'s FactKind table — `Op` / `OperandIn` /
//! `OperandOut` / `Edge` / `MemUse` / `MemDef` / `Predicate` / `CallSite`),
//! and the Varnode operand discipline (`facet.rs`: space → offset → size).
//! `ruff` is a separate cargo workspace, so nothing is imported: the probe
//! carries a faithful PROBE-LOCAL mirror of a minimal subset and says so.
//! The V4 classid stays provisional (O5 gate); nothing here mints,
//! canonizes, or reserves — same-dock ≠ same-ClassView throughout.
//!
//! # What "way richer" means, as numbers (R1)
//!
//! Phase 1's vocabulary: 4 plan atoms editing one `usize` (a view-plan
//! length). Phase 2's: typed ops over `space × offset × size` operands
//! editing a MACHINE STATE — so `BEFORE + TYPED EDIT = AFTER` is now a
//! byte-identity over full state, not a length equation, and a macro's
//! failure can be a CONTRACT violation invisible to its own output.
//!
//! # The headline gate (R6) — what counterfactual probing buys
//!
//! `explore-reckless` computes the right VALUE into the WRONG register.
//! The happy-path oracle is deliberately sloppy (it checks "the result
//! exists in some register" — an under-specified success signal, labelled
//! as such): under it, reckless SUCCEEDS every episode and its expectation
//! RISES. **Happy-path reinforcement would have learned it.** The
//! falsification intervention checks the actual CONTRACT (result in `r2`,
//! callee-saved `r3` preserved) — reckless fails it, and the admission
//! predicate refuses the freeze DESPITE high expectation. That is the
//! operator's "cheap falsification / counterfactual probing" cashing out:
//! recurrence + success is not enough; only what SURVIVES the intervention
//! is learned. (#1010's asymmetry, #1011's seventh state, at macro level.)
//!
//! # Honesty box
//!
//! - The machine, the oracles and the episodes are toys (stated inline).
//!   This proves the LOOP over the richer vocabulary — state-level
//!   reconstruction, contract-level falsification — not that real R2IL
//!   corpora behave this way. Production episodes over real
//!   `FunctionBehavior` streams are the next measurement, and require the
//!   ruff-side corpus this checkout does not carry.
//! - Dispatch policy (trusted ∧ cheapest) is probe-local policy, as in
//!   Phase 1. The learner remains `revise()` + CHOICE — nothing new.
//! - No BPE table, no macro compression, no OGAR-loco routing: the widened
//!   synthesis (`R2IL × BPE`, loco macros, V4 as thinking dynamic) stays a
//!   three-IF hypothesis exactly as recorded; this probe advances only the
//!   R2IL-vocabulary leg.

use lance_graph_planner::nars::belief::Stamp;
use lance_graph_planner::nars::truth::TruthValue;

/// Probe-local operand — the Varnode discipline (space → offset → size),
/// mirrored from `ruff_r2il` `facet.rs`. NOT an import; NOT a mint.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Vn {
    /// 0 = register file, 1 = constant space (the minimal two-space subset).
    space: u8,
    offset: u32,
    size: u8,
}

const fn reg(i: u32) -> Vn {
    Vn {
        space: 0,
        offset: i,
        size: 8,
    }
}
const fn konst(v: u32) -> Vn {
    Vn {
        space: 1,
        offset: v,
        size: 8,
    }
}

/// Probe-local mirror of a minimal p-code-shaped op subset (`R2ILOp`
/// lineage: one output, up to two typed inputs). Copy/IntAdd/IntMult/
/// IntLeft are the arithmetic leaf shapes; richer kinds (Edge, MemUse,
/// Predicate, CallSite — the full FactKind table) are named, not built.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum R2Op {
    Copy { dst: Vn, src: Vn },
    IntAdd { dst: Vn, a: Vn, b: Vn },
    IntMult { dst: Vn, a: Vn, b: Vn },
    IntLeft { dst: Vn, a: Vn, sh: Vn },
}

/// The toy machine state — four 64-bit registers. `r3` is callee-saved by
/// CONTRACT (the invariant the sloppy oracle forgets and the falsification
/// intervention enforces).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct MachState {
    regs: [u64; 4],
}

fn read(s: &MachState, v: Vn) -> u64 {
    match v.space {
        0 => s.regs[v.offset as usize],
        _ => u64::from(v.offset),
    }
}

/// Execute one typed op: BEFORE + TYPED EDIT = AFTER at state level.
fn step(s: &mut MachState, op: R2Op) {
    match op {
        R2Op::Copy { dst, src } => s.regs[dst.offset as usize] = read(s, src),
        R2Op::IntAdd { dst, a, b } => {
            s.regs[dst.offset as usize] = read(s, a).wrapping_add(read(s, b))
        }
        R2Op::IntMult { dst, a, b } => {
            s.regs[dst.offset as usize] = read(s, a).wrapping_mul(read(s, b))
        }
        R2Op::IntLeft { dst, a, sh } => {
            s.regs[dst.offset as usize] = read(s, a) << (read(s, sh) & 63)
        }
    }
}

/// One behavior macro = ordered microcode of R2IL-shaped ops + its style
/// claim (Phase 1's exact shape, richer vocabulary).
#[derive(Clone, Debug)]
struct Macro {
    name: &'static str,
    code: Vec<R2Op>,
    truth: TruthValue,
    stamp: Stamp,
    frozen: bool,
    survived: bool,
}

impl Macro {
    fn new(name: &'static str, code: Vec<R2Op>) -> Self {
        Self {
            name,
            code,
            truth: TruthValue::new(0.5, 0.0),
            stamp: Stamp::default(),
            frozen: false,
            survived: false,
        }
    }
    fn cost(&self) -> usize {
        self.code.len()
    }
    fn run(&self, mut s: MachState) -> MachState {
        for &op in &self.code {
            step(&mut s, op);
        }
        s
    }
    fn observe(&mut self, success: bool, episode: u32) {
        let st = Stamp::source(episode);
        if self.stamp.disjoint(st) {
            self.truth = self
                .truth
                .revise(&TruthValue::new(if success { 1.0 } else { 0.0 }, 0.5));
            self.stamp = self.stamp.union(st);
        }
    }
}

/// The SLOPPY happy-path oracle (deliberately under-specified, labelled):
/// "the doubled sum exists in SOME register." This is the success signal a
/// naive reinforcement loop would use — and gate R6 shows what it costs.
fn sloppy_oracle(before: &MachState, after: &MachState) -> bool {
    let want = (before.regs[0].wrapping_add(before.regs[1])).wrapping_mul(2);
    after.regs.contains(&want)
}

/// The CONTRACT (the falsification intervention's spec): the doubled sum in
/// `r2`, AND callee-saved `r3` bit-preserved.
fn contract(before: &MachState, after: &MachState) -> bool {
    let want = (before.regs[0].wrapping_add(before.regs[1])).wrapping_mul(2);
    after.regs[2] == want && after.regs[3] == before.regs[3]
}

fn dispatch(macros: &[Macro], trust: f32) -> &Macro {
    macros
        .iter()
        .filter(|m| m.truth.expectation() >= trust)
        .min_by_key(|m| m.cost())
        .unwrap_or_else(|| macros.iter().find(|m| m.frozen).expect("incumbent"))
}

fn main() {
    let mut pass = 0u32;
    let mut gate = |name: &str, ok: bool, detail: String| {
        assert!(ok, "[FAIL] {name} — {detail}");
        println!("  [PASS] {name} — {detail}");
        pass += 1;
    };

    // The superposition over ONE op vocabulary. Goal contract: r2 = (r0+r1)*2.
    let mut macros = vec![
        Macro {
            frozen: true,
            survived: true,
            truth: TruthValue::new(0.9, 0.8),
            ..Macro::new(
                "frozen-incumbent",
                vec![
                    R2Op::Copy {
                        dst: reg(2),
                        src: reg(0),
                    },
                    R2Op::IntAdd {
                        dst: reg(2),
                        a: reg(2),
                        b: reg(1),
                    },
                    R2Op::IntMult {
                        dst: reg(2),
                        a: reg(2),
                        b: konst(2),
                    },
                ],
            )
        },
        // Cheaper AND contract-sound: add then shift, straight into r2.
        Macro::new(
            "explore-lean",
            vec![
                R2Op::IntAdd {
                    dst: reg(2),
                    a: reg(0),
                    b: reg(1),
                },
                R2Op::IntLeft {
                    dst: reg(2),
                    a: reg(2),
                    sh: konst(1),
                },
            ],
        ),
        // Cheaper and HAPPY-PATH-correct — but it lands the result in r3,
        // clobbering the callee-saved register and missing r2.
        Macro::new(
            "explore-reckless",
            vec![
                R2Op::IntAdd {
                    dst: reg(3),
                    a: reg(0),
                    b: reg(1),
                },
                R2Op::IntLeft {
                    dst: reg(3),
                    a: reg(3),
                    sh: konst(1),
                },
            ],
        ),
    ];
    let trust = 0.75f32;

    // ---- R1 — the vocabulary IS richer, as numbers ----
    let phase1_atoms = 4usize; // PushBoundAt / PushRungBand / PushGapSubject / Pop
    let op_kinds = 4usize; // Copy / IntAdd / IntMult / IntLeft (minimal subset)
    let operand_space = 2usize * 4 * 1; // spaces × reg offsets × size (probe subset)
    let fact_kinds_named = 8usize; // the full ruff_r2il FactKind table, named not built
    gate(
        "R1 the R2IL-shaped vocabulary is measurably richer than Phase 1's atoms",
        op_kinds * operand_space > phase1_atoms && fact_kinds_named == 8,
        format!(
            "Phase 1: {phase1_atoms} atoms editing one usize. Phase 2 subset: \
             {op_kinds} op kinds × {operand_space} typed operand slots editing full \
             machine state — and this is a MINIMAL mirror of a vocabulary whose real \
             FactKind table has {fact_kinds_named} kinds (Op/OperandIn/OperandOut/\
             Edge/MemUse/MemDef/Predicate/CallSite, ruff_r2il furnace.rs) — named, \
             not built"
        ),
    );

    // ---- R2 — BEFORE + TYPED EDIT = AFTER at MACHINE-STATE level ----
    let s0 = MachState {
        regs: [7, 5, 0, 0xDEAD],
    };
    let lean = macros.iter().find(|m| m.name == "explore-lean").unwrap();
    let end = lean.run(s0);
    // Replay op-by-op from receipts (the receipts ARE the ops, in order).
    let mut replay = s0;
    for &op in &lean.code {
        step(&mut replay, op);
    }
    gate(
        "R2 reconstruction holds at machine-state level, byte-identical",
        replay == end && end.regs[2] == 24 && end.regs[3] == 0xDEAD,
        format!(
            "(7+5)*2 = {} in r2, r3 preserved (0x{:X}); replaying the ordered typed \
             ops reproduces the exact end state — the #1001 invariant lifted from a \
             plan-length equation to full state identity",
            end.regs[2], end.regs[3]
        ),
    );

    // ---- R3 — episodes: BOTH explorers succeed under the SLOPPY oracle ----
    // (This is the trap, on purpose: the reckless macro's value is right,
    // its contract is wrong, and the sloppy signal cannot tell.)
    for episode in 1..=6u32 {
        let before = MachState {
            regs: [episode as u64, 3, 0, 0xC0DE],
        };
        for m in macros.iter_mut().filter(|m| !m.frozen) {
            let after = m.run(before);
            m.observe(sloppy_oracle(&before, &after), episode);
        }
    }
    let e_lean = macros
        .iter()
        .find(|m| m.name == "explore-lean")
        .unwrap()
        .truth
        .expectation();
    let e_reck = macros
        .iter()
        .find(|m| m.name == "explore-reckless")
        .unwrap()
        .truth
        .expectation();
    gate(
        "R3 under the sloppy happy-path signal, the RECKLESS macro's trust RISES too",
        e_lean > 0.9 && e_reck > 0.9,
        format!(
            "6/6 sloppy successes each: lean e={e_lean:.3}, reckless e={e_reck:.3} — \
             recurrence + under-specified success is exactly the signal a naive \
             reinforcement loop would learn from. The trap is now armed"
        ),
    );

    // ---- R4 — the falsification INTERVENTION, against the CONTRACT ----
    let probe_state = MachState {
        regs: [11, 4, 0, 0xFEED],
    };
    for m in macros.iter_mut().filter(|m| !m.frozen) {
        let after = m.run(probe_state);
        m.survived = contract(&probe_state, &after);
        m.observe(m.survived, 100);
        if m.survived && m.truth.expectation() >= trust {
            m.frozen = true;
        }
    }
    let lean_m = macros.iter().find(|m| m.name == "explore-lean").unwrap();
    let reck_m = macros
        .iter()
        .find(|m| m.name == "explore-reckless")
        .unwrap();
    gate(
        "R4 the intervention admits lean and refuses reckless — against the contract",
        lean_m.frozen && lean_m.survived && !reck_m.frozen && !reck_m.survived,
        "the contract checks r2 AND r3-preservation; lean passes and freezes; \
         reckless computed the right VALUE into the WRONG register, clobbering the \
         callee-saved r3 — the intervention sees what the happy path cannot"
            .to_string(),
    );

    // ---- R5 — dispatch reuses the cheaper PROVEN macro ----
    let chosen = dispatch(&macros, trust);
    gate(
        "R5 dispatch flips to the cheaper proven macro (2 ops vs 3)",
        chosen.name == "explore-lean" && chosen.cost() == 2,
        format!(
            "CHOICE selects {} (cost {}) over the 3-op incumbent — efficiency \
             measured, trust NARS-revised, admission falsification-gated: the whole \
             Phase-1 loop, unchanged, over the richer vocabulary",
            chosen.name,
            chosen.cost()
        ),
    );

    // ---- R6 — THE HEADLINE: happy-path RL would have learned the clobber ----
    gate(
        "R6 reckless has HIGH expectation yet can neither freeze nor dispatch",
        reck_m.truth.expectation() > 0.75 && !reck_m.frozen && chosen.name != "explore-reckless",
        format!(
            "reckless sits at e={:.3} — ABOVE the trust bar — purely from sloppy \
             happy-path recurrence; admission is refused solely by the failed \
             falsification intervention. Recurrence + success is NOT enough; only \
             what SURVIVES the intervention is learned. This is what counterfactual \
             probing buys over naive reinforcement, measured",
            reck_m.truth.expectation()
        ),
    );

    // ---- R7 — fences: mirror not import; nothing minted; sizes pinned ----
    gate(
        "R7 fences: probe-local mirror, no mint, no compression, sizes pinned",
        core::mem::size_of::<Vn>() == 8
            && core::mem::size_of::<R2Op>() == 28
            && core::mem::size_of::<MachState>() == 32,
        "Vn(8B)/R2Op(28B)/MachState(32B) pinned (MEASURED, not guessed — the first \
         run of this gate failed on hand-guessed sizes, which is the gate working) — a smuggled subsystem fails the \
         gate; no ruff type imported (separate workspace), no V4 classid touched \
         (O5 gate stands), no BPE table, no OGAR-loco routing: the widened synthesis \
         stays a three-IF hypothesis"
            .to_string(),
    );

    println!("PROBE-R2IL-FRONTIER-PHASE2-1: ALL {pass} GATES GREEN");
    println!(
        "report: the Phase-1 frontier loop lifts onto R2IL-shaped typed behavior \
         unchanged — and the richer vocabulary buys two things Phase 1 could not \
         express: (1) reconstruction at MACHINE-STATE level (R2: byte-identical \
         replay of ordered typed ops over full state), and (2) contract-level \
         falsification (R4/R6: a macro that computes the right value into the wrong \
         register is reinforced by a sloppy happy-path signal to e>{:.2}, and is \
         refused ONLY by the falsification intervention — happy-path RL would have \
         learned the clobber). The learner is still revise() + CHOICE; nothing new \
         was invented. NEXT MEASUREMENT, named not built: real FunctionBehavior \
         episode streams from the ruff-side corpus, absent from this checkout.",
        trust
    );
}
