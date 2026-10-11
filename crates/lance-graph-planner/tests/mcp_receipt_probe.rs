//! **D-NAL-MCP-0 — does recipe 10 (MCP, "Meta-Cognition Prompting") read an
//! earlier executed operation, or only two scalars?**
//!
//! EXPERIMENTAL, test-local. Nothing in production changes.
//!
//! The catalogue (`recipes.rs`, id 10) describes MCP's substrate as
//! "MUL DK + Brier calibration; Meta lane". Its kernel (`recipe_kernels.rs`,
//! `impl Tactic for Mcp`) declares `requires = {Confidence, FreeEnergy}` and
//! lowers confidence by 0.2 when `confidence > 0.7 && free_energy > 0.5`.
//! A Brier score needs past predictions paired with later outcomes, and
//! `ThoughtCtx` carries neither, so MCP cannot compute one.
//!
//! What this probe establishes, each by execution:
//!
//! 1. MCP's verdict is a function of `(confidence, free_energy)` alone: every
//!    other `ThoughtCtx` field can change and the outcome does not.
//! 2. MCP is history-blind: a well-calibrated past and an overconfident past
//!    give the same verdict on the same present context.
//! 3. The missing object is a RECEIPT: an executed step (`ExecutedStep`,
//!    which `run_recipes` already returns) plus the confidence it left behind
//!    plus the outcome observed later, under a stable identity. A test-local
//!    reader that feeds receipts into production
//!    `mul::compass::LearningLoop` (a real Brier score) separates the two
//!    histories that MCP cannot.
//! 4. Ablation: with no receipts the reader abstains; MCP still acts.
//! 5. No evidence laundering: the same receipt counted twice must not move the
//!    verdict. `LearningLoop` itself has no identity and DOES count a repeated
//!    observation, so de-duplication has to happen at the receipt.
//!
//! The receipt reader below is the PROPOSED shape, not a recipe: `Tactic::apply`
//! takes `&mut ThoughtCtx` only, so no recipe can be handed a receipt today.

use std::collections::BTreeSet;

use lance_graph_contract::recipe_dispatch::{run_recipes, ExecutedStep};
use lance_graph_contract::recipe_kernels::ThoughtCtx;
use lance_graph_planner::mul::compass::LearningLoop;

const MCP: u8 = 10;

fn ctx(confidence: f32, free_energy: f32) -> ThoughtCtx {
    let mut c = ThoughtCtx::new(vec![0.3, 0.6]);
    c.confidence = confidence;
    c.free_energy = free_energy;
    c
}

/// Run MCP once; return (fired, delta_conf, confidence afterwards).
fn mcp(c: &mut ThoughtCtx) -> (bool, f32, f32) {
    let steps = run_recipes(c, &[MCP]);
    assert_eq!(steps.len(), 1, "MCP is a registered recipe");
    let o = &steps[0].outcome;
    (o.fired, o.delta_conf, c.confidence)
}

// ── 1. two scalars, nothing else ──────────────────────────────────────────

#[test]
fn mcp_verdict_depends_on_confidence_and_free_energy_only() {
    // The rule fires on one side of the boundary and not the other, so the
    // invariance checks below are not vacuously comparing two no-ops.
    let (_, fire, _) = mcp(&mut ctx(0.9, 0.6));
    let (_, quiet, _) = mcp(&mut ctx(0.9, 0.4));
    assert!(fire < 0.0 && quiet == 0.0, "fixture: one side fires, one does not");

    for (conf, fe) in [(0.9, 0.6), (0.9, 0.4), (0.5, 0.9), (0.71, 0.51)] {
        let mut base = ctx(conf, fe);
        let expected = mcp(&mut base);
        // Every field MCP does not declare in `requires`, perturbed hard.
        let mut other = ctx(conf, fe);
        other.sd = 0.9;
        other.dissonance = 0.8;
        other.temperature = 0.05;
        other.rung = 9;
        other.candidates = vec![1.0, 0.0, 0.5, 0.25];
        other.beliefs = vec![(7, 1.0, 0.9), (7, 0.0, 0.9), (8, 0.4, 0.2)];
        assert_eq!(
            mcp(&mut other),
            expected,
            "conf={conf} fe={fe}: MCP must not see undeclared fields"
        );
    }
}

// ── receipts: the object MCP is missing ───────────────────────────────────

/// One executed step, the confidence it left in the context, the outcome
/// observed later, and an identity. `ExecutedStep` supplies the first part
/// today; the other three have no carrier.
#[derive(Debug, Clone)]
struct Receipt {
    run: u32,
    step: ExecutedStep,
    predicted: f32,
    hit: bool,
}

/// A history of `n` executed MCP steps that each left confidence `p`, of
/// which `hits` were later confirmed. MCP runs for real; the outcomes are the
/// later evidence the context never carried.
fn history(n: u32, p: f32, hits: u32) -> Vec<Receipt> {
    (0..n)
        .map(|run| {
            // free_energy 0.4: MCP's rule stays quiet, so the step leaves `p`.
            let mut c = ctx(p, 0.4);
            let step = run_recipes(&mut c, &[MCP]).remove(0);
            Receipt {
                run,
                step,
                predicted: c.confidence,
                hit: run < hits,
            }
        })
        .collect()
}

/// PROPOSED meta-operation: read prior receipts, deduplicate by identity,
/// score calibration with production `LearningLoop`, and say how far to move
/// confidence. `None` = abstain (no evidence about its own past).
fn calibrate(receipts: &[Receipt]) -> Option<f32> {
    let mut seen = BTreeSet::new();
    let mut lp = LearningLoop::new();
    let (mut sum_p, mut hits, mut n) = (0.0f64, 0u32, 0u32);
    for r in receipts {
        if !seen.insert((r.run, r.step.id)) {
            continue; // the same execution again is not new evidence
        }
        let outcome = if r.hit { 1.0 } else { 0.0 };
        lp.observe(f64::from(r.predicted), outcome);
        sum_p += f64::from(r.predicted);
        hits += u32::from(r.hit);
        n += 1;
    }
    if n < 10 {
        return None;
    }
    if lp.is_calibrated() {
        return Some(0.0);
    }
    let hit_rate = f64::from(hits) / f64::from(n);
    Some((hit_rate - sum_p / f64::from(n)) as f32)
}

// ── 2. history-blind MCP vs a receipt reader ──────────────────────────────

#[test]
fn mcp_cannot_tell_a_calibrated_past_from_an_overconfident_one() {
    let calibrated = history(10, 0.9, 9); // said 0.9, right 9 of 10
    let overconfident = history(10, 0.9, 2); // said 0.9, right 2 of 10
    assert!(
        calibrated.iter().chain(&overconfident).all(|r| r.step.outcome.fired),
        "fixture: every receipt comes from an MCP step that actually ran"
    );

    // The present context is the same in both worlds, so MCP must answer the
    // same way: it has no input through which the past could reach it.
    let now_a = mcp(&mut ctx(0.9, 0.4));
    let now_b = mcp(&mut ctx(0.9, 0.4));
    assert_eq!(now_a, now_b);
    assert_eq!(now_a.1, 0.0, "MCP calls this 'calibration ok' in both worlds");

    // The receipt reader separates them.
    let a = calibrate(&calibrated).expect("ten receipts");
    let b = calibrate(&overconfident).expect("ten receipts");
    assert_eq!(a, 0.0, "a calibrated past leaves confidence alone");
    assert!(
        (b - (0.2 - 0.9)).abs() < 1e-6,
        "an overconfident past pulls confidence toward its hit rate, got {b}"
    );
}

// ── 4. ablation ───────────────────────────────────────────────────────────

#[test]
fn without_receipts_the_reader_abstains_while_mcp_still_acts() {
    assert_eq!(calibrate(&[]), None);
    assert_eq!(calibrate(&history(9, 0.9, 2)), None, "below 10 receipts");
    let (fired, delta, _) = mcp(&mut ctx(0.9, 0.6));
    assert!(fired && delta < 0.0, "MCP lowers confidence with no past at all");
}

// ── 5. no evidence laundering ─────────────────────────────────────────────

#[test]
fn a_repeated_receipt_is_not_new_evidence() {
    let base = history(10, 0.9, 9);
    // Re-deliver the one miss nine more times: same run, same step.
    let miss = base.iter().find(|r| !r.hit).expect("one miss").clone();
    let mut echoed = base.clone();
    echoed.extend(std::iter::repeat(miss).take(9));
    assert_eq!(calibrate(&echoed), calibrate(&base));

    // The production carrier alone has no identity: the echo moves it.
    let mut raw = LearningLoop::new();
    for r in &echoed {
        raw.observe(f64::from(r.predicted), if r.hit { 1.0 } else { 0.0 });
    }
    let mut clean = LearningLoop::new();
    for r in &base {
        clean.observe(f64::from(r.predicted), if r.hit { 1.0 } else { 0.0 });
    }
    assert!(clean.is_calibrated() && !raw.is_calibrated());
    assert!(raw.brier_score() > clean.brier_score());
}
