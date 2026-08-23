//! PROBE-RECIPE-EXECUTION-1 — do the 34 lance-graph-ogar recipes generate
//! *distinguishable* state transitions above the shared computational core,
//! or do they collapse onto indistinguishable `ThoughtCtx` behaviour?
//!
//! # Where this comes from
//!
//! `PROBE-LOCO-INTERPRETER-1` (`AdaWorldAPI/OGAR` #281,
//! `AdaWorldAPI/lance-graph` #992) built and ran a minimal interpreter for
//! `ogar_loco::FunctionBody` over the shared core, and explicitly left
//! KC1 — "do the 34 recipes have separable executable effects" — untested,
//! because the recipe layer's semantics were assumed to live only in
//! `lance-graph-ogar`'s `ThoughtCtx`/`recipe_dispatch` wiring, outside that
//! interpreter's scope.
//!
//! Reframed (per session discussion, 2026-08-23): KC1 was never "can all 34
//! be executed via `ogar_loco::Call` bytes" — that is an ABI-plumbing
//! question. The actual question is behavioural: **given the same starting
//! context, do different recipe ids produce state transitions an observer
//! could tell apart?** That question does not need the `ogar_loco::Call`
//! bridge at all — `lance_graph_contract::recipe_kernels::kernel(id)` already
//! dispatches by the SAME `1..=34` id space `recipe_vocab::op_of`/
//! `recipe_of` uses for the `FnIndex` mapping (checked below as a sanity
//! assertion, not assumed).
//!
//! # Scope
//!
//! This probe calls `Tactic::run` directly through the existing
//! `kernel(id)` registry — it does NOT build or exercise an
//! `ogar_loco::Call`/`FunctionBody` bridge. That bridge (an `FnIndex` in
//! `RECIPE_OP_BASE..RECIPE_OP_END` → `recipe_of` → `kernel` → `run`
//! dispatch arm inside an interpreter) is a separate, smaller, still-open
//! task this probe's results motivate but does not perform.
//!
//! Run (lance-graph-ogar is workspace-EXCLUDED, own [workspace] — BBB
//! firewall keeps OGAR out of the default lance-graph build):
//! `cargo run --manifest-path crates/lance-graph-ogar/Cargo.toml --example recipe_execution_probe`

use lance_graph_contract::recipe_kernels::{all_kernels, MaturityPolicy, ThoughtCtx, ThoughtField};
use lance_graph_contract::recipes::recipe;
use lance_graph_ogar::recipe_vocab::{op_of, recipe_of};
use ogar_loco::DOMAIN_FLOOR;

/// One (starting-context, id) trial's observable effect — what a downstream
/// consumer could actually tell apart, not the raw f32 values (which would
/// make every trial "distinguishable" by floating-point noise alone).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EffectSignature {
    fired: bool,
    /// Coarse: did `delta_conf` move at all (sign), not its exact magnitude.
    delta_conf_sign: i8,
    /// Which `ThoughtCtx` fields actually changed value, as a bitmask over
    /// the same 8 fields `ThoughtMask`/`ThoughtField` name — computed by
    /// comparing before/after, independent of what the tactic *declared* it
    /// might write.
    changed_fields: u8,
    /// Coarse bucket of the resulting candidate-vector length change (a
    /// prune/filter tactic shrinks it; nothing here grows it).
    candidates_len_delta_sign: i8,
}

fn field_bit(f: ThoughtField) -> u8 {
    1 << (f as u8)
}

fn changed_mask(before: &ThoughtCtx, after: &ThoughtCtx) -> u8 {
    let mut m = 0u8;
    if before.sd != after.sd {
        m |= field_bit(ThoughtField::Sd);
    }
    if before.free_energy != after.free_energy {
        m |= field_bit(ThoughtField::FreeEnergy);
    }
    if before.dissonance != after.dissonance {
        m |= field_bit(ThoughtField::Dissonance);
    }
    if before.temperature != after.temperature {
        m |= field_bit(ThoughtField::Temperature);
    }
    if before.confidence != after.confidence {
        m |= field_bit(ThoughtField::Confidence);
    }
    if before.rung != after.rung {
        m |= field_bit(ThoughtField::Rung);
    }
    if before.candidates != after.candidates {
        m |= field_bit(ThoughtField::Candidates);
    }
    if before.beliefs != after.beliefs {
        m |= field_bit(ThoughtField::Beliefs);
    }
    m
}

fn sign(x: f32) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// A small battery of REAL starting contexts, chosen to exercise the
/// conditional branches the effect-census tests in `recipe_kernels.rs`
/// already probe (hot/cold/empty) — not cherry-picked to flatter any
/// particular id, since every id sees every context.
fn battery() -> Vec<(&'static str, ThoughtCtx)> {
    let mut hot = ThoughtCtx::new(vec![0.9, 0.6, 0.3, 0.1]);
    hot.sd = 0.5;
    hot.temperature = 0.9;
    hot.free_energy = 0.9;
    hot.rung = 3;
    hot.beliefs = vec![(7, 0.9, 0.8), (7, 0.1, 0.7)];

    let mut cold = ThoughtCtx::new(vec![0.4, 0.45, 0.5, 0.55]);
    cold.sd = 0.05;
    cold.temperature = 0.1;
    cold.free_energy = 0.05;
    cold.rung = 8;
    cold.beliefs = vec![(3, 0.6, 0.5)];

    let mut empty = ThoughtCtx::new(vec![]);
    empty.sd = 0.25;

    let neutral = ThoughtCtx::new(vec![0.5, 0.5, 0.5]);

    vec![
        ("hot", hot),
        ("cold", cold),
        ("empty", empty),
        ("neutral", neutral),
    ]
}

fn main() {
    println!("═══ PROBE-RECIPE-EXECUTION-1 ═══\n");

    // ── Sanity check: the FnIndex <-> recipe-id mapping used by
    // recipe_vocab (the ogar-loco ABI side) round-trips against every id
    // the kernel registry (the ThoughtCtx execution side) actually serves.
    // If this fails, the "same id space" claim in the module doc is false
    // and the rest of this probe's framing is wrong.
    let kernels = all_kernels();
    assert_eq!(kernels.len(), 34, "kernel registry must serve exactly 34");
    for id in 1u8..=34 {
        let f = op_of(id).unwrap_or_else(|| panic!("op_of({id}) refused"));
        assert!(
            f.0 >= DOMAIN_FLOOR,
            "recipe op {id} landed below DOMAIN_FLOOR: {f:?}"
        );
        assert_eq!(
            recipe_of(f),
            Some(id),
            "op_of/recipe_of round trip broke for id {id}"
        );
        let r = recipe(id).unwrap_or_else(|| panic!("recipes::recipe({id}) missing"));
        assert_eq!(r.id, id, "catalogue id mismatch for {id}");
    }
    println!("Sanity: FnIndex(0x90+id-1) <-> recipe_id <-> kernel(id) <-> recipes::recipe(id)");
    println!("        round-trips for all 34 ids. [G] — not assumed.\n");

    let ctxs = battery();
    let policy = MaturityPolicy::Any; // report everything; classify by maturity below.

    // signature -> ids that produced it, per id we also keep its own set of
    // per-context signatures for the report.
    let mut per_id_signatures: Vec<(u8, &str, &str, Vec<EffectSignature>)> = Vec::new();

    for id in 1u8..=34 {
        let k = kernels[usize::from(id) - 1];
        let meta = k.meta();
        let mut sigs = Vec::with_capacity(ctxs.len());
        for (_name, ctx0) in &ctxs {
            let mut ctx = ctx0.clone();
            let outcome = k.run_with(&mut ctx, policy);
            let sig = EffectSignature {
                fired: outcome.fired,
                delta_conf_sign: sign(outcome.delta_conf),
                changed_fields: changed_mask(ctx0, &ctx),
                candidates_len_delta_sign: sign(
                    ctx.candidates.len() as f32 - ctx0.candidates.len() as f32,
                ),
            };
            sigs.push(sig);
        }
        per_id_signatures.push((id, meta.code, meta.name, sigs));
    }

    // ── Separability: is id X's FULL signature vector (across the whole
    // battery) unique among the 34, or does some OTHER id produce the exact
    // same sequence of effect signatures on every context in the battery?
    let mut collisions: Vec<(u8, u8)> = Vec::new();
    for i in 0..per_id_signatures.len() {
        for j in (i + 1)..per_id_signatures.len() {
            if per_id_signatures[i].3 == per_id_signatures[j].3 {
                collisions.push((per_id_signatures[i].0, per_id_signatures[j].0));
            }
        }
    }

    println!("── Per-recipe effect (fired / Δconf sign / changed-fields mask / Δlen sign), one column per battery context [{}] ──",
        ctxs.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", "));
    for (id, code, name, sigs) in &per_id_signatures {
        let cells: Vec<String> = sigs
            .iter()
            .map(|s| {
                format!(
                    "{}{:+}/{:#04b}/{:+}",
                    if s.fired { "F" } else { "-" },
                    s.delta_conf_sign,
                    s.changed_fields,
                    s.candidates_len_delta_sign
                )
            })
            .collect();
        println!("  {id:>2} {code:<5} {name:<28} {}", cells.join("  "));
    }

    let distinct = {
        let mut seen: Vec<&Vec<EffectSignature>> = Vec::new();
        for (_, _, _, sigs) in &per_id_signatures {
            if !seen.iter().any(|s| **s == *sigs) {
                seen.push(sigs);
            }
        }
        seen.len()
    };

    println!("\n═══ Report ═══");
    println!(
        "Distinguishable recipes: {distinct}/34 have a signature-across-the-battery unique among the 34."
    );
    if collisions.is_empty() {
        println!("Collisions: none — no two ids share an identical effect signature across every context tested.");
    } else {
        println!("Collisions ({} pairs):", collisions.len());
        for (a, b) in &collisions {
            let ca = per_id_signatures[usize::from(*a) - 1].1;
            let cb = per_id_signatures[usize::from(*b) - 1].1;
            println!("  {a} ({ca}) == {b} ({cb})");
        }
    }

    // ── Maturity breakdown, MEASURED (not quoted from a doc comment).
    let mut operational = 0u32;
    let mut demonstration = 0u32;
    let mut stub = 0u32;
    let mut moves_confidence = 0u32;
    for k in &kernels {
        match k.maturity() {
            lance_graph_contract::recipe_kernels::KernelMaturity::Operational => operational += 1,
            lance_graph_contract::recipe_kernels::KernelMaturity::Demonstration => {
                demonstration += 1
            }
            lance_graph_contract::recipe_kernels::KernelMaturity::Stub => stub += 1,
        }
        if k.moves_confidence() {
            moves_confidence += 1;
        }
    }
    println!(
        "\nMaturity (measured): Operational={operational} Demonstration={demonstration} Stub={stub} (of 34)"
    );
    println!("moves_confidence()=true (measured): {moves_confidence}/34");

    // ── Operational-only separability — the "production dispatch" reading:
    // a channel that only samples production kernels, does IT still see 34
    // distinguishable behaviours, or fewer?
    let op_only: Vec<&(u8, &str, &str, Vec<EffectSignature>)> = per_id_signatures
        .iter()
        .filter(|(id, ..)| {
            kernels[usize::from(*id) - 1].maturity()
                == lance_graph_contract::recipe_kernels::KernelMaturity::Operational
        })
        .collect();
    let op_distinct = {
        let mut seen: Vec<&Vec<EffectSignature>> = Vec::new();
        for (_, _, _, sigs) in &op_only {
            if !seen.iter().any(|s| **s == **sigs) {
                seen.push(sigs);
            }
        }
        seen.len()
    };
    println!(
        "Distinguishable AMONG OPERATIONAL-ONLY kernels: {op_distinct}/{} (ProductionOnly-dispatch reading).",
        op_only.len()
    );

    println!(
        "\nKC1 (reframed): the 34 recipes {} generate separable state transitions above the \
         shared core, {} the recipe vocabulary is executable cognition rather than pure \
         taxonomy — as measured by this battery. This does NOT yet prove they execute via \
         the ogar_loco::Call/FunctionBody ABI: no FnIndex(0x90..0xB2) -> kernel(id) dispatch \
         arm exists in any interpreter today. That bridge is the next, smaller, open task.",
        if distinct == 34 { "DO" } else { "DO NOT ALL" },
        if distinct == 34 {
            "confirming"
        } else {
            "with exceptions noted above — qualifying"
        }
    );
}
