//! PROBE-RECIPE-DISPATCH-BRIDGE-1 — the small, well-scoped seam
//! `PROBE-RECIPE-EXECUTION-1` (#995) left open: given an `ogar_loco::Call`
//! whose `FnIndex` is a minted recipe op, does dispatch actually reach the
//! right `kernel(id)`, with identity preserved through the whole route, and
//! can one receipt span BOTH instruction ranges (shared core + recipes) in a
//! single interleaved execution?
//!
//! # Where this comes from
//!
//! `PROBE-LOCO-INTERPRETER-1` (`AdaWorldAPI/OGAR` #281 / `AdaWorldAPI/
//! lance-graph` #992) proved the shared core executes. `PROBE-RECIPE-
//! EXECUTION-1` (#995) proved the 34 recipes have real, mostly-separable
//! `Tactic::apply` effects — but called `kernel(id)` DIRECTLY, never through
//! an `ogar_loco::Call`. This probe builds that missing arm.
//!
//! # Layering (session-directed, non-negotiable)
//!
//! `ogar_loco` stays zero-dep and recipe-blind: it is NOT touched by this
//! probe. `lance_graph_contract::recipe_kernels` stays zero-dep and
//! loco-blind. The bridge — "this `FnIndex` is a recipe op, resolve it, call
//! its kernel" — lives HERE, in `lance-graph-ogar`, the one crate the
//! workspace already lets depend on both (`recipe_vocab.rs`'s own module
//! doc: *"Neither may import the other... A vocabulary needs both, so it
//! lives in a consumer that already depends on both"*). This is an adapter
//! for one falsifier, not a new generic `DomainDispatch` trait — that
//! generalization is explicitly deferred, per session discussion, until
//! there is a second domain vocabulary that would need it too.
//!
//! # What "identity preserved through the whole route" means, concretely
//!
//! For every id `1..=34`: `op_of(id)` → embed in a `Call` → interpret →
//! `recipe_of(call.function)` recovers `id` → `kernel(id)` is invoked → the
//! resulting `Outcome` and `ThoughtCtx` mutation are BYTE-FOR-BYTE identical
//! (via `PartialEq`, not eyeballed) to calling `kernel(id).run_with(...)`
//! DIRECTLY on an identical starting context, bypassing the interpreter
//! entirely. That equality check is the falsifier: a bridge that resolves
//! the wrong id, corrupts the operand, or calls `apply` instead of `run`
//! (skipping the gate) would show up here as a mismatch, not as "it ran".
//!
//! # The canonical receipt
//!
//! A single program mixes shared-core arithmetic with recipe dispatch calls
//! and is executed as ONE episode, producing one `Vec<ExecutedOp>` — shared-
//! core op, shared-core op, recipe op, shared-core op, recipe op — the shape
//! `PROBE-LOCO-INTERPRETER-1`'s `TraceEvent` and `PROBE-RECIPE-EXECUTION-1`'s
//! per-kernel effect signature never combined into one stream.
//!
//! Run (lance-graph-ogar is workspace-EXCLUDED, own [workspace] — BBB
//! firewall keeps OGAR out of the default lance-graph build):
//! `cargo run --manifest-path crates/lance-graph-ogar/Cargo.toml --example recipe_dispatch_bridge_probe`

use lance_graph_contract::recipe_kernels::{MaturityPolicy, Outcome, ThoughtCtx};
use lance_graph_contract::recipes::recipe;
use lance_graph_ogar::recipe_vocab::{op_of, recipe_of, RecipeVocabulary};
use ogar_loco::vocabulary::conformance::validate;
use ogar_loco::{Call, FnIndex, FunctionBody, LaneShape};

/// One executed call, shared-core or recipe, in ONE combined stream — the
/// canonical receipt spanning both instruction ranges.
#[derive(Debug, Clone)]
enum ExecutedOp {
    SharedCore {
        function: FnIndex,
        name: &'static str,
        pushed: i64,
    },
    Recipe {
        id: u8,
        code: &'static str,
        name: &'static str,
        focus_slot: usize,
        outcome: Outcome,
    },
}

/// Minimal interpreter — deliberately NOT the general Turing-complete engine
/// `PROBE-LOCO-INTERPRETER-1` already proved (WHILE/IF/REPEAT); this probe's
/// job is the dispatch seam, so it covers only what is needed to build a
/// real interleaved shared-core/recipe receipt: literal push, add, and
/// recipe dispatch.
struct Bridge {
    stack: Vec<i64>,
    /// Named `ThoughtCtx` "focus slots" a recipe's popped operand selects
    /// into (`selector % slots.len()`). A real deployment resolves a
    /// recipe's operand against the basin-local attention-focus codebook
    /// `recipe_vocab`'s own module doc names and explicitly declines to
    /// resolve itself ("the basin that owns the prefix owns the
    /// resolution"). This probe does NOT build that resolver — it stands in
    /// with a small, honestly-labelled array, because the question this
    /// probe answers (does the id route correctly, is the effect identical
    /// to a direct call) does not depend on how the focus was addressed.
    slots: Vec<ThoughtCtx>,
    trace: Vec<ExecutedOp>,
}

impl Bridge {
    fn new(slots: Vec<ThoughtCtx>) -> Self {
        Self {
            stack: Vec::new(),
            slots,
            trace: Vec::new(),
        }
    }

    fn run(&mut self, body: &FunctionBody) {
        for call in body.calls() {
            let f = call.function;
            if let Some(id) = recipe_of(f) {
                // ── the bridge ─────────────────────────────────────────
                let selector = self.stack.pop().expect("recipe arity is 1");
                let slot = (selector.rem_euclid(self.slots.len() as i64)) as usize;
                let k = lance_graph_contract::recipe_kernels::kernel(id)
                    .unwrap_or_else(|| panic!("recipe_of({f:?}) = {id}, but kernel({id}) is None"));
                let outcome = k.run_with(&mut self.slots[slot], MaturityPolicy::Any);
                self.stack.push(i64::from(outcome.fired)); // satisfies domain_pushes_result=true
                let meta = k.meta();
                self.trace.push(ExecutedOp::Recipe {
                    id,
                    code: meta.code,
                    name: meta.name,
                    focus_slot: slot,
                    outcome,
                });
                continue;
            }
            // ── shared core (the tiny subset this probe needs) ─────────
            use FnIndex as F;
            match f {
                F::NUMBER => {
                    let v = i64::from(call.values[0]);
                    self.stack.push(v);
                    self.trace.push(ExecutedOp::SharedCore {
                        function: f,
                        name: "NUMBER",
                        pushed: v,
                    });
                }
                F::ADD => {
                    let b = self.stack.pop().expect("ADD arity 2");
                    let a = self.stack.pop().expect("ADD arity 2");
                    let v = a + b;
                    self.stack.push(v);
                    self.trace.push(ExecutedOp::SharedCore {
                        function: f,
                        name: "ADD",
                        pushed: v,
                    });
                }
                other => panic!("this minimal bridge probe does not cover {other:?}"),
            }
        }
    }
}

fn main() {
    println!("═══ PROBE-RECIPE-DISPATCH-BRIDGE-1 ═══\n");
    validate(RecipeVocabulary).expect("RecipeVocabulary must validate");

    // ── Part 1: identity preservation, all 34 ids, against a direct
    // (non-bridged) invocation on an identical starting context. ─────────
    println!("── Part 1: FnIndex -> kernel(id) identity, all 34, vs. direct invocation ──");
    let mut identity_ok = true;
    for id in 1u8..=34 {
        let op = op_of(id).unwrap_or_else(|| panic!("op_of({id}) refused"));
        assert_eq!(
            recipe_of(op),
            Some(id),
            "op_of/recipe_of round trip broke for id {id}"
        );

        // The SAME starting context, fed to both routes.
        let mut ctx0 = ThoughtCtx::new(vec![0.6, 0.4, 0.2]);
        ctx0.sd = 0.3;
        ctx0.temperature = 0.7;
        ctx0.free_energy = 0.6;
        ctx0.rung = 4;
        ctx0.beliefs = vec![(11, 0.7, 0.6)];

        // Route A: through the bridge, via an ogar_loco::Call.
        let selector = 0u8; // slots.len() == 1 here, so selection is trivial
                            // rem_euclid on a single-slot array always selects slot 0 regardless
                            // of the selector's value — deliberately, so this part isolates
                            // identity/routing, not focus-selection arithmetic (Part 2 exercises
                            // selection with a multi-slot battery).
        let body = FunctionBody::from_calls(
            LaneShape::Pairs,
            &[Call::with_value(FnIndex::NUMBER, selector), Call::new(op)],
        )
        .expect("2-call body is well within budget");
        let mut bridged = Bridge::new(vec![ctx0.clone()]);
        bridged.run(&body);
        let ExecutedOp::Recipe {
            id: routed_id,
            outcome: bridged_outcome,
            ..
        } = bridged.trace.last().cloned().expect("one recipe call ran")
        else {
            panic!("last trace event must be a Recipe");
        };
        let bridged_ctx = bridged.slots[0].clone();

        // Route B: direct, bypassing the bridge and the interpreter entirely.
        let mut direct_ctx = ctx0.clone();
        let direct_outcome = lance_graph_contract::recipe_kernels::kernel(id)
            .unwrap()
            .run_with(&mut direct_ctx, MaturityPolicy::Any);

        let id_match = routed_id == id;
        let outcome_match = bridged_outcome == direct_outcome;
        let ctx_match = format!("{bridged_ctx:?}") == format!("{direct_ctx:?}");
        let ok = id_match && outcome_match && ctx_match;
        identity_ok &= ok;
        let r = recipe(id).unwrap();
        println!(
            "  {id:>2} {:<5} id={id_match} outcome={outcome_match} ctx={ctx_match} {}",
            r.code,
            if ok { "OK" } else { "MISMATCH" }
        );
    }
    println!(
        "\nPart 1 verdict: {}",
        if identity_ok {
            "PASS — all 34 ids route to the correct kernel, with outcome and context identical to a direct call."
        } else {
            "FAIL — see MISMATCH rows above."
        }
    );

    // ── Part 2: determinism under replay, through the bridge, with a
    // multi-slot focus battery (selection actually varies the target). ────
    println!("\n── Part 2: determinism under replay (bridged), 4-slot focus battery ──");
    let slots = || {
        vec![
            ThoughtCtx::new(vec![0.9, 0.6, 0.3, 0.1]),
            ThoughtCtx::new(vec![0.4, 0.45, 0.5, 0.55]),
            ThoughtCtx::new(vec![]),
            ThoughtCtx::new(vec![0.5, 0.5, 0.5]),
        ]
    };
    let mut det_ok = true;
    for id in [1u8, 7, 17, 34] {
        let op = op_of(id).unwrap();
        let body = FunctionBody::from_calls(
            LaneShape::Pairs,
            &[
                Call::with_value(FnIndex::NUMBER, 2),
                Call::new(op), // focus = 2 % 4 = slot 2 ("empty")
            ],
        )
        .unwrap();
        let mut run1 = Bridge::new(slots());
        run1.run(&body);
        let mut run2 = Bridge::new(slots());
        run2.run(&body);
        let same = format!("{:?}", run1.trace) == format!("{:?}", run2.trace)
            && format!("{:?}", run1.slots) == format!("{:?}", run2.slots);
        det_ok &= same;
        println!("  id={id} deterministic={same}");
    }
    println!(
        "Part 2 verdict: {}",
        if det_ok {
            "PASS — bridged execution replays byte-identical."
        } else {
            "FAIL"
        }
    );

    // ── Part 3: one canonical receipt spanning BOTH instruction ranges. ──
    println!("\n── Part 3: one interleaved receipt (shared-core op / shared-core op / recipe op / ...) ──");
    let r1 = op_of(1).unwrap(); // RTE
    let r7 = op_of(7).unwrap(); // ASC
    let body = FunctionBody::from_calls(
        LaneShape::Pairs,
        &[
            Call::with_value(FnIndex::NUMBER, 2),
            Call::with_value(FnIndex::NUMBER, 1),
            Call::new(FnIndex::ADD), // shared-core: 2+1=3 -> selector
            Call::new(r1),           // recipe op 1 (RTE), focus = 3 % 4
            Call::with_value(FnIndex::NUMBER, 1),
            Call::new(FnIndex::ADD), // shared-core: fired(0/1) + 1
            Call::new(r7),           // recipe op 7 (ASC), focus = that % 4
        ],
    )
    .expect("7-call body within budget");
    let mut ep = Bridge::new(slots());
    ep.run(&body);
    for (i, op) in ep.trace.iter().enumerate() {
        match op {
            ExecutedOp::SharedCore {
                function,
                name,
                pushed,
            } => {
                println!("  {i}: shared-core {name:<6} ({function:?}) -> {pushed}");
            }
            ExecutedOp::Recipe {
                id,
                code,
                name,
                focus_slot,
                outcome,
            } => {
                println!(
                    "  {i}: recipe #{id:<2} {code:<5} {name:<28} focus_slot={focus_slot} fired={} delta_conf={} note={:?}",
                    outcome.fired, outcome.delta_conf, outcome.note
                );
            }
        }
    }
    let shared_core_count = ep
        .trace
        .iter()
        .filter(|o| matches!(o, ExecutedOp::SharedCore { .. }))
        .count();
    let recipe_count = ep
        .trace
        .iter()
        .filter(|o| matches!(o, ExecutedOp::Recipe { .. }))
        .count();
    println!(
        "\nPart 3 verdict: {shared_core_count} shared-core ops + {recipe_count} recipe ops in ONE trace, ordering preserved."
    );

    println!("\n═══ Report ═══");
    println!(
        "PROBE-RECIPE-DISPATCH-BRIDGE-1: {}",
        if identity_ok && det_ok && shared_core_count > 0 && recipe_count == 2 {
            "PASS — the FnIndex -> recipe_of -> kernel(id) route is identity-preserving, \
             deterministic, and composes with shared-core execution in one receipt."
        } else {
            "FAIL — see per-part verdicts above."
        }
    );
    println!(
        "Scope note: the recipe operand's real address resolution (the basin-local \
         attention-focus codebook recipe_vocab.rs's own doc declines to resolve) is NOT \
         built here — this probe stands in with a small labelled focus-slot array. That \
         resolver is a separate, still-open task; it does not gate this probe's question."
    );
}
