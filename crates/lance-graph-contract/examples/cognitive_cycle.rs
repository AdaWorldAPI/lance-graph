//! Orchestration deep-dive #1 — the active-inference cognitive cycle.
//!
//! Run it:  `cargo run -p lance-graph-contract --example cognitive_cycle`
//!
//! THE IDEA (Elixir-style):
//!   * every reasoning tactic implements ONE behaviour — `Tactic` (meta/gate/apply/run),
//!     exactly like an Elixir `@behaviour` / GenServer callback.
//!   * a "recipe" is a *pipeline* of tactics (the Elixir `|>` operator), applied in order.
//!   * each tactic is **clock-gated** by markers: a Gate-bucket tactic only fires when there
//!     is surprise to act on (CollapseGate SD not in FLOW). Most stay dark — like a CPU.
//!   * the loop runs until free energy descends below the homeostasis floor: "the shader
//!     can't resist the thinking… F descends, bits clear, shader rests."
//!
//! Nothing here is a toy: every call is a real `lance_graph_contract` kernel.

use lance_graph_contract::recipe_kernels::{kernel, GateState, ThoughtCtx, SD_FLOW};
use lance_graph_contract::recipes::recipe_by_code;

/// A "deep-think" recipe = an ordered pipeline of tactics (by their catalogue code).
/// Read it top-to-bottom like an Elixir pipe:
///   ctx |> Expand |> Decompose |> Prune |> Contradiction |> Converge |> Meta |> Filter |> Reduce
const DEEP_THINK: [&str; 8] = ["RTE", "HTD", "TCP", "CR", "CDT", "MCP", "TCF", "CUR"];

fn gate_state(sd: f32) -> GateState {
    if sd < SD_FLOW { GateState::Flow } else if sd <= 0.35 { GateState::Hold } else { GateState::Block }
}

fn main() {
    // A *surprised* starting state: high dispersion (SD=BLOCK), high free energy,
    // four candidate interpretations, and a same-topic contradiction in the belief set.
    let mut ctx = ThoughtCtx::new(vec![0.92, 0.61, 0.34, 0.12]);
    ctx.sd = 0.42; // > 0.35 → BLOCK: the gate is wide open, deep tactics will fire
    ctx.free_energy = 0.80;
    ctx.confidence = 0.50;
    ctx.beliefs = vec![(7, 0.90, 0.8), (7, 0.10, 0.7)]; // topic 7 asserted true AND false

    println!("== cognitive cycle: one Think, run to rest ==\n");
    println!("start: gate={:?} F={:.2} conf={:.2} candidates={} beliefs={}\n",
        gate_state(ctx.sd), ctx.free_energy, ctx.confidence, ctx.candidates.len(), ctx.beliefs.len());

    // The active-inference loop: keep thinking while there is surprise (gate != FLOW).
    let mut round = 0;
    while gate_state(ctx.sd) != GateState::Flow && round < 5 {
        round += 1;
        println!("── round {round} (gate {:?}) ───────────────────────", gate_state(ctx.sd));

        // Pipe the context through the recipe. Each step is `ctx |> tactic`.
        for code in DEEP_THINK {
            let rec = recipe_by_code(code).expect("recipe in catalogue");
            let k = kernel(rec.id).expect("kernel for id");
            let out = k.run(&mut ctx); // gate + apply; confidence auto-clamped
            println!(
                "  {:<5} [{:<12}] {:<22} {}  (Δconf {:+.2})",
                code,
                format!("{:?}", rec.bucket),
                rec.name,
                if out.fired { out.note } else { "· gated off (FLOW)" },
                out.delta_conf,
            );
        }

        // The cycle resolved some surprise: dispersion + free energy anneal toward rest.
        // (In the wired system this falls out of the codec sweep; here we make it explicit.)
        ctx.sd *= 0.55;
        ctx.free_energy *= 0.5;
        println!("  → after round: gate={:?} SD={:.3} F={:.3} conf={:.2}\n",
            gate_state(ctx.sd), ctx.sd, ctx.free_energy, ctx.confidence);
    }

    println!("== rest ==  the shader stopped because gate reached {:?} (SD={:.3} < FLOW {SD_FLOW}).",
        gate_state(ctx.sd), ctx.sd);
    println!("final: conf={:.2}, {} candidate(s) survived pruning, {} beliefs.",
        ctx.confidence, ctx.candidates.len(), ctx.beliefs.len());
    println!("\nKey: Gate-bucket tactics (TCP/CDT/TCF/CUR) skip while in FLOW — the markers,");
    println!("not a scheduler, decide what fires. Same `Tactic` behaviour, 34 hot-swappable units.");
}
