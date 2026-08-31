//! **The agnostic wave dispatcher** — schedule × tunnel × executed recipes,
//! wired HERE, not in a consumer.
//!
//! Operator-ruled (2026-08-31, verbatim): *"i meant to wire EVERYTHING in
//! lancegraph including the 8 domain spog AND alpha tunnel"* — the
//! composition itself is thinking substrate. A consumer supplies exactly
//! three things, all data: a base spine (`&[NodeRow]`), a seeded
//! [`ThoughtCtx`] (its domain adaptation), and the positional key map from
//! ctx candidates to spine addresses. Everything else — the dependency-wave
//! plan, the per-rung lanes, the recipe execution, the claim rule — happens
//! here, in one call.
//!
//! # The claim rule — a measured footprint, not invented salience
//!
//! A lane claims the addresses of the candidates its OWN execution moved
//! ([`touched_indices`]): changed expectation or removal counts, an
//! untouched candidate never does. Each lane computes on its own microcopy
//! of the seed ctx (readonly base + owned microcopies — the borrow
//! strategy); the base spine is only ever read.
//!
//! An unmapped candidate (`keys[i] == None`) is a consumer-reported hole —
//! this module never fabricates an address for it.

use crate::alpha::{AlphaAddr, AlphaAllocation, AlphaStamp};
use crate::alpha_tunnel::AlphaTunnel;
use crate::canonical_node::NodeRow;
use crate::recipe_dispatch::run_recipes;
use crate::recipe_kernels::ThoughtCtx;
use crate::rung_schedule::schedule_for;

/// Which candidate positions an execution MOVED — the write footprint a lane
/// claims. Changed (value moved) and removed (vector shrank) count; untouched
/// does not.
#[must_use]
pub fn touched_indices(before: &[f32], after: &[f32]) -> Vec<usize> {
    (0..before.len())
        .filter(|&i| match after.get(i) {
            Some(a) => (before[i] - a).abs() > f32::EPSILON,
            None => true,
        })
        .collect()
}

/// What a dispatch produced: the merged scanpath plus the plan's shape.
#[derive(Debug, Clone)]
pub struct WaveDispatchOutcome {
    /// The deterministic scanpath from [`AlphaTunnel::merge`] — `(address,
    /// stamp)` in `(rung, seq)` order. The path IS the index.
    pub scanpath: Vec<(AlphaAddr, AlphaStamp)>,
    /// Sequential waves the dependency plan needed.
    pub waves: usize,
}

/// Drive one seeded context through the rung tunnel over a base spine.
///
/// Per wave, per rung: one lane, one ctx microcopy, ONE
/// [`run_recipes`] call with `wave.ids_at(rung)`, then claims for the
/// candidates that call moved. `keys[i]` maps `seed.candidates[i]` to its
/// spine address (`None` = reported hole, never claimed). Deterministic:
/// same inputs ⇒ byte-identical scanpath.
#[must_use]
pub fn dispatch_thought(
    base: &[NodeRow],
    seed: &ThoughtCtx,
    keys: &[Option<AlphaAddr>],
    cycle: u32,
) -> WaveDispatchOutcome {
    let alloc = AlphaAllocation::over(base);
    let plan = schedule_for(seed);
    let mut tunnel = AlphaTunnel::over(&alloc, cycle);
    for wave in &plan.waves {
        tunnel.run_wave(wave, |rung, ids, lane| {
            let mut ctx = seed.clone();
            let before = ctx.candidates.clone();
            let _steps = run_recipes(&mut ctx, ids);
            for i in touched_indices(&before, &ctx.candidates) {
                if let Some(Some(k)) = keys.get(i) {
                    let _ = lane.claim(*k, rung);
                }
            }
        });
    }
    WaveDispatchOutcome {
        scanpath: tunnel.merge(),
        waves: plan.waves.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical_node::{EdgeBlock, NodeGuid};

    fn base(n: u32) -> Vec<NodeRow> {
        (0..n)
            .map(|i| NodeRow {
                key: NodeGuid::new(0x0E0E_0000 + i, 1, 2, 3, 0x55, i + 1),
                edges: EdgeBlock::default(),
                value: [0u8; 480],
            })
            .collect()
    }

    /// Two-sided: moved counts, untouched does not, removed counts.
    #[test]
    fn touched_indices_counts_moved_and_removed_never_untouched() {
        let before = [0.5_f32, 0.7, 0.9, 0.2];
        let after = [0.5_f32, 0.1, 0.9];
        assert_eq!(touched_indices(&before, &after), vec![1, 3]);
        assert!(touched_indices(&before, &before).is_empty());
    }

    /// A grounded seed fills the tunnel with exactly the MAPPED addresses;
    /// an empty seed leaves it empty. Disable-verified: removing the
    /// `run_recipes` call (no thinking) empties the scanpath — attention
    /// comes from EXECUTION.
    #[test]
    fn a_grounded_seed_fills_the_tunnel_and_an_empty_one_leaves_it_empty() {
        let b = base(8);
        let mut seed = ThoughtCtx::new(vec![0.9, 0.6, 0.4, 0.3]);
        seed.free_energy = 0.8;
        seed.beliefs = vec![(0, 0.9, 0.5), (1, 0.6, 0.4)];
        // Map candidates 0..3 to rows 0..3; candidate 2 is a reported hole.
        let keys: Vec<Option<AlphaAddr>> =
            vec![Some(b[0].key), Some(b[1].key), None, Some(b[3].key)];
        let d = dispatch_thought(&b, &seed, &keys, 5);
        assert!(!d.scanpath.is_empty(), "a grounded seed must claim");
        assert!(
            d.scanpath.iter().all(|(a, _)| *a != b[2].key),
            "an unmapped candidate is NEVER claimed — no fabricated address"
        );
        assert!(
            d.scanpath
                .iter()
                .all(|(a, _)| b.iter().any(|r| r.key == *a)),
            "every claim is a base address"
        );

        let empty = dispatch_thought(&b, &ThoughtCtx::new(vec![]), &[], 5);
        assert!(empty.scanpath.is_empty(), "no candidates, no attention");
    }

    /// Same inputs ⇒ byte-identical scanpath.
    #[test]
    fn dispatch_is_deterministic() {
        let b = base(6);
        let mut seed = ThoughtCtx::new(vec![0.8, 0.5, 0.2]);
        seed.beliefs = vec![(0, 0.8, 0.6)];
        let keys: Vec<Option<AlphaAddr>> = b.iter().take(3).map(|r| Some(r.key)).collect();
        let x = dispatch_thought(&b, &seed, &keys, 9);
        let y = dispatch_thought(&b, &seed, &keys, 9);
        assert!(!x.scanpath.is_empty(), "empty would be a vacuous compare");
        assert_eq!(x.scanpath, y.scanpath);
    }
}
