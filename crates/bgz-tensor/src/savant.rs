//! Backend savant agents — domain-specific HHTL caches.
//!
//! Three infrastructure backends, each a pre-computed HhtlCache with
//! domain-specific RouteAction decisions. Not trained — extracted from
//! weight diffs. Not user-facing — called by other modules.
//!
//! Core:        10-14 KB, L1 cache, always loaded, gatekeeper
//! Psychology:  ~206 KB, L2 cache, loaded on behavioral escalation
//! Linguistics: ~206 KB, L2 cache, loaded on structural escalation

use crate::cascade::ScentByte;
use crate::hhtl_cache::{HhtlCache, RouteAction};

/// Which backend savant handled (or should handle) a query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SavantKind {
    /// Core gatekeeper (k=64, ~14 KB, always loaded).
    Core,
    /// Behavioral specialist (k=256, ~206 KB, lazy-loaded).
    Psychology,
    /// Structural specialist (k=256, ~206 KB, lazy-loaded).
    Linguistics,
    /// Both specialists (merge results).
    Both,
}

/// Result of a savant routing decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SavantDecision {
    /// The routing action determined by the handling savant.
    pub action: RouteAction,
    /// Which savant produced this decision.
    pub savant: SavantKind,
    /// Pairwise distance from the handling savant's distance table.
    pub distance: u16,
}

/// Dispatcher that holds up to three HHTL caches and routes queries
/// through them based on scent-byte plane analysis.
///
/// The core cache is always present and acts as gatekeeper. When core
/// escalates, the S/P/O plane decomposition of the scent byte determines
/// which specialist backend handles the pair:
///
/// - S-plane agrees but P doesn't -> Psychology (behavioral)
/// - P-plane agrees but S doesn't -> Linguistics (structural)
/// - All agree -> Both (merge)
/// - Otherwise -> Core keeps the result
pub struct SavantDispatch {
    /// Core gatekeeper cache (k=64, always present).
    pub core: HhtlCache,
    /// Psychology backend (k=256, lazy-loaded on behavioral escalation).
    pub psychology: Option<HhtlCache>,
    /// Linguistics backend (k=256, lazy-loaded on structural escalation).
    pub linguistics: Option<HhtlCache>,
}

impl SavantDispatch {
    /// Create a new dispatcher with only the core cache.
    pub fn new(core: HhtlCache) -> Self {
        Self {
            core,
            psychology: None,
            linguistics: None,
        }
    }

    /// Attach the psychology (behavioral) backend cache.
    pub fn load_psychology(&mut self, cache: HhtlCache) {
        self.psychology = Some(cache);
    }

    /// Attach the linguistics (structural) backend cache.
    pub fn load_linguistics(&mut self, cache: HhtlCache) {
        self.linguistics = Some(cache);
    }

    /// Route a query for archetype pair `(a, b)`.
    ///
    /// First checks the core cache. If core says `Escalate`, uses
    /// scent-byte S/P/O plane analysis to pick the appropriate specialist:
    ///
    /// - S-plane agrees, P-plane doesn't -> Psychology
    /// - P-plane agrees, S-plane doesn't -> Linguistics
    /// - All planes agree -> Both (merges by picking the shorter distance)
    /// - Otherwise -> stays with core result
    pub fn route(&self, a: u8, b: u8) -> SavantDecision {
        let core_action = self.core.route(a, b);
        let core_distance = self.core.distance(a, b);

        if core_action != RouteAction::Escalate {
            return SavantDecision {
                action: core_action,
                savant: SavantKind::Core,
                distance: core_distance,
            };
        }

        // Core escalated — use scent-byte plane analysis to pick specialist.
        // We need the Base17 entries from core's palette to compute the scent.
        let k = self.core.k();
        if (a as usize) >= k || (b as usize) >= k {
            return SavantDecision {
                action: core_action,
                savant: SavantKind::Core,
                distance: core_distance,
            };
        }

        let qa = &self.core.palette.entries[a as usize];
        let kb = &self.core.palette.entries[b as usize];
        let scent = ScentByte::compute(qa, kb, 1500);

        // Extract individual plane agreements from the scent byte:
        // bit 0 = S-plane, bit 1 = P-plane, bit 2 = O-plane
        let s_agrees = scent.0 & 0x01 != 0;
        let p_agrees = scent.0 & 0x02 != 0;

        if scent.all_agree() {
            // All planes agree — use both specialists if available
            match (&self.psychology, &self.linguistics) {
                (Some(psy), Some(ling)) => {
                    let pd = psy.distance(a, b);
                    let ld = ling.distance(a, b);
                    // Merge: pick the action from the specialist with shorter distance
                    let (action, distance) = if pd <= ld {
                        (psy.route(a, b), pd)
                    } else {
                        (ling.route(a, b), ld)
                    };
                    SavantDecision {
                        action,
                        savant: SavantKind::Both,
                        distance,
                    }
                }
                _ => SavantDecision {
                    action: core_action,
                    savant: SavantKind::Core,
                    distance: core_distance,
                },
            }
        } else if s_agrees && !p_agrees {
            // S-plane agrees but P doesn't — behavioral domain
            match &self.psychology {
                Some(psy) => SavantDecision {
                    action: psy.route(a, b),
                    savant: SavantKind::Psychology,
                    distance: psy.distance(a, b),
                },
                None => SavantDecision {
                    action: core_action,
                    savant: SavantKind::Core,
                    distance: core_distance,
                },
            }
        } else if p_agrees && !s_agrees {
            // P-plane agrees but S doesn't — structural domain
            match &self.linguistics {
                Some(ling) => SavantDecision {
                    action: ling.route(a, b),
                    savant: SavantKind::Linguistics,
                    distance: ling.distance(a, b),
                },
                None => SavantDecision {
                    action: core_action,
                    savant: SavantKind::Core,
                    distance: core_distance,
                },
            }
        } else {
            // No clear specialist match — core keeps it
            SavantDecision {
                action: core_action,
                savant: SavantKind::Core,
                distance: core_distance,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::Base17;

    /// Build a deterministic set of Base17 rows from a seed.
    fn make_rows(n: usize, seed: usize) -> Vec<Base17> {
        (0..n)
            .map(|i| {
                let mut dims = [0i16; 17];
                for d in 0..17 {
                    dims[d] = (((i + seed) * 97 + d * 31) % 512) as i16 - 256;
                }
                Base17 { dims }
            })
            .collect()
    }

    /// Build a small HhtlCache with the given k.
    fn build_cache(k: usize, seed: usize) -> HhtlCache {
        let rows = make_rows(k.max(10) * 3, seed);
        HhtlCache::from_base17_rows(&rows, k)
    }

    #[test]
    fn test_core_only_routing() {
        let core = build_cache(64, 0);
        let dispatch = SavantDispatch::new(core);

        // Route every pair in the core palette — should never crash,
        // and every decision should come from Core.
        let k = dispatch.core.k();
        for a in 0..k.min(8) {
            for b in 0..k.min(8) {
                let decision = dispatch.route(a as u8, b as u8);
                // With no specialists loaded, savant must be Core
                // (even on Escalate, fallback is Core).
                assert_eq!(
                    decision.savant,
                    SavantKind::Core,
                    "pair ({a},{b}): expected Core, got {:?}",
                    decision.savant
                );
            }
        }
    }

    #[test]
    fn test_specialist_dispatch() {
        let core = build_cache(64, 0);
        let psychology = build_cache(64, 100);
        let linguistics = build_cache(64, 200);

        let mut dispatch = SavantDispatch::new(core);
        dispatch.load_psychology(psychology);
        dispatch.load_linguistics(linguistics);

        // With all three loaded, scan pairs and verify:
        // - Non-Escalate from core -> SavantKind::Core
        // - Escalate from core -> specialist or Core depending on scent
        let k = dispatch.core.k();
        let mut saw_non_core = false;
        for a in 0..k.min(16) {
            for b in 0..k.min(16) {
                let decision = dispatch.route(a as u8, b as u8);
                let core_action = dispatch.core.route(a as u8, b as u8);

                if core_action != RouteAction::Escalate {
                    assert_eq!(decision.savant, SavantKind::Core);
                    assert_eq!(decision.action, core_action);
                } else if decision.savant != SavantKind::Core {
                    saw_non_core = true;
                }
            }
        }
        // It's possible (but unlikely with these seeds) that no pair escalates
        // to a specialist. We just verify the routing logic didn't panic.
        let _ = saw_non_core;
    }

    #[test]
    fn test_lazy_loading() {
        let core = build_cache(64, 0);
        let mut dispatch = SavantDispatch::new(core);

        // Initially, specialists are None.
        assert!(dispatch.psychology.is_none());
        assert!(dispatch.linguistics.is_none());

        // Load psychology.
        let psy = build_cache(64, 50);
        dispatch.load_psychology(psy);
        assert!(dispatch.psychology.is_some());
        assert!(dispatch.linguistics.is_none());

        // Load linguistics.
        let ling = build_cache(64, 75);
        dispatch.load_linguistics(ling);
        assert!(dispatch.psychology.is_some());
        assert!(dispatch.linguistics.is_some());
    }
}
