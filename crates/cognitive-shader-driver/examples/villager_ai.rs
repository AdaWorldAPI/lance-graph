//! # Villager & Pet AI — reference integration for our Pumpkin fork
//!
//! Motivates the lance-graph-contract state-classification stack against
//! the NPC AI problem in a block-game server (our fork of the Pumpkin
//! Rust Minecraft server). Each contract primitive has a direct
//! correspondence to an AI tick the server has to run every game tick:
//!
//! - **Villager mood** — a `StateAnchor` (Intake/Focused/Rest/Flow/Observer/
//!   Balanced/Baseline) is the current behavioural anchor.
//! - **Pet bond strength** — the `bond` glyph (relational-family slot 20) is
//!   the scalar that grows through positive interaction and decays over time.
//! - **Villager empathy** — `WorldModelDto::user_state` is the villager's
//!   inferred model of the player (or another villager) — the engine's
//!   opponent-model.
//! - **Trading disposition** — `CollapseGate` (Flow/Hold/Block) gates when
//!   the villager is willing to commit to a trade.
//! - **Pathfinding drive** — `DriveMode::Explore` seeks new territory;
//!   `Exploit` returns to known resources; `Reflect` stays put.
//!
//! ## Run
//!
//! ```bash
//! cargo run --example villager_ai -p cognitive-shader-driver
//! ```
//!
//! Prints a tick-by-tick trace of a pet bonding sequence, a villager
//! trading interaction, and a group socialisation scene, rendered
//! through a custom `VillagerRenderer` that relabels the generic
//! anchor names as villager moods.

use lance_graph_contract::proprioception::{
    AnchorState, DriveMode, ProprioceptionAxes, StateAnchor, STATE_DIMS,
};
use lance_graph_contract::world_map::{WorldMapDto, WorldMapRenderer};

// ═══════════════════════════════════════════════════════════════════════════
// Villager-flavoured renderer
// ═══════════════════════════════════════════════════════════════════════════

/// Relabels the generic state anchors as villager moods.
/// Demonstrates the drop-in renderer pattern.
struct VillagerRenderer;

impl WorldMapRenderer for VillagerRenderer {
    fn axis_label(&self, idx: usize) -> &str {
        // Villager-flavoured axis names mapped to the canonical 11 dimensions.
        const LABELS: [&str; 11] = [
            "comfort",       // warmth
            "awareness",     // clarity
            "patience",      // depth
            "safety",        // safety
            "energy",        // vitality
            "intuition",     // insight
            "sociability",   // contact
            "unease",        // tension
            "curiosity",     // novelty
            "wonder",        // wonder
            "rapport",       // attunement
        ];
        LABELS.get(idx).copied().unwrap_or("")
    }

    fn anchor_label(&self, anchor: StateAnchor) -> &str {
        match anchor {
            StateAnchor::Intake   => "listening",
            StateAnchor::Focused  => "trading",
            StateAnchor::Rest     => "sleeping",
            StateAnchor::Flow     => "working",
            StateAnchor::Observer => "watching",
            StateAnchor::Balanced => "socialising",
            StateAnchor::Baseline => "idle",
        }
    }

    fn drive_label(&self, mode: DriveMode) -> &str {
        match mode {
            DriveMode::Explore => "wandering",
            DriveMode::Exploit => "returning",
            DriveMode::Reflect => "loitering",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pet bond tracker — uses the `bond` glyph (relational slot 20)
// ═══════════════════════════════════════════════════════════════════════════

/// A tame-able pet entity (wolf / cat / axolotl / parrot).
struct Pet {
    name: &'static str,
    species: &'static str,
    /// `bond` strength in [0.0, 1.0] — glyph GLYPHS[20] in the shader driver.
    bond: f32,
    /// Current behavioural anchor.
    anchor: StateAnchor,
}

impl Pet {
    fn new(name: &'static str, species: &'static str) -> Self {
        Self { name, species, bond: 0.0, anchor: StateAnchor::Baseline }
    }

    /// Positive interaction: feed, pet, play. Bond climbs toward 1.
    fn interact_positive(&mut self, intensity: f32) {
        self.bond = (self.bond + 0.1 * intensity).clamp(0.0, 1.0);
        // High bond shifts anchor toward Flow (active engagement).
        self.anchor = if self.bond > 0.7 {
            StateAnchor::Flow
        } else if self.bond > 0.3 {
            StateAnchor::Focused
        } else {
            StateAnchor::Baseline
        };
    }

    /// Bond decay per tick when the pet is left alone.
    fn tick_decay(&mut self) {
        self.bond = (self.bond - 0.01).max(0.0);
    }

    fn describe(&self) -> String {
        let renderer = VillagerRenderer;
        format!(
            "{} the {} — bond={:.2} mood={}",
            self.name,
            self.species,
            self.bond,
            renderer.anchor_label(self.anchor),
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Villager with axes + anchor classification
// ═══════════════════════════════════════════════════════════════════════════

struct Villager {
    #[allow(dead_code)]
    name: &'static str,
    #[allow(dead_code)]
    profession: &'static str,
    axes: ProprioceptionAxes,
}

impl Villager {
    fn new(name: &'static str, profession: &'static str) -> Self {
        Self {
            name,
            profession,
            axes: ProprioceptionAxes {
                warmth: 0.5, clarity: 0.5, depth: 0.5, safety: 0.6,
                vitality: 0.5, insight: 0.5, contact: 0.4,
                tension: 0.3, novelty: 0.5, wonder: 0.4, attunement: 0.5,
            },
        }
    }

    /// One AI tick — classify the villager's current state against
    /// the calibration anchors and return the world map.
    fn tick(&self, cycle_index: u64) -> WorldMapDto {
        WorldMapDto::from_state_vector(&self.axes.to_vector(), cycle_index)
    }

    /// Move the villager toward a target anchor (e.g. "want to trade").
    fn approach(&mut self, target: StateAnchor, step: f32) {
        let target_coords = anchor_state_ref(target).coords;
        let current = self.axes.to_vector();
        let mut next = [0.0f32; STATE_DIMS];
        for i in 0..STATE_DIMS {
            next[i] = current[i] + (target_coords[i] - current[i]) * step;
        }
        self.axes = ProprioceptionAxes::from_vector(&next);
    }
}

/// Helper to borrow an AnchorState without repeating the contract path.
fn anchor_state_ref(a: StateAnchor) -> &'static AnchorState {
    lance_graph_contract::proprioception::anchor_state(a)
}

// ═══════════════════════════════════════════════════════════════════════════
// Scene runner
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let renderer = VillagerRenderer;

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Scene 1 — Pet bonding (wolf)");
    println!("═══════════════════════════════════════════════════════════════════════");
    let mut wolf = Pet::new("Biscuit", "wolf");
    println!("  T0: {}", wolf.describe());
    for tick in 1..=6 {
        wolf.interact_positive(1.0);
        println!("  T{}: {}", tick, wolf.describe());
    }
    println!("  (leave wolf alone — bond decays)");
    for tick in 7..=10 {
        wolf.tick_decay();
        println!("  T{}: {}", tick, wolf.describe());
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Scene 2 — Villager approaching a trade state");
    println!("═══════════════════════════════════════════════════════════════════════");
    let mut librarian = Villager::new("Eldra", "librarian");
    for tick in 0..=5 {
        let map = librarian.tick(tick);
        println!("  T{}: {}", tick, map.render(&renderer));
        librarian.approach(StateAnchor::Focused, 0.25);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Scene 3 — Anchor gallery (all 7 villager moods)");
    println!("═══════════════════════════════════════════════════════════════════════");
    for anchor in StateAnchor::ALL {
        let a = anchor_state_ref(anchor);
        let map = WorldMapDto::from_state_vector(&a.coords, 0);
        println!(
            "  {:<12} rung={} drive={:<10} — {}",
            renderer.anchor_label(anchor),
            a.rung,
            renderer.drive_label(a.drive_mode()),
            map.render(&renderer),
        );
    }
}
