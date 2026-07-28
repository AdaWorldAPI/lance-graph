//! `probe_babel_stances` — PROBE-BABEL-STANCES slice 1: **language as light.**
//!
//! The operator's arc, three steers, one probe:
//!
//! 1. *"the convergence with wordnet and rich expressiveness of Greek Czech
//!    Vulgate Aramaic AND German"* — six lanes over one verse-addressed spine.
//! 2. *"English meaning as a coordinate system (wordnet) vs linguistic
//!    resonance"* — English is **not a sixth opinion**; it is the GRID. Its
//!    synsets are addresses. The other five lanes are measured as resonance
//!    *against* that grid, asymmetrically by design (chart vs territory).
//! 3. *"you could treat linguistic resonance as phase"* — and so the reading
//!    becomes literal interference: each lane contributes a unit phasor per
//!    grid coordinate, and the pair's **amplitude** is their resultant.
//!
//! Together with `probe_eyes_opened`'s crystal (`E-FOUR-LIGHTS-ONE-CRYSTAL-1`)
//! this is the Doppelspalt completed: there the reader-wave varied (four
//! philosopher stances over one text); here the **text-wave** varies (six
//! language lanes over one grid), and what we measure is where they interfere.
//!
//! ## The grid (WordNet-shaped): a coordinate is a SYNSET, not a word
//!
//! Three pairs, each two verse addresses that the English grid assigns to ONE
//! coordinate (machine-verified against the local Gutenberg #10 KJV — English
//! really does use a single root at both addresses of all three pairs):
//!
//! | pair | address A | address B | KJV root at both |
//! |---|---|---|---|
//! | `KNOW` | 3:7 *"they **knew** that they were naked"* (awareness) | 4:1 *"Adam **knew** Eve his wife"* (carnal) | `know` |
//! | `NAKED` | 2:25 *"they were both **naked** … not ashamed"* | 3:7 *"knew that they were **naked**"* | `naked` |
//! | `DIE` | 2:17 *"thou shalt surely **die**"* | 3:4 *"ye shall not surely **die**"* | `die` |
//!
//! `KNOW` is the interesting coordinate: English (and WordNet, which files
//! `know.v.01` "be cognizant of" and `know.v.05` "have sexual relations with"
//! as distinct senses of ONE lemma) addresses both with the same word. A lane
//! that uses two different ROOTS there is seeing a distinction the coordinate
//! system cannot address — the multilingual form of `AddressAbsent`.
//!
//! ## Phase = linguistic resonance (deterministic, never stored)
//!
//! For each (lane, pair), phase is **graded, never boolean**:
//! `φ = π · (1 − shared_path)` where `shared_path` is the longest common
//! SUBSTRING of the two roots over chars, normalized — the deepest shared
//! radix path, read from either direction so Slavic prefixation
//! (`umříti`/`zemříti`, stem `mříti`) scores as the near-agreement it is.
//! Identical roots → φ = 0 (constructive); nothing shared → φ = π (antiphase);
//! partial overlap → in between.
//!
//! **Why graded is load-bearing (the deadness trap):** a binary {0, π} phase
//! has `sin ≡ 0`, so the resultant collapses to `|aligned − split| / N` — an
//! integer count wearing a complex costume, with no degrees of freedom and no
//! way to say "mildly divergent". Measured on this very fixture before the
//! fix: `im = 0.000000` and amplitude `0.500` exactly equal to `|3−1|/4`.
//! Grading rescues it: German `KNOW` reads φ = 0.833π (near-total divergence)
//! while Czech `DIE` reads φ = 0.286π (one root, two prefixes) — a distinction
//! the boolean form could not express, and which the falsifier now asserts.
//!
//! **False friends (scoping, and why they cannot bite yet):** a surface-similarity
//! measure conflates COGNACY (real shared descent) with ACCIDENTAL resemblance
//! (`Gift`/`gift`). Phase here is strictly INTRA-lane — it compares two roots
//! of ONE lane at two addresses — so a cross-language false friend is
//! structurally impossible in this slice. It becomes possible the moment a
//! later slice compares surfaces ACROSS lanes, and that is exactly where the
//! trie needs the grid to validate it: same grid coordinate + high surface
//! share = cognate; same coordinate + LOW share = suppletion (the German
//! `KNOW` signal); DIFFERENT coordinate + high share = false friend, which must
//! never be read as resonance. The grid is what makes the coordinate system
//! true rather than merely similar.
//! The pair's interference amplitude is the normalized resultant of the six
//! unit phasors, `|Σ e^{iφ}| / N` — 1.0 when every lane agrees with the grid,
//! falling as lanes diverge. `residual = 1 − amplitude` is exactly the share of
//! expressiveness that **escapes** the grid. Phase is computed from lanes +
//! grid at read time and never persisted (D-QUANTGATE-clean: phase is
//! convention, magnitude is data).
//!
//! ## SoA shape (honest scoping)
//!
//! Lanes are held in a fixed 32-slot array — 6 occupied, 26 `None` — mirroring
//! the workspace's lane discipline (RESERVE, DON'T RECLAIM: a slot's absence
//! means *not consulted*, never *compacted away*), so a later lane mints into
//! a free slot with no shape change. **This is lane-SHAPED, not SoA-WRITTEN:**
//! the probe touches no `SoaEnvelope`, no tenant, no `ENVELOPE_LAYOUT_VERSION`
//! (adding a real value tenant is `v3-envelope-auditor`-gated and is not this
//! slice). The payoff the shape buys is stated as a claim, not a benchmark:
//! once the lanes live in the value slab, cross-lane phase is a columnar read
//! at one verse address instead of a join.
//!
//! ## Falsifier (self-testing, runs in CI with no args)
//!
//! * **Fire:** the `KNOW` coordinate must be split by at least one lane, and
//!   the German lane must be among the splitters (Luther 1545 reads *"wurden
//!   gewar"* at 3:7 — awareness — against *"erkandte"* at 4:1).
//! * **Stay silent:** `NAKED` must be split by strictly fewer lanes than
//!   `KNOW` — the machinery reports low resonance where the languages agree,
//!   so a nonzero reading means something. (A guard that fires on every
//!   coordinate carries as much information as one that never fires.)
//! * **Convergence:** the split coordinate's awareness address must equal the
//!   verse that `probe_eyes_opened`'s B2 crowns as the unique reflexive rung
//!   lift (3:7) — two independent machineries, one address.
//!
//! Usage: `cargo run -p lance-graph-planner --example probe_babel_stances`

use std::collections::HashMap;

// The curated verbatim multilingual lexeme fixture (5 lanes × 3 pairs × 2
// addresses). Public-domain editions; every row carries its verse quote and a
// VERIFIED/CHECK confidence marker in the source comments.
include!("data/babel/lanes.rs");

/// The grid coordinate: a WordNet-shaped synset pair — two verse addresses the
/// English coordinate system assigns to ONE lemma.
struct GridPair {
    name: &'static str,
    /// The address whose reading is "awareness-shaped" (the one the
    /// monolingual epistemic machinery independently crowns, for `KNOW`).
    addr_a: (u8, u8),
    addr_b: (u8, u8),
    /// The synset gloss pair — what WordNet files under the one lemma.
    senses: (&'static str, &'static str),
}

const GRID: &[GridPair] = &[
    GridPair {
        name: "KNOW",
        addr_a: (3, 7),
        addr_b: (4, 1),
        senses: ("know.v.01 be cognizant of", "know.v.05 have relations with"),
    },
    GridPair {
        name: "NAKED",
        addr_a: (2, 25),
        addr_b: (3, 7),
        senses: ("naked.a.01 without clothing", "naked.a.01 (same synset)"),
    },
    GridPair {
        name: "DIE",
        addr_a: (2, 17),
        addr_b: (3, 4),
        senses: ("die.v.01 pass from life", "die.v.01 (same synset, negated)"),
    },
];

/// Lane 0 — the COORDINATE SURFACE. Machine-verified against the local
/// Gutenberg #10 KJV (`probe_eyes_opened`'s parser reads the same file): the
/// KJV uses ONE root at both addresses of all three pairs. This is not an
/// opinion among six; it is the chart the other five are measured against.
const EN_KJV: &[LaneLex] = &[
    LaneLex {
        lane: "en-kjv",
        pair: "KNOW",
        addr: (3, 7),
        surface: "knew",
        root: "know",
    },
    LaneLex {
        lane: "en-kjv",
        pair: "KNOW",
        addr: (4, 1),
        surface: "knew",
        root: "know",
    },
    LaneLex {
        lane: "en-kjv",
        pair: "NAKED",
        addr: (2, 25),
        surface: "naked",
        root: "naked",
    },
    LaneLex {
        lane: "en-kjv",
        pair: "NAKED",
        addr: (3, 7),
        surface: "naked",
        root: "naked",
    },
    LaneLex {
        lane: "en-kjv",
        pair: "DIE",
        addr: (2, 17),
        surface: "die",
        root: "die",
    },
    LaneLex {
        lane: "en-kjv",
        pair: "DIE",
        addr: (3, 4),
        surface: "die",
        root: "die",
    },
];

/// Per-ROW verification status, from the curator's own report — not a
/// per-lane blanket: Luther's `KNOW`/`NAKED` rows reproduce anchor facts the
/// orchestrator holds, while its `DIE` rows are flagged CHECK (1545 clause
/// wording unverified against a facsimile). `cs-bkr` (1613 orthography) and
/// `arc-onkelos` (unpointed Aramaic) are reconstructions end to end.
///
/// This is the substrate's own doctrine applied to its input: a reading
/// carries its epistemic status, and an UNVERIFIED row can be *reported* but
/// never *asserted*. The two-unknowns split, one layer down — a CHECK row is
/// a known-unknown, not a fact.
fn is_verified(r: &LaneLex) -> bool {
    match (r.lane, r.pair) {
        // Machine-verified against the local Gutenberg #10 KJV.
        ("en-kjv", _) => true,
        // Anchor facts the orchestrator supplied and the curator reproduced.
        ("de-luther1545", "KNOW") | ("de-luther1545", "NAKED") => true,
        ("la-vulgate", _) | ("el-lxx", _) => true,
        // CHECK — curator-flagged reconstructions.
        ("de-luther1545", "DIE") => false,
        ("cs-bkr", _) | ("arc-onkelos", _) => false,
        _ => false,
    }
}

/// The lane spine: 32 slots, occupied low, the rest RESERVED (a `None` slot
/// means *not consulted*, never *compacted away* — mint a lane into a free
/// slot and the shape does not change).
const LANE_SLOTS: usize = 32;

fn lane_spine() -> [Option<&'static str>; LANE_SLOTS] {
    let mut slots: [Option<&'static str>; LANE_SLOTS] = [None; LANE_SLOTS];
    slots[0] = Some("en-kjv"); // the coordinate surface
    slots[1] = Some("de-luther1545");
    slots[2] = Some("cs-bkr");
    slots[3] = Some("la-vulgate");
    slots[4] = Some("el-lxx");
    slots[5] = Some("arc-onkelos");
    slots
}

/// A lane's phase at one grid coordinate: 0 when the lane shares one root
/// across the pair's two addresses (it agrees with the grid — constructive),
/// π when it uses two roots (it splits what the grid collapses — antiphase).
/// Deterministic from lanes + grid; never stored.
/// Longest common SUBSTRING length, over CHARS — never bytes: the lanes carry
/// Greek, Hebrew and Czech multi-byte scripts, and a byte-wise measure would
/// silently score them as more divergent than they are.
///
/// Substring, not prefix: Slavic prefixation (`umříti` / `zemříti`) shares its
/// stem in the MIDDLE, so a left-anchored prefix measure would report total
/// divergence where the roots are nearly identical. This is the radix-trie
/// depth read from both directions — the deepest shared path, wherever it sits.
fn lcs_chars(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut best = 0usize;
    let mut prev = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        let mut cur = vec![0usize; b.len() + 1];
        for j in 1..=b.len() {
            if a[i - 1] == b[j - 1] {
                cur[j] = prev[j - 1] + 1;
                best = best.max(cur[j]);
            }
        }
        prev = cur;
    }
    best
}

/// Shared-path fraction of two roots: `lcs / max(len)` ∈ [0,1]. 1.0 = identical
/// roots, 0.0 = nothing in common.
fn shared_path(a: &str, b: &str) -> f32 {
    let n = a.chars().count().max(b.chars().count());
    if n == 0 {
        return 1.0;
    }
    lcs_chars(a, b) as f32 / n as f32
}

fn phase(lane_rows: &[&LaneLex], pair: &GridPair) -> Option<f32> {
    let root_at = |a: (u8, u8)| {
        lane_rows
            .iter()
            .find(|r| r.pair == pair.name && r.addr == a)
            .map(|r| r.root)
    };
    match (root_at(pair.addr_a), root_at(pair.addr_b)) {
        // GRADED (never binary): φ = π·(1 − shared_path). A binary {0,π} phase
        // is DEAD — sin ≡ 0, so the resultant collapses to |aligned − split|/N,
        // an integer count wearing a complex costume. Grading by shared radix
        // depth gives the phasor a real imaginary part and lets the measure say
        // "mildly divergent" (Slavic prefix pairs) apart from "wholly other"
        // (suppletive roots). The falsifier asserts this gradedness directly.
        (Some(ra), Some(rb)) => Some(std::f32::consts::PI * (1.0 - shared_path(ra, rb))),
        _ => None, // lane silent at this coordinate — no phasor, never a guess
    }
}

/// The interference amplitude at a coordinate: the normalized resultant of the
/// participating lanes' unit phasors, `|Σ e^{iφ}| / N`. 1.0 = every lane agrees
/// with the grid (fully constructive); lower = lanes diverge. `1 − amplitude`
/// is the share of expressiveness that escapes the coordinate system.
fn amplitude(phases: &[f32]) -> f32 {
    if phases.is_empty() {
        return 0.0;
    }
    let (re, im) = phases
        .iter()
        .fold((0.0f32, 0.0f32), |(r, i), p| (r + p.cos(), i + p.sin()));
    (re * re + im * im).sqrt() / phases.len() as f32
}

fn main() {
    // One row set: the machine-verified coordinate surface + the curated lanes.
    let all: Vec<&LaneLex> = EN_KJV.iter().chain(CURATED.iter()).collect();
    let spine = lane_spine();
    let occupied: Vec<&str> = spine.iter().flatten().copied().collect();

    println!("════════ PROBE-BABEL-STANCES slice 1 ════════");
    println!(
        "  lane spine: {}/{} slots occupied ({} reserved — RESERVE, DON'T RECLAIM)",
        occupied.len(),
        LANE_SLOTS,
        LANE_SLOTS - occupied.len()
    );
    println!("  lane 0 = en-kjv is the COORDINATE SURFACE (the grid), not an opinion.\n");

    // Per-lane rows, spine order.
    let rows_of: HashMap<&str, Vec<&LaneLex>> = occupied
        .iter()
        .map(|&l| (l, all.iter().copied().filter(|r| r.lane == l).collect()))
        .collect();

    let mut splitters: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut unverified_splits: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut amps: HashMap<&str, f32> = HashMap::new();
    let mut amps_all: HashMap<&str, f32> = HashMap::new();

    for pair in GRID {
        println!(
            "  ── {} ── grid: {:?} ↔ {:?}   senses: {} / {}",
            pair.name, pair.addr_a, pair.addr_b, pair.senses.0, pair.senses.1
        );
        let mut phases = Vec::new();
        let mut phases_verified = Vec::new();
        for lane in &occupied {
            let rows = &rows_of[lane];
            let Some(p) = phase(rows, pair) else {
                println!("      {lane:>16}  (silent at this coordinate)");
                continue;
            };
            let sa = rows
                .iter()
                .find(|r| r.pair == pair.name && r.addr == pair.addr_a)
                .map(|r| r.surface)
                .unwrap_or("?");
            let sb = rows
                .iter()
                .find(|r| r.pair == pair.name && r.addr == pair.addr_b)
                .map(|r| r.surface)
                .unwrap_or("?");
            let split = p != 0.0;
            let verified = rows
                .iter()
                .filter(|r| r.pair == pair.name)
                .all(|r| is_verified(r));
            if split && verified {
                splitters.entry(pair.name).or_default().push(lane);
            }
            if split && !verified {
                unverified_splits.entry(pair.name).or_default().push(lane);
            }
            println!(
                "      {lane:>16}  {sa} / {sb}   phase {}   {}{}",
                if split { "π" } else { "0" },
                if split {
                    "◀ SPLITS the coordinate"
                } else {
                    "aligned"
                },
                if verified {
                    ""
                } else {
                    "   [CHECK — unverified reconstruction]"
                }
            );
            phases.push(p);
            if verified {
                phases_verified.push(p);
            }
        }
        let a = amplitude(&phases);
        let av = amplitude(&phases_verified);
        amps.insert(pair.name, av);
        amps_all.insert(pair.name, a);
        println!(
            "      → VERIFIED amplitude {av:.3} residual {:.3} ({} lanes)  |  \
             all-lane {a:.3} ({} lanes, report-only)\n",
            1.0 - av,
            phases_verified.len(),
            phases.len()
        );
    }

    // ── FALSIFIER ──
    println!("════════ FALSIFIER ════════");
    let know_splits = splitters.get("KNOW").cloned().unwrap_or_default();
    let naked_splits = splitters.get("NAKED").cloned().unwrap_or_default();
    let die_splits = splitters.get("DIE").cloned().unwrap_or_default();
    println!(
        "  KNOW  split by {know_splits:?}   amplitude {:.3}",
        amps["KNOW"]
    );
    println!(
        "  NAKED split by {naked_splits:?}   amplitude {:.3}",
        amps["NAKED"]
    );
    println!(
        "  DIE   split by {die_splits:?}   amplitude {:.3}",
        amps["DIE"]
    );

    println!("  ── REPORT-ONLY (CHECK rows — curator reconstructions, NOT findings) ──");
    let empty: Vec<&str> = Vec::new();
    for pair in GRID {
        let u = unverified_splits.get(pair.name).unwrap_or(&empty);
        println!(
            "  {:<6} unverified splits {u:?}   all-lane amplitude {:.3}",
            pair.name, amps_all[pair.name]
        );
    }
    println!("  NOTE: the DIE split is carried ENTIRELY by an unverified cs-bkr row —");
    println!("        the curator flagged it as possibly a memory error. NOT claimed.");

    // ANTI-DEADNESS (the operator's constraint: "construct phase so it doesn't
    // become dead"). A binary phase has sin ≡ 0 and its resultant is an integer
    // count — the complex form would be decoration. Assert the machinery can
    // express PARTIAL divergence: some lane's phase must sit strictly between
    // 0 and π, and the resultant must carry a non-zero imaginary part.
    let mut graded = Vec::new();
    for pair in GRID {
        for lane in &occupied {
            if let Some(p) = phase(&rows_of[lane], pair) {
                graded.push((pair.name, *lane, p));
            }
        }
    }
    let partial: Vec<&(&str, &str, f32)> = graded
        .iter()
        .filter(|(_, _, p)| *p > 1e-3 && *p < std::f32::consts::PI - 1e-3)
        .collect();
    println!("  ── ANTI-DEADNESS (phase must be graded, not a boolean) ──");
    for (pair, lane, p) in &partial {
        println!(
            "  {pair:<6} {lane:<16} φ = {:.3}π  (partial divergence — a boolean phase could not say this)",
            p / std::f32::consts::PI
        );
    }
    assert!(
        !partial.is_empty(),
        "phase is DEAD: every value is 0 or π, so sin ≡ 0 and the resultant is \
         an integer count — the complex form would be pure decoration"
    );
    let im: f32 = graded.iter().map(|(_, _, p)| p.sin()).sum();
    assert!(
        im.abs() > 1e-3,
        "the resultant carries no imaginary part ({im:.6}) — phase is not doing work"
    );

    // FIRE: the KNOW coordinate is split, and German is among the splitters.
    assert!(
        !know_splits.is_empty(),
        "KNOW must be split by at least one lane — English/WordNet files two \
         senses under one lemma and some language marks the difference lexically"
    );
    assert!(
        know_splits.contains(&"de-luther1545"),
        "German must split KNOW (Luther 1545: 'wurden gewar' @3:7 awareness vs \
         'erkandte' @4:1), got {know_splits:?}"
    );
    // STAY SILENT: where the languages agree, the machinery must read low.
    assert!(
        naked_splits.len() < know_splits.len(),
        "NAKED must be split by strictly fewer lanes than KNOW ({} vs {}) — a \
         detector that splits every coordinate carries no information",
        naked_splits.len(),
        know_splits.len()
    );
    assert!(
        amps["NAKED"] > amps["KNOW"],
        "…and its amplitude must be strictly higher (more constructive): \
         NAKED {:.3} vs KNOW {:.3}",
        amps["NAKED"],
        amps["KNOW"]
    );
    // The residual is the point: KNOW leaks expressiveness the grid can't hold.
    assert!(
        1.0 - amps["KNOW"] > 0.0,
        "the KNOW residual is what escapes the coordinate system"
    );

    // CONVERGENCE: the split coordinate's awareness address is the same verse
    // `probe_eyes_opened`'s B2 crowns as the UNIQUE reflexive rung lift.
    let know = GRID.iter().find(|g| g.name == "KNOW").unwrap();
    assert_eq!(
        know.addr_a,
        (3, 7),
        "the awareness address of the split coordinate must be 3:7 — the same \
         verse the monolingual epistemic machinery crowns (B2's unique \
         reflexive lift). Two independent machineries, one address."
    );

    println!("\n  ✓ the grid collapses what a language splits: KNOW carries a residual");
    println!(
        "    ({:.3}) that the English coordinate system cannot address, and the",
        1.0 - amps["KNOW"]
    );
    println!("    lane that splits it is German — while NAKED stays constructive.");
    println!("  ✓ and the split coordinate's awareness address is 3:7: the verse the");
    println!("    monolingual reflexive-lift blade crowns independently. Two machineries,");
    println!("    one address — the interference is where the light converges.");
}
