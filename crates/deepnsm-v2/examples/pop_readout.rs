//! `pop_readout` — does population-typicality (an object's distance to its
//! subject's own basin centroid) improve the exploration-frontier RANKING
//! function beyond what `FrontierEdge::curiosity` already gives?
//!
//! PROBE-POP-READOUT-1, plan `.claude/plans/post-teardown-buildup-survey-v1.md`
//! §6, D-POP-1.
//!
//! ## What is actually under test
//!
//! This probe evaluates the RANKING FUNCTION
//! [`lance_graph_contract::exploration::FrontierEdge::curiosity`] (and its MUL-
//! weighted sibling [`FrontierEdge::curiosity_gestalt`]) — which is exactly what
//! [`MassExplorer::next_frontier_edge`] (upstream, `lance-graph-planner`) sorts
//! its frontier by — rather than the whole `MassExplorer` fetch/extract/revise
//! loop. TD-EXPLORATION-1 leaves `MassExplorer::from_graph`'s frontier
//! structurally empty on this repo's real corpora, so the loop cannot be run
//! end-to-end here; the RANKING is the one piece both reachable and directly
//! testable in isolation, and it is the piece a population readout could
//! actually improve (it changes which edge gets picked next, not what the
//! fetch/extract machinery does with it).
//!
//! ## The label
//!
//! A candidate `(s, p, o)` triple mined from the PREFIX of a book (verses
//! before a split point) is labelled `1.0` iff the exact same `(s, p, o)`
//! recurs in the SUFFIX (verses at or after the split) — i.e. does ranking
//! this candidate highly help find something the rest of the book actually
//! confirms.
//!
//! ## Strengthening beyond the pre-registered design: the frequency control
//!
//! The plan's original arms are `curiosity` (A0) and `curiosity_gestalt`
//! magnitude (A1) vs. the population readout (AP). Both `curiosity` and `pop`
//! are functions of quantities that co-vary with SAMPLE COUNT `n` (how often a
//! candidate was actually observed in the prefix): `curiosity`'s `novelty` term
//! is `1/(n+1)`-shaped, and `n` also drives the confidence that seeds a
//! candidate's basin membership. A raw cross-candidate correlation between any
//! such quantity and a recurrence label is therefore vulnerable to exactly the
//! member-count ARTIFACT `E-BASIN-WIDTH-IS-N-ARTIFACT-1` (and its restatement in
//! `bible_wave.rs`'s own G-SRS3b-3 leg, "does the composite predict beyond
//! size?") measured for basin width: a quantity that merely tracks `n` will
//! look predictive of anything that itself correlates with `n`, with no
//! semantic content behind it. This probe therefore adds, beyond the
//! pre-registered design: an explicit frequency arm `AF = n`, and the decisive
//! statistic is [`partial_spearman`] of the population readout against the
//! label, controlling for frequency — the same instrument `bible_wave.rs`
//! already uses for its own size-confound (G-SRS3b-3). This is a
//! strengthening of the pre-registered design, not a deviation from it: every
//! pre-registered arm and metric is still computed and reported unchanged.
//!
//! ## What this probe does NOT run, and why
//!
//! The plan additionally names a Fisher-z leg and a RollingFloor-occupancy leg
//! (`helix`). Neither runs here, for two separate reasons:
//!
//! 1. `helix` is not a dependency of `deepnsm-v2`. Adding one here would pull
//!    the `ndarray` git fork into this crate's build graph for a probe whose
//!    whole point is to stay inside the already-available contract + this
//!    crate's own dependency set.
//! 2. Under a RANK-based combination (which is what every metric in this probe
//!    is — precision@k, Spearman, partial Spearman), a Fisher-z transform is a
//!    STRICTLY MONOTONE reparameterization of a correlation coefficient. A
//!    strictly monotone transform cannot change any RANKING it is folded into,
//!    so Fisher-z is analytically inert for a rank-based readout; it would only
//!    matter for a MAGNITUDE-valued combination rule, which this probe does not
//!    use.
//!
//! ## Run
//!
//! ```sh
//! cargo run --manifest-path crates/deepnsm-v2/Cargo.toml --example pop_readout -- <spo.tsv>
//! ```
//!
//! `<spo.tsv>` is the 7-column export `bible_wave --export <spo.tsv>` writes
//! (`subject_id \t subject_word \t predicate_id \t predicate_word \t object_id
//! \t object_word \t verse`); only the id and verse columns are read here.

use lance_graph_contract::exploration::{FrontierEdge, NarsTruth};
use lance_graph_contract::mul::{
    DkPosition, FlowState, Homeostasis, MulAssessment, TrustQualia, TrustTexture,
};
use lance_graph_contract::sensorium::GraphSignals;

use deepnsm_v2::{basin_self_code, load_cam96_codes, load_cam96_space, partial_spearman, Cam96};
use std::collections::HashSet;
use std::path::PathBuf;

/// The trained artifacts are NOT committed — they ship as the
/// `AdaWorldAPI/lance-graph` release `v0.1.0-cam96-data` (see `data/README.md`
/// for the fetch commands). Loaded at runtime from `data/` (override the
/// directory with `DEEPNSM_V2_DATA`). Copied verbatim from `bible_wave.rs`
/// (examples cannot import each other's helpers).
fn data_file(name: &str) -> Vec<u8> {
    let dir = std::env::var("DEEPNSM_V2_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data"));
    let path = dir.join(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing {} ({e}) — fetch the v0.1.0-cam96-data release assets per data/README.md",
            path.display()
        )
    })
}

const MIN_BASIN: usize = 6;
const SPLITS: &[f64] = &[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70];
const SHUFFLES: usize = 25;

/// One row of the input export: (verse, subject_id, predicate_id, object_id).
type Row = (u32, u16, u16, u16);

/// One candidate triple with its measured prefix support count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    s: u16,
    p: u16,
    o: u16,
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: pop_readout <spo.tsv> (from `bible_wave --export <spo.tsv>`)");
    let raw = std::fs::read_to_string(&path).expect("read spo.tsv");

    // ── load rows ────────────────────────────────────────────────────────
    let mut rows: Vec<Row> = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert!(
            f.len() == 7,
            "line {}: expected 7 tab-separated columns, got {} — is this a \
             `bible_wave --export` TSV?",
            lineno + 1,
            f.len()
        );
        let s: u16 = f[0]
            .parse()
            .unwrap_or_else(|e| panic!("line {}: bad subject_id {:?}: {e}", lineno + 1, f[0]));
        let p: u16 = f[2]
            .parse()
            .unwrap_or_else(|e| panic!("line {}: bad predicate_id {:?}: {e}", lineno + 1, f[2]));
        let o: u16 = f[4]
            .parse()
            .unwrap_or_else(|e| panic!("line {}: bad object_id {:?}: {e}", lineno + 1, f[4]));
        let v: u32 = f[6]
            .parse()
            .unwrap_or_else(|e| panic!("line {}: bad verse {:?}: {e}", lineno + 1, f[6]));
        rows.push((v, s, p, o));
    }
    assert!(!rows.is_empty(), "input TSV has no rows");
    let max_verse = rows.iter().map(|&(v, ..)| v).max().unwrap();
    println!(
        "LOAD  {} rows, max_verse={} ({})",
        rows.len(),
        max_verse,
        path
    );

    // ── load vocab + trained codebook (routing not needed; codes are) ─────
    let vocab_text = String::from_utf8(data_file("bible_vocab.txt")).expect("utf8 vocab");
    let mut vocab = deepnsm_v2::PaletteVocab::new();
    vocab.from_frequency_ranked(vocab_text.lines());
    let space = load_cam96_space(&data_file("cam96_codebook.bin")).expect("codebook artifact");
    let codes = load_cam96_codes(&data_file("cam96_codes.bin")).expect("codes artifact");
    assert_eq!(
        codes.len(),
        vocab.len(),
        "codes/vocab misaligned: {} codes, {} vocab words",
        codes.len(),
        vocab.len()
    );
    println!("LOAD  trained codebook: {} words, 12 axes", vocab.len());

    // ── the two MUL assessments A1 (baseline) vs A1B (contrasting) ────────
    // A1: a calibrated, in-flow, moderately-autonomous assessment.
    let assess_a = MulAssessment {
        trust: TrustQualia {
            value: 0.75,
            texture: TrustTexture::Calibrated,
        },
        dk_position: DkPosition::SlopeOfEnlightenment,
        homeostasis: Homeostasis {
            flow_state: FlowState::Flow,
            allostatic_load: 0.3,
        },
        complexity_mapped: true,
        free_will_modifier: 0.7,
    };
    // A1B: a starkly different reading — overconfident, anxious, low free will —
    // to check whether curiosity_gestalt's magnitude can actually re-order a
    // frontier under a genuinely different MUL state, not just rescale it.
    let assess_b = MulAssessment {
        trust: TrustQualia {
            value: 0.20,
            texture: TrustTexture::Overconfident,
        },
        dk_position: DkPosition::MountStupid,
        homeostasis: Homeostasis {
            flow_state: FlowState::Anxiety,
            allostatic_load: 0.9,
        },
        complexity_mapped: false,
        free_will_modifier: 0.1,
    };
    let signals = GraphSignals::default();

    // ── per-split results, aggregated at the end ───────────────────────────
    struct SplitResult {
        arms: Vec<ArmResult>,
        partial_real: f32,
        null_partial_mean: f32,
        null_partial_p95: f32,
        null_p10_ap_mean: f32,
        null_p10_ap_p95: f32,
        null_p10_a2_mean: f32,
        null_p10_a2_p95: f32,
    }
    struct ArmResult {
        name: &'static str,
        p10: f32,
        p25: f32,
        p100: f32,
        spearman: f32,
    }

    let mut all_splits: Vec<SplitResult> = Vec::new();

    for &frac in SPLITS {
        let cut = (max_verse as f64 * frac) as u32;

        let prefix: Vec<&Row> = rows.iter().filter(|&&(v, ..)| v < cut).collect();
        let suffix_set: HashSet<(u16, u16, u16)> = rows
            .iter()
            .filter(|&&(v, ..)| v >= cut)
            .map(|&(_, s, p, o)| (s, p, o))
            .collect();

        // objects[s] = distinct object ids seen for subject s in the prefix,
        // sorted ascending; edges[s] = the subject's (predicate, object) pairs
        // in prefix order (both needed by `basin_self_code`).
        use std::collections::HashMap;
        let mut objects: HashMap<u16, Vec<u16>> = HashMap::new();
        let mut edges: HashMap<u16, Vec<(u16, u16)>> = HashMap::new();
        // n(s,p,o) = number of prefix rows equal to that exact triple.
        let mut counts: HashMap<(u16, u16, u16), u32> = HashMap::new();
        for &&(_, s, p, o) in &prefix {
            edges.entry(s).or_default().push((p, o));
            let objs = objects.entry(s).or_default();
            if !objs.contains(&o) {
                objs.push(o);
            }
            *counts.entry((s, p, o)).or_insert(0) += 1;
        }
        for objs in objects.values_mut() {
            objs.sort_unstable();
        }

        // Candidate set: distinct (s,p,o) in prefix whose subject has
        // objects[s].len() >= MIN_BASIN. Deterministic ascending (s,p,o) order.
        let mut candidates: Vec<Candidate> = counts
            .keys()
            .filter(|&&(s, _, _)| objects.get(&s).map_or(0, Vec::len) >= MIN_BASIN)
            .map(|&(s, p, o)| Candidate { s, p, o })
            .collect();
        candidates.sort_unstable();

        // Basins: one per eligible subject, computed once and reused per
        // candidate of that subject.
        let mut basin_of: HashMap<u16, Option<deepnsm_v2::BasinCode>> = HashMap::new();
        for &s in objects.keys() {
            if objects[&s].len() < MIN_BASIN {
                continue;
            }
            let member_codes: Vec<Cam96> = objects[&s].iter().map(|&o| codes[o as usize]).collect();
            let b = basin_self_code(&space, s, &member_codes, &edges[&s]);
            basin_of.insert(s, b);
        }

        // Drop candidates whose subject's basin came back None (members empty
        // — cannot happen given the MIN_BASIN>=6 filter, but guard anyway).
        candidates.retain(|c| basin_of.get(&c.s).is_some_and(Option::is_some));

        assert!(
            !candidates.is_empty(),
            "split frac={frac}: zero eligible candidates — MIN_BASIN={MIN_BASIN} too strict \
             for this corpus, or the split point leaves too small a prefix"
        );

        let n_total = candidates.len();
        let labels: Vec<f32> = candidates
            .iter()
            .map(|c| {
                if suffix_set.contains(&(c.s, c.p, c.o)) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let n_pos = labels.iter().filter(|&&l| l > 0.5).count();
        let base_rate = n_pos as f32 / n_total as f32;
        assert!(
            base_rate > 0.0 && base_rate < 1.0,
            "split frac={frac}: base rate {base_rate} is degenerate (all-positive or \
             all-negative labels) — {n_pos}/{n_total}"
        );

        // ── per-candidate: n, pop, and the FrontierEdge-based scores ───────
        let freqs: Vec<f32> = candidates
            .iter()
            .map(|c| *counts.get(&(c.s, c.p, c.o)).unwrap() as f32)
            .collect();

        let pop_of =
            |c: &Candidate, basin_lookup: &HashMap<u16, Option<deepnsm_v2::BasinCode>>| -> f32 {
                let b = basin_lookup[&c.s].as_ref().expect("filtered to Some above");
                space.distance(&codes[c.o as usize], &b.self_code)
            };
        let pop: Vec<f32> = candidates.iter().map(|c| pop_of(c, &basin_of)).collect();
        let pop_min = pop.iter().cloned().fold(f32::INFINITY, f32::min);
        let pop_max = pop.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            pop_min < pop_max,
            "split frac={frac}: population readout is CONSTANT ({pop_min}) across all \
             {n_total} candidates — the basin/codebook signal is not varying"
        );

        let edges_for: Vec<FrontierEdge> = candidates
            .iter()
            .map(|c| FrontierEdge {
                source: vocab.word(c.s).unwrap_or("?").to_string(),
                target: vocab.word(c.o).unwrap_or("?").to_string(),
                label: vocab.word(c.p).unwrap_or("?").to_string(),
                truth: NarsTruth {
                    frequency: 1.0,
                    confidence: *counts.get(&(c.s, c.p, c.o)).unwrap() as f32
                        / (*counts.get(&(c.s, c.p, c.o)).unwrap() as f32 + 1.0),
                },
                query_count: 0,
                is_seed: false,
            })
            .collect();

        let a0: Vec<f32> = edges_for.iter().map(FrontierEdge::curiosity).collect();
        let a1: Vec<f32> = edges_for
            .iter()
            .map(|e| e.curiosity_gestalt(&assess_a, &signals).magnitude)
            .collect();
        let a1b: Vec<f32> = edges_for
            .iter()
            .map(|e| e.curiosity_gestalt(&assess_b, &signals).magnitude)
            .collect();
        let af: Vec<f32> = freqs.clone();
        let ap: Vec<f32> = pop.iter().map(|&p| -p).collect();
        let a2: Vec<f32> = combine_ranks(&a0, &ap);
        let a3: Vec<f32> = combine_ranks(&a1, &ap);

        let arm = |name: &'static str, scores: &[f32]| -> ArmResult {
            ArmResult {
                name,
                p10: precision_at(scores, &labels, 10),
                p25: precision_at(scores, &labels, 25),
                p100: precision_at(scores, &labels, 100),
                spearman: spearman(scores, &labels),
            }
        };
        let arms = vec![
            arm("A0  curiosity        ", &a0),
            arm("A1  gestalt(assessA) ", &a1),
            arm("A1B gestalt(assessB) ", &a1b),
            arm("AF  frequency        ", &af),
            arm("AP  -pop             ", &ap),
            arm("A2  A0+AP ranks      ", &a2),
            arm("A3  A1+AP ranks      ", &a3),
        ];

        // ── decisive statistic: partial Spearman(AP, label | frequency) ────
        let partial_real = partial_spearman(&ap, &labels, &freqs);

        // ── null: permute which word owns which code (SplitMix64/Fisher-
        // Yates), destroying the word↔meaning binding while preserving the
        // code multiset exactly. Recompute basins/pop/AP/A2/partial-Spearman
        // under the shuffle. ──
        let mut null_partials: Vec<f32> = Vec::with_capacity(SHUFFLES);
        let mut null_p10_ap: Vec<f32> = Vec::with_capacity(SHUFFLES);
        let mut null_p10_a2: Vec<f32> = Vec::with_capacity(SHUFFLES);
        let mut mean_abs_pop_vs_null_rho = 0.0f32;

        for r in 0..SHUFFLES {
            let seed = 0x9E37_79B9_7F4A_7C15u64 ^ (r as u64);
            let shuffled_codes = fisher_yates_shuffle(&codes, seed);

            // Recompute basins with the shuffled codes.
            let mut null_basin_of: HashMap<u16, Option<deepnsm_v2::BasinCode>> = HashMap::new();
            for &s in objects.keys() {
                if objects[&s].len() < MIN_BASIN {
                    continue;
                }
                let member_codes: Vec<Cam96> = objects[&s]
                    .iter()
                    .map(|&o| shuffled_codes[o as usize])
                    .collect();
                let b = basin_self_code(&space, s, &member_codes, &edges[&s]);
                null_basin_of.insert(s, b);
            }
            let null_pop: Vec<f32> = candidates
                .iter()
                .map(|c| {
                    let b = null_basin_of[&c.s]
                        .as_ref()
                        .expect("filtered to Some above");
                    space.distance(&shuffled_codes[c.o as usize], &b.self_code)
                })
                .collect();
            let null_ap: Vec<f32> = null_pop.iter().map(|&p| -p).collect();
            let null_a2: Vec<f32> = combine_ranks(&a0, &null_ap);

            mean_abs_pop_vs_null_rho += spearman(&pop, &null_pop).abs();
            null_partials.push(partial_spearman(&null_ap, &labels, &freqs));
            null_p10_ap.push(precision_at(&null_ap, &labels, 10));
            null_p10_a2.push(precision_at(&null_a2, &labels, 10));
        }
        mean_abs_pop_vs_null_rho /= SHUFFLES as f32;

        // ── guard 3: the null actually destroys the binding ────────────────
        if mean_abs_pop_vs_null_rho >= 0.5 {
            panic!(
                "GUARD 3 FAIL split frac={frac}: mean |spearman(pop_real, pop_null)| = \
                 {mean_abs_pop_vs_null_rho:.3} >= 0.5 — the null shuffle is not actually \
                 destroying the word↔code binding"
            );
        }
        println!(
            "  guard3 PASS  mean|spearman(pop_real,pop_null)| = {mean_abs_pop_vs_null_rho:.3} < 0.5"
        );

        let (null_partial_mean, null_partial_p95) = mean_and_p95(&null_partials);
        let (null_p10_ap_mean, null_p10_ap_p95) = mean_and_p95(&null_p10_ap);
        let (null_p10_a2_mean, null_p10_a2_p95) = mean_and_p95(&null_p10_a2);

        // ── report this split ───────────────────────────────────────────────
        println!(
            "\nSPLIT frac={frac:.2} cut=verse{cut}  candidates={n_total}  base_rate={base_rate:.3}"
        );
        println!(
            "  {:<22} {:>7} {:>7} {:>7} {:>9}",
            "arm", "p@10", "p@25", "p@100", "spearman"
        );
        for a in &arms {
            println!(
                "  {:<22} {:>7.3} {:>7.3} {:>7.3} {:>9.3}",
                a.name, a.p10, a.p25, a.p100, a.spearman
            );
        }
        println!(
            "  partial_spearman(AP,label|freq): real={partial_real:.3}  null_mean={null_partial_mean:.3}  null_p95={null_partial_p95:.3}"
        );

        all_splits.push(SplitResult {
            arms,
            partial_real,
            null_partial_mean,
            null_partial_p95,
            null_p10_ap_mean,
            null_p10_ap_p95,
            null_p10_a2_mean,
            null_p10_a2_p95,
        });
    }

    // ── guard 4: does curiosity_gestalt's magnitude ever re-order the base
    // curiosity ranking, or is it always a monotone (hence rank-inert) rescale?
    // Measured across ALL splits pooled, so the answer is not an artifact of
    // one split's small candidate count. ──
    {
        let mut pooled_a0: Vec<f32> = Vec::new();
        let mut pooled_a1: Vec<f32> = Vec::new();
        let mut pooled_a1b: Vec<f32> = Vec::new();
        for &frac in SPLITS {
            let cut = (max_verse as f64 * frac) as u32;
            let prefix: Vec<&Row> = rows.iter().filter(|&&(v, ..)| v < cut).collect();
            use std::collections::HashMap;
            let mut objects: HashMap<u16, Vec<u16>> = HashMap::new();
            let mut edges: HashMap<u16, Vec<(u16, u16)>> = HashMap::new();
            let mut counts: HashMap<(u16, u16, u16), u32> = HashMap::new();
            for &&(_, s, p, o) in &prefix {
                edges.entry(s).or_default().push((p, o));
                let objs = objects.entry(s).or_default();
                if !objs.contains(&o) {
                    objs.push(o);
                }
                *counts.entry((s, p, o)).or_insert(0) += 1;
            }
            let mut candidates: Vec<Candidate> = counts
                .keys()
                .filter(|&&(s, _, _)| objects.get(&s).map_or(0, Vec::len) >= MIN_BASIN)
                .map(|&(s, p, o)| Candidate { s, p, o })
                .collect();
            candidates.sort_unstable();
            for c in &candidates {
                let n = *counts.get(&(c.s, c.p, c.o)).unwrap() as f32;
                let e = FrontierEdge {
                    source: vocab.word(c.s).unwrap_or("?").to_string(),
                    target: vocab.word(c.o).unwrap_or("?").to_string(),
                    label: vocab.word(c.p).unwrap_or("?").to_string(),
                    truth: NarsTruth {
                        frequency: 1.0,
                        confidence: n / (n + 1.0),
                    },
                    query_count: 0,
                    is_seed: false,
                };
                pooled_a0.push(e.curiosity());
                pooled_a1.push(e.curiosity_gestalt(&assess_a, &signals).magnitude);
                pooled_a1b.push(e.curiosity_gestalt(&assess_b, &signals).magnitude);
            }
        }
        let rho_a0_a1 = spearman(&pooled_a0, &pooled_a1);
        let rho_a0_a1b = spearman(&pooled_a0, &pooled_a1b);
        println!(
            "\nGUARD 4  pooled over {} candidates across all splits",
            pooled_a0.len()
        );
        println!("  spearman(A0, A1[assessA])  = {rho_a0_a1:.6}");
        println!("  spearman(A0, A1B[assessB]) = {rho_a0_a1b:.6}");
        if rho_a0_a1 > 0.999 && rho_a0_a1b > 0.999 {
            println!(
                "FINDING qualia-inert: curiosity_gestalt magnitude is a per-graph scalar \
                 multiple of curiosity, so it cannot reorder a frontier (measured rho = {rho_a0_a1:.6} / {rho_a0_a1b:.6})"
            );
            assert!(
                rho_a0_a1 > 0.999 && rho_a0_a1b > 0.999,
                "guard 4 internal contradiction"
            );
        } else {
            println!(
                "FINDING qualia-reorders: curiosity_gestalt magnitude changes the frontier \
                 ordering relative to bare curiosity (measured rho = {rho_a0_a1:.6} / {rho_a0_a1b:.6})"
            );
        }
    }

    // ── aggregate across splits ─────────────────────────────────────────────
    println!(
        "\n════════════════════ AGGREGATE (mean over {} splits) ════════════════════",
        all_splits.len()
    );
    let n = all_splits.len() as f32;
    let arm_names: Vec<&str> = all_splits[0].arms.iter().map(|a| a.name).collect();
    for (i, name) in arm_names.iter().enumerate() {
        let mean_p10: f32 = all_splits.iter().map(|s| s.arms[i].p10).sum::<f32>() / n;
        let mean_p25: f32 = all_splits.iter().map(|s| s.arms[i].p25).sum::<f32>() / n;
        let mean_p100: f32 = all_splits.iter().map(|s| s.arms[i].p100).sum::<f32>() / n;
        println!("  {name:<22} mean p@10={mean_p10:.3}  p@25={mean_p25:.3}  p@100={mean_p100:.3}");
    }
    let mean_partial_real: f32 = all_splits.iter().map(|s| s.partial_real).sum::<f32>() / n;
    let mean_null_partial_p95: f32 = all_splits.iter().map(|s| s.null_partial_p95).sum::<f32>() / n;
    let mean_p10_a2: f32 = all_splits
        .iter()
        .map(|s| {
            s.arms
                .iter()
                .find(|a| a.name.starts_with("A2"))
                .unwrap()
                .p10
        })
        .sum::<f32>()
        / n;
    let mean_p10_a0: f32 = all_splits
        .iter()
        .map(|s| {
            s.arms
                .iter()
                .find(|a| a.name.starts_with("A0"))
                .unwrap()
                .p10
        })
        .sum::<f32>()
        / n;
    let mean_null_p10_a2_p95: f32 = all_splits.iter().map(|s| s.null_p10_a2_p95).sum::<f32>() / n;
    // The null MEANS are reported alongside the 95th percentiles: a null whose
    // mean sits near zero while its p95 is small is a null that genuinely
    // destroyed the binding, which is the claim guard 3 makes per split. Printed
    // rather than dropped -- a computed statistic that is never shown is a
    // measurement the reader cannot check.
    let mean_null_partial_mean: f32 =
        all_splits.iter().map(|s| s.null_partial_mean).sum::<f32>() / n;
    let mean_null_p10_a2_mean: f32 = all_splits.iter().map(|s| s.null_p10_a2_mean).sum::<f32>() / n;

    println!("\nmean real partial Spearman(AP,label|freq)     = {mean_partial_real:.3}");
    println!("mean null partial Spearman (mean over shuffles) = {mean_null_partial_mean:.3}");
    println!("mean null partial Spearman 95th percentile     = {mean_null_partial_p95:.3}");
    println!(
        "mean(A2 p@10) - mean(A0 p@10)                  = {:.3}",
        mean_p10_a2 - mean_p10_a0
    );
    println!("mean(A2 p@10)                                  = {mean_p10_a2:.3}");
    println!("mean null A2 p@10 (mean over shuffles)         = {mean_null_p10_a2_mean:.3}");
    println!("mean null A2 p@10 95th percentile              = {mean_null_p10_a2_p95:.3}");

    // The AP arm alone (population readout, no curiosity) against its own null.
    let mean_p10_ap: f32 = all_splits
        .iter()
        .map(|s| {
            s.arms
                .iter()
                .find(|a| a.name.starts_with("AP"))
                .unwrap()
                .p10
        })
        .sum::<f32>()
        / n;
    let mean_null_p10_ap_mean: f32 = all_splits.iter().map(|s| s.null_p10_ap_mean).sum::<f32>() / n;
    let mean_null_p10_ap_p95: f32 = all_splits.iter().map(|s| s.null_p10_ap_p95).sum::<f32>() / n;
    println!("mean(AP p@10) real / null-mean / null-p95      = {mean_p10_ap:.3} / {mean_null_p10_ap_mean:.3} / {mean_null_p10_ap_p95:.3}");

    let cond_a = mean_partial_real > mean_null_partial_p95 + 0.02;
    let cond_b = (mean_p10_a2 - mean_p10_a0) >= 0.05 && mean_p10_a2 > mean_null_p10_a2_p95;
    if cond_a && cond_b {
        println!("\nVERDICT PASS");
    } else {
        println!("\nVERDICT KILL");
    }
}

/// Precision@k: mean label over the k highest-scoring candidates. Ties broken
/// by candidate index ascending (deterministic).
fn precision_at(scores: &[f32], labels: &[f32], k: usize) -> f32 {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let k = k.min(idx.len());
    if k == 0 {
        return 0.0;
    }
    idx[..k].iter().map(|&i| labels[i]).sum::<f32>() / k as f32
}

/// Normalized ascending rank in `[0, 1]`, average ranks for ties. Rank 0 =
/// smallest value.
fn nrank(v: &[f32]) -> Vec<f32> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0f32; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        // average rank over the tie block [i, j]
        let avg_rank = (i + j) as f32 / 2.0;
        for &k in &idx[i..=j] {
            ranks[k] = avg_rank;
        }
        i = j + 1;
    }
    if n <= 1 {
        return ranks;
    }
    let denom = (n - 1) as f32;
    ranks.into_iter().map(|r| r / denom).collect()
}

/// Combine two score vectors as `0.5 * (nrank(x) + nrank(y))`.
fn combine_ranks(x: &[f32], y: &[f32]) -> Vec<f32> {
    let rx = nrank(x);
    let ry = nrank(y);
    rx.iter().zip(&ry).map(|(&a, &b)| 0.5 * (a + b)).collect()
}

/// Spearman rank correlation (Pearson over average ranks). `0.0` if either
/// side has zero variance.
fn spearman(x: &[f32], y: &[f32]) -> f32 {
    let rx = nrank(x);
    let ry = nrank(y);
    pearson(&rx, &ry)
}

fn pearson(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f32>() / n as f32;
    let my = y.iter().sum::<f32>() / n as f32;
    let mut cov = 0f32;
    let mut vx = 0f32;
    let mut vy = 0f32;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 0.0 || vy <= 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// Mean and 95th percentile (index `(0.95*(len-1)).round()` of the ascending
/// sort) of a value set.
fn mean_and_p95(v: &[f32]) -> (f32, f32) {
    let n = v.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = v.iter().sum::<f32>() / n as f32;
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((0.95 * (n - 1) as f32).round() as usize).min(n - 1);
    (mean, sorted[idx])
}

/// SplitMix64 PRNG — deterministic, no external rng crate, the workspace's
/// standard seed constant (see `bible_wave.rs::shuffle_null`).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Deterministic Fisher-Yates shuffle of the code vector (does not mutate the
/// original; the multiset of codes is preserved exactly, only the
/// word-id↔code binding is permuted).
fn fisher_yates_shuffle(codes: &[Cam96], seed: u64) -> Vec<Cam96> {
    let mut out = codes.to_vec();
    let mut rng = SplitMix64(seed);
    for i in (1..out.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        out.swap(i, j);
    }
    out
}
