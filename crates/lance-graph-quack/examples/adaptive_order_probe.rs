//! A1 — does conjunction term ORDER change how much work the survivor skip
//! avoids? The measurement the translation matrix demands before any port.
//!
//! # Why this probe exists, and why it comes before the code
//!
//! `duckdb-to-v3-translation-matrix-v1.md` row A1 is the one place DuckDB has
//! something V3 does not: `AdaptiveFilter` permutes a conjunction's terms at
//! runtime by measured selectivity, via an adjacent-transposition hill-climb
//! with a swap-likeliness decay. The row's falsifier is unusually blunt about
//! what to do with that:
//!
//! > **The falsifier is prior to the port, and it may kill the whole idea.**
//! > DuckDB's reordering pays because term k runs only on survivors of
//! > 1..k−1, so a selective term first *shrinks the input*. In V3 a predicate
//! > sweep costs the **full column** regardless of position — so **reordering
//! > saves nothing on generation**. It can only save by *avoidance*.
//! > **Measure:** what fraction of `Pred` ops are skippable by `under`, and
//! > does term order change that fraction? If order does not move the skip
//! > fraction, A1 is ELIMINATE and the machinery must not be ported.
//!
//! So this prints a number and draws no conclusion the number does not force.
//!
//! # What is counted, and what that models
//!
//! `MaskOp::Pred { under }` skips at 64-row WORD granularity: a word where the
//! gate has no survivor is not evaluated. Under a conjunction lowered in place,
//! term `i + 1` is gated on the accumulated result of terms `1..i`, so the
//! skippable fraction for term `i + 1` is the fraction of gate words that are
//! entirely zero.
//!
//! That is an exact model of WHICH words are skipped — the same quantity the
//! facade's `*_to_mask_under` decides on — and it is deliberately not a timing.
//! A wall-clock number here would be dominated by the fixture's cache
//! behaviour and would not answer the row's question, which is about skip
//! OPPORTUNITIES. The honest reading of the output is "how much work becomes
//! avoidable", not "how much time is saved".
//!
//! # Two granularities — and the rail is neither
//!
//! The 64-row word is the FACADE's unit — the `u64` the executor tests before
//! it loads a chunk. A 256-row block (four words, one 256-bit vector) is the
//! 2-nibble prefix cell of the OGAR tier tile — a coarser unit an executor may
//! skip in, with an identical result. The probe reports BOTH, because on
//! scattered survivors they disagree: a block with one live word is live.
//!
//! What a rail is NOT is a unit of either. Operator, 2026-09-15: *"64k sind 2
//! byte. 256:256 sind 2 byte für die exakte SoA in a given table — needle in a
//! haystack × table. Für Maske über 64k als Fläche bräuchte es entsprechend
//! mehr."* A `u8:u8` rail is the exact row ADDRESS of a 64k table: 256 × 256 =
//! 65 536 = this fixture's `N`, every value a row and every row a value — that
//! is the "no remainder". A mask over the same 64k is a different, larger
//! object (bitpacked: 1 024 words = 256 blocks, tiling with no remainder in
//! either unit), and the rail says nothing about which unit it is skipped in.
//! The clustered regime's prefix is `/48` — the hi byte of the row index, one
//! representative radius. The radius itself is STEPLESS: the operator's V3
//! variant (2026-09-15) masks a unit by its own facet × its distance from
//! root, 0–96 bits, exact and stepless, and the sweep at the end walks every
//! d from 40 to 56. An earlier run used `/50`; its figures (99.90 % =
//! 1023/1024) are the point where the word-skip saturates.
//!
//! Run: `cargo run -p lance-graph-quack --example adaptive_order_probe --release`

use lance_graph_mask_risc::{MaskOp, Operand};
use lance_graph_quack::{Agg, Cmp, Col, Filter, Query};

const VALS: Col = Col(0);
const CLASS: Col = Col(1);
const ADDR: Col = Col(2);

const N: usize = 1 << 16;
const WORDS: usize = N / 64;
/// Rows per block: the 2-nibble prefix cell of the tier tile — 256 rows, four
/// 64-row words, one 256-bit vector. A coarser skip unit than the word; not a
/// property of the rail, which is a row address (see the module doc).
const BLOCK_ROWS: usize = 256;
const BLOCK_WORDS: usize = BLOCK_ROWS / 64;
const BLOCKS: usize = N / BLOCK_ROWS;
const _: () = assert!(
    N == 256 * 256 && BLOCKS * BLOCK_WORDS == WORDS,
    "a 2-byte rail addresses exactly N rows, and N is whole words and whole blocks — no remainder"
);

/// One conjunct: a label, the filter, and the rows it selects.
struct Term {
    label: &'static str,
    filter: Filter,
    mask: Vec<u64>,
    selectivity: f64,
}

/// Knuth's LCG constants. Deterministic on purpose and seeded from a
/// constant: this probe's numbers are quoted in
/// `duckdb-to-v3-translation-matrix-v1.md` §8a and in `Query::and_by_skip`'s
/// doc table, so a run that did not reproduce them would silently invalidate
/// a recorded measurement rather than fail. The `>> 11` drops the low bits,
/// which are the weakest in an LCG.
fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

/// Words of `mask` that are entirely zero — exactly the words a `Pred` gated
/// on this mask does not evaluate.
fn dead_words(mask: &[u64]) -> usize {
    mask.iter().filter(|w| **w == 0).count()
}

/// Blocks of `mask` (four words each) that are entirely zero — a coarser skip
/// unit an executor may use. A block with one live word is LIVE, not
/// three-quarters dead, so block counting is stricter than word counting on
/// scattered survivors.
fn dead_blocks(mask: &[u64]) -> usize {
    let (blocks, rest) = mask.as_chunks::<BLOCK_WORDS>();
    assert!(rest.is_empty(), "a 64k mask is whole blocks — no remainder");
    blocks.iter().filter(|c| c.iter().all(|w| *w == 0)).count()
}

fn and_into(acc: &mut [u64], other: &[u64]) {
    for (a, b) in acc.iter_mut().zip(other) {
        *a &= *b;
    }
}

fn popcount(mask: &[u64]) -> u32 {
    mask.iter().map(|w| w.count_ones()).sum()
}

/// Total words and blocks SKIPPED across a conjunction evaluated in this order.
///
/// Term 0 is ungated and skips nothing — it is the seed. Term `i` for `i > 0`
/// is gated on the accumulation of `0..i`, and skips that accumulation's dead
/// words (the executor's unit) and dead blocks (the rail's unit).
fn skipped(terms: &[&Term]) -> (usize, usize, u32) {
    let masks: Vec<&[u64]> = terms.iter().map(|t| t.mask.as_slice()).collect();
    skipped_masks(&masks)
}

/// The same accounting over bare masks, for the radius sweep below.
fn skipped_masks(masks: &[&[u64]]) -> (usize, usize, u32) {
    let mut acc = vec![u64::MAX; WORDS];
    and_into(&mut acc, masks[0]);
    let (mut words, mut blocks) = (0usize, 0usize);
    for m in &masks[1..] {
        words += dead_words(&acc);
        blocks += dead_blocks(&acc);
        and_into(&mut acc, m);
    }
    (words, blocks, popcount(&acc))
}

/// EXHAUSTIVE, not sampled — which is what lets the probe report a true
/// best and worst order rather than the best and worst it happened to try.
/// Factorial in the term count, so it is only viable because the fixtures
/// are deliberately small (4 terms = 24 orders).
fn permutations<'a>(items: &[&'a Term]) -> Vec<Vec<&'a Term>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, head);
            out.push(p);
        }
    }
    out
}

/// One regime: a set of conjuncts, and the spread its orderings produce.
struct Scenario {
    name: &'static str,
    terms: Vec<Term>,
}

fn main() {
    let mut seed = 0x51ED_C0DEu64;
    let vals: Vec<i32> = (0..N)
        .map(|_| (lcg(&mut seed) % 2000) as i32 - 1000)
        .collect();
    let classes: Vec<u32> = (0..N).map(|_| (lcg(&mut seed) % 64) as u32).collect();
    // An ADDRESS-ORDERED lane, the shape the V3 substrate actually has.
    let addr: Vec<u64> = (0..N).map(|i| (i as u64) << 8).collect();

    let build = |label: &'static str, filter: Filter, keep: &dyn Fn(usize) -> bool| {
        let mut mask = vec![0u64; WORDS];
        let mut live = 0usize;
        for r in 0..N {
            if keep(r) {
                mask[r / 64] |= 1u64 << (r % 64);
                live += 1;
            }
        }
        Term {
            label,
            filter,
            mask,
            selectivity: live as f64 / N as f64,
        }
    };

    // THREE regimes, not one. A single fixture would answer the row's question
    // only for that fixture, and the two ends behave differently for a reason
    // that matters: a conjunction whose accumulator collapses to nearly empty
    // makes almost every later word skippable regardless of order, while a
    // permissive one leaves little to skip at all. The interesting number is
    // what order buys in between.
    //
    // The first draft of this probe had ONE regime and it was degenerate:
    // `v < 500` and `v > 900` are disjoint, so the conjunction selected ZERO
    // rows and the "best" ordering was just measuring how fast the accumulator
    // died. Every scenario below is asserted to select a proper subset.
    let scenarios = vec![
        Scenario {
            name: "selective  (a long tail of survivors)",
            terms: vec![
                build(
                    "v > -900   (0.95)",
                    Filter::cmp(VALS, Cmp::GtI32(-900)),
                    &|r| vals[r] > -900,
                ),
                build(
                    "v < 500    (0.75)",
                    Filter::cmp(VALS, Cmp::LtI32(500)),
                    &|r| vals[r] < 500,
                ),
                build(
                    "class != 0 (0.98)",
                    Filter::cmp(CLASS, Cmp::NeU32(0)),
                    &|r| classes[r] != 0,
                ),
                build(
                    "v < -850   (selective)",
                    Filter::cmp(VALS, Cmp::LtI32(-850)),
                    &|r| vals[r] < -850,
                ),
                build(
                    "class == 7 (very selective)",
                    Filter::cmp(CLASS, Cmp::EqU32(7)),
                    &|r| classes[r] == 7,
                ),
            ],
        },
        Scenario {
            name: "moderate",
            terms: vec![
                build(
                    "v > -900   (0.95)",
                    Filter::cmp(VALS, Cmp::GtI32(-900)),
                    &|r| vals[r] > -900,
                ),
                build(
                    "v < 500    (0.75)",
                    Filter::cmp(VALS, Cmp::LtI32(500)),
                    &|r| vals[r] < 500,
                ),
                build(
                    "class != 0 (0.98)",
                    Filter::cmp(CLASS, Cmp::NeU32(0)),
                    &|r| classes[r] != 0,
                ),
                build(
                    "v > -400   (0.70)",
                    Filter::cmp(VALS, Cmp::GtI32(-400)),
                    &|r| vals[r] > -400,
                ),
                build(
                    "class < 32 (0.50)",
                    Filter::cmp(
                        CLASS,
                        Cmp::MatchU32 {
                            pattern: 0,
                            care: 32,
                        },
                    ),
                    &|r| classes[r] & 32 == 0,
                ),
            ],
        },
        Scenario {
            name: "permissive (little to skip at all)",
            terms: vec![
                build(
                    "v > -990   (0.995)",
                    Filter::cmp(VALS, Cmp::GtI32(-990)),
                    &|r| vals[r] > -990,
                ),
                build(
                    "v < 990    (0.995)",
                    Filter::cmp(VALS, Cmp::LtI32(990)),
                    &|r| vals[r] < 990,
                ),
                build(
                    "class != 0 (0.98)",
                    Filter::cmp(CLASS, Cmp::NeU32(0)),
                    &|r| classes[r] != 0,
                ),
                build(
                    "class != 1 (0.98)",
                    Filter::cmp(CLASS, Cmp::NeU32(1)),
                    &|r| classes[r] != 1,
                ),
                build(
                    "class != 2 (0.98)",
                    Filter::cmp(CLASS, Cmp::NeU32(2)),
                    &|r| classes[r] != 2,
                ),
            ],
        },
        Scenario {
            name: "clustered (one conjunct is an ADDRESS PREFIX)",
            terms: vec![
                // /48 on `i << 8` pins the row index's hi byte — one 256-row
                // block, a representative radius. Any d is a legal radius (the
                // sweep below walks 40..=56); /50 is where the word-skip
                // saturates.
                build(
                    "addr prefix /48  (hi byte: one block)",
                    Filter::prefix_u64(ADDR, addr[N / 4], 48),
                    &|r| {
                        let care = u64::MAX << (64 - 48);
                        (addr[r] ^ addr[N / 4]) & care == 0
                    },
                ),
                build(
                    "v > -900   (0.95)",
                    Filter::cmp(VALS, Cmp::GtI32(-900)),
                    &|r| vals[r] > -900,
                ),
                build(
                    "class != 0 (0.98)",
                    Filter::cmp(CLASS, Cmp::NeU32(0)),
                    &|r| classes[r] != 0,
                ),
                build(
                    "v < 500    (0.75)",
                    Filter::cmp(VALS, Cmp::LtI32(500)),
                    &|r| vals[r] < 500,
                ),
                build(
                    "v > -400   (0.70)",
                    Filter::cmp(VALS, Cmp::GtI32(-400)),
                    &|r| vals[r] > -400,
                ),
            ],
        },
    ];

    println!(
        "N = {N} rows = {BLOCKS} blocks x {BLOCK_ROWS} = {WORDS} words x 64, 5 conjuncts per scenario"
    );
    println!(
        "\nskipped = 64-row words (the executor's unit) and 256-row blocks (the tile's\n         2-nibble cell) a gated `Pred` does not evaluate, summed over the {} word /\n         {} block gated positions of one ordering (term 0 is the ungated seed).\n",
        4 * WORDS,
        4 * BLOCKS
    );

    for sc in &scenarios {
        let refs: Vec<&Term> = sc.terms.iter().collect();
        let gated_words = (sc.terms.len() - 1) * WORDS;
        let gated_blocks = (sc.terms.len() - 1) * BLOCKS;

        // The model is tied to the SHIPPED lowering rather than assumed: the
        // conjunction is really lowered, and every term after the first must
        // come back gated on a scratch slot. If that stops being true the
        // probe is measuring something the executor does not do.
        let program = lance_graph_quack::lower(&Query {
            filter: Filter::and(refs.iter().map(|t| t.filter.clone())),
            agg: Agg::Count,
        })
        .expect("lowers");
        let accumulator_gated = program
            .ops
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    MaskOp::Pred {
                        under: Some(Operand::Scratch(_)),
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            accumulator_gated,
            sc.terms.len() - 1,
            "{}: the lowering gated {accumulator_gated} of {} later terms on the \
             accumulator; this probe measures a skip the executor would not perform",
            sc.name,
            sc.terms.len() - 1
        );

        let perms = permutations(&refs);
        // (words, blocks, survivors, order)
        let mut results: Vec<(usize, usize, u32, Vec<&'static str>)> = perms
            .iter()
            .map(|p| {
                let (words, blocks, count) = skipped(p);
                (words, blocks, count, p.iter().map(|t| t.label).collect())
            })
            .collect();

        let answer = results[0].2;
        assert!(
            results.iter().all(|r| r.2 == answer),
            "{}: orderings disagree on the count; the probe is measuring a defect",
            sc.name
        );
        assert!(
            answer > 0 && (answer as usize) < N,
            "{}: selects {answer}/{N} — a degenerate conjunction proves nothing \
             about order",
            sc.name
        );

        results.sort_by_key(|r| r.0);
        let worst = &results[0];
        let best = results.last().expect("non-empty");
        let (written_w, written_b, _) = skipped(&refs);
        let pct = |s: usize| 100.0 * s as f64 / gated_words as f64;
        let pct_b = |s: usize| 100.0 * s as f64 / gated_blocks as f64;

        println!(
            "=== {} — {answer} survivors ({:.3}%)",
            sc.name,
            100.0 * answer as f64 / N as f64
        );
        for t in &sc.terms {
            println!("      {:<30} sel {:.4}", t.label, t.selectivity);
        }
        println!(
            "      {:<12} {:>7} {:>8}   {:>7} {:>8}",
            "ordering", "words", "of gated", "blocks", "of gated"
        );
        for (name, w, b) in [
            ("as written", written_w, written_b),
            ("worst", worst.0, worst.1),
            ("best", best.0, best.1),
        ] {
            println!(
                "      {name:<12} {w:>7} {:>7.2}%   {b:>7} {:>7.2}%",
                pct(w),
                pct_b(b)
            );
        }
        // 0/0 is not an infinite ratio, it is NO SPREAD — the `moderate` and
        // `permissive` regimes skip nothing in ANY order, and printing `infx`
        // beside `spread 0.00` made the line contradict itself. Those two rows
        // are cited in `Query::and_by_skip`'s doc table, so the print is the
        // evidence a reader sees.
        match (best.0, worst.0) {
            (0, 0) => println!("      spread 0.00 percentage points, no skip in any order"),
            (_, 0) => println!(
                "      spread {:.2} percentage points, best/worst unbounded (worst skips 0)",
                pct(best.0) - pct(worst.0)
            ),
            (_, w) => println!(
                "      spread {:.2} percentage points, best/worst {:.2}x",
                pct(best.0) - pct(worst.0),
                best.0 as f64 / w as f64
            ),
        }
        println!("      best order: {}", best.3.join("  <  "));
        // Term 0 walked through every index with the rest in written order —
        // the ramp `Filter::and_by_skip`'s doc quotes. Printed, so the quoted
        // figure is a re-runnable measurement rather than a one-off
        // instrumentation.
        let ramp: Vec<(usize, usize)> = (0..refs.len())
            .map(|p| {
                let mut order: Vec<&Term> = refs[1..].to_vec();
                order.insert(p, refs[0]);
                let (w, b, _) = skipped(&order);
                (w, b)
            })
            .collect();
        println!(
            "      term 0 at index 0..={}: words {:?}  blocks {:?}",
            refs.len() - 1,
            ramp.iter().map(|r| r.0).collect::<Vec<_>>(),
            ramp.iter().map(|r| r.1).collect::<Vec<_>>()
        );
        // The rail's unit, ranked on its own: the best-by-blocks ordering can
        // differ from the best-by-words one, and the spread in blocks is the
        // number a rail-addressed skip can actually realise.
        let (wb, bb) = (
            results.iter().map(|r| r.1).min().expect("non-empty"),
            results.iter().map(|r| r.1).max().expect("non-empty"),
        );
        println!(
            "      blocks:      worst {wb} ({:.2}%)  best {bb} ({:.2}%)  spread {:.2} percentage points",
            pct_b(wb),
            pct_b(bb),
            pct_b(bb) - pct_b(wb)
        );
        println!();
    }

    // The operator's V3 masking variant (2026-09-15): a unit masks ITSELF by its
    // distance from root — `care = the first d bits`, d ∈ 0..=96 over the
    // facet; here 0..=56 over this lane's 16 row bits (bits 8..=23 of `i << 8`,
    // so d = 40 is every row and d = 56 is one). Any d is a legal radius — the
    // selection is STEPLESS; nibble boundaries are where the codebook's
    // centroid cells sit, which is a fact about meaning, not about the mask.
    // The sweep leads the clustered conjunction with each radius and reports
    // what it selects and skips: the ceiling family §8a's clustered row is one
    // point of, in both units — and the point where the units part ways.
    let clustered = scenarios
        .iter()
        .find(|s| s.name.starts_with("clustered"))
        .expect("the clustered scenario exists");
    let others: Vec<&[u64]> = clustered.terms[1..]
        .iter()
        .map(|t| t.mask.as_slice())
        .collect();
    let base = addr[N / 4];
    println!(
        "=== stepless radius d on the address lane, prefix first, then the clustered conjuncts"
    );
    println!(
        "      {:>3} {:>7} {:>7} {:>8}   {:>7} {:>8}",
        "d", "rows", "words", "of gated", "blocks", "of gated"
    );
    for d in 40..=56u32 {
        let care = u64::MAX << (64 - d);
        let mut prefix = vec![0u64; WORDS];
        let mut rows = 0usize;
        for r in 0..N {
            if (addr[r] ^ base) & care == 0 {
                prefix[r / 64] |= 1u64 << (r % 64);
                rows += 1;
            }
        }
        assert_eq!(
            rows,
            1usize << (56 - d),
            "a radius of d bits selects 2^(56-d) rows"
        );
        let mut masks: Vec<&[u64]> = vec![prefix.as_slice()];
        masks.extend(others.iter().copied());
        let (w, b, _) = skipped_masks(&masks);
        println!(
            "      {d:>3} {rows:>7} {w:>7} {:>7.2}%   {b:>7} {:>7.2}%",
            100.0 * w as f64 / (4 * WORDS) as f64,
            100.0 * b as f64 / (4 * BLOCKS) as f64
        );
    }
    println!();

    println!(
        "A1's falsifier: if term order does not move the skipped fraction, the row is\n\
         ELIMINATE and DuckDB's hill-climb must not be ported. These are counts of\n\
         AVOIDABLE word- and block-evaluations, not timings."
    );
}
