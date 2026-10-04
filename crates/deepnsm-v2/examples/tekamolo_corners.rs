//! `tekamolo_corners` — how often does the RIGHT-corner TEKAMOLO read overturn
//! what the LEFT-corner read commits, per corpus?
//!
//! Pre-registered prediction: the overturn rate is higher on biblical German
//! (Luther 1545, Elberfelder 1905 — Semitic-shaped order inherited through Koine
//! Greek) than on edited German (UD German-GSD test text).
//!
//! Per clause the program reads the same tokens twice with
//! [`ReadParams::LEFT_CORNER`] and [`ReadParams::RIGHT_CORNER`]. For each of the
//! four lanes: a "left commitment" is a lane the left read committed; an
//! "overturn" is a left commitment whose right-read address differs (a different
//! address, or an abstention); "right-only" is a lane the left read left empty
//! and the right read committed.
//!
//! Inputs (positional):
//! 1. `TEKAMOLO.tsv` — the heuristics, passed straight to `GrammarHeuristics::parse`.
//! 2. `GSD.conllu` — UD German-GSD test; only the FORM column is read, never a
//!    gold column.
//! 3. one or more Bible JSON files (`abbreviation`, `books[].chapters[].verses[].text`).
//!
//! Clauses: every text is split ONLY at `, ; : . ! ? ( )` by one shared splitter;
//! tokens are the whitespace split of the clause.
//!
//! KILL rule (pre-registered): the line `verdict:` is `KILL` if the first Bible
//! file's overturn rate is not above GSD's beyond the bootstrap 95% CI (Bible CI
//! lower bound not greater than GSD CI upper bound), else `PASS`.
//!
//! Matched arm (added after the first run, so post hoc — reported, never the
//! verdict): the two presets differ in TWO knobs, the commit point and whether
//! an ambiguous form (`da`) may win. The matched arm reads with the right
//! preset's fan-out, margin and ambiguity admission and only the commit point
//! moved to the left corner, so its overturn rate isolates "where it commits".
//!
//! ```text
//! cargo run --release --example tekamolo_corners -- TEKAMOLO.tsv de_gsd-ud-test.conllu bible_luther1545.json bible_elberfelder1905.json
//! ```

use deepnsm_v2::tekamolo::{read_clause, Commit, GrammarHeuristics, ReadParams};
use std::collections::HashMap;

const LANE_NAMES: [&str; 4] = ["Temporal", "Kausal", "Modal", "Lokal"];
const BOOTSTRAP_ROUNDS: usize = 1000;

/// `(rate, ci_lo, ci_hi)`.
type Rate = (f64, f64, f64);

/// The right preset with only the commit point moved to the left corner.
const LEFT_MATCHED: ReadParams = ReadParams {
    commit: Commit::LeftCorner,
    ..ReadParams::RIGHT_CORNER
};

/// Per-clause counts, kept so clauses can be resampled.
#[derive(Clone, Copy, Default)]
struct ClauseStat {
    left_commits: usize,
    overturns: usize,
    right_only: usize,
    has_hypothesis: bool,
    right_abstained: usize,
    right_ambiguous_wins: usize,
    /// Matched arm: commitments and overturns of [`LEFT_MATCHED`].
    matched_commits: usize,
    matched_overturns: usize,
}

/// One corpus's accumulated measurement.
struct Corpus {
    label: String,
    stats: Vec<ClauseStat>,
    /// Form that opened an overturned left commitment -> count.
    forms: HashMap<String, usize>,
}

impl Corpus {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            stats: Vec::new(),
            forms: HashMap::new(),
        }
    }
}

/// The normalisation `read_clause` applies to a token before lookup.
fn normalise(tok: &str) -> String {
    tok.chars()
        .filter(|c| c.is_alphabetic())
        .collect::<String>()
        .to_lowercase()
}

/// The ONE clause splitter: split only at `, ; : . ! ? ( )`, then on
/// whitespace; empty clauses and empty tokens are dropped.
fn for_each_clause<'a>(text: &'a str, mut f: impl FnMut(&[&'a str])) {
    for clause in text.split([',', ';', ':', '.', '!', '?', '(', ')']) {
        let tokens: Vec<&str> = clause.split_whitespace().collect();
        if !tokens.is_empty() {
            f(&tokens);
        }
    }
}

/// Read one text, adding its clauses to `corpus`.
fn measure_text(h: &GrammarHeuristics, text: &str, corpus: &mut Corpus) {
    for_each_clause(text, |tokens| {
        let l = read_clause(h, tokens, ReadParams::LEFT_CORNER);
        let r = read_clause(h, tokens, ReadParams::RIGHT_CORNER);
        let mut st = ClauseStat {
            has_hypothesis: r.opened > 0,
            right_abstained: r.abstained,
            right_ambiguous_wins: r.ambiguous_wins,
            ..ClauseStat::default()
        };
        for lane in 0..4 {
            match (l.lanes[lane], r.lanes[lane]) {
                (Some(la), ra) => {
                    st.left_commits += 1;
                    if ra != Some(la) {
                        st.overturns += 1;
                        // The left winner is the FIRST token with an entry for
                        // this lane.
                        let opener = tokens.iter().map(|t| normalise(t)).find(|w| {
                            !w.is_empty() && h.lookup(w).iter().any(|e| e.0 as usize == lane)
                        });
                        if let Some(w) = opener {
                            *corpus.forms.entry(w).or_default() += 1;
                        }
                    }
                }
                (None, Some(_)) => st.right_only += 1,
                (None, None) => {}
            }
        }
        let m = read_clause(h, tokens, LEFT_MATCHED);
        for lane in 0..4 {
            if let Some(ma) = m.lanes[lane] {
                st.matched_commits += 1;
                st.matched_overturns += usize::from(r.lanes[lane] != Some(ma));
            }
        }
        corpus.stats.push(st);
    });
}

/// GSD: rebuild each sentence from FORM tokens (col 2 only), then run it
/// through the shared clause splitter.
fn measure_gsd(h: &GrammarHeuristics, path: &str) -> Corpus {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut corpus = Corpus::new("GSD");
    let mut forms: Vec<&str> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !forms.is_empty() {
                measure_text(h, &forms.join(" "), &mut corpus);
                forms.clear();
            }
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 2 || cols[0].contains(['-', '.']) {
            continue;
        }
        forms.push(cols[1]);
    }
    if !forms.is_empty() {
        measure_text(h, &forms.join(" "), &mut corpus);
    }
    corpus
}

/// Bible JSON: `abbreviation` + `books[].chapters[].verses[].text`.
fn measure_bible(h: &GrammarHeuristics, path: &str) -> Corpus {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let json: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{path}: invalid JSON: {e}"));
    let label = json
        .get("abbreviation")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| panic!("{path}: missing string \"abbreviation\""));
    let mut corpus = Corpus::new(label);
    let books = json
        .get("books")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("{path}: missing array \"books\""));
    for book in books {
        let chapters = book.get("chapters").and_then(serde_json::Value::as_array);
        for chapter in chapters.into_iter().flatten() {
            let verses = chapter.get("verses").and_then(serde_json::Value::as_array);
            for verse in verses.into_iter().flatten() {
                if let Some(t) = verse.get("text").and_then(serde_json::Value::as_str) {
                    measure_text(h, t, &mut corpus);
                }
            }
        }
    }
    corpus
}

/// xorshift64, seeded 0x9E3779B97F4A7C15.
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn rate(overturns: usize, commits: usize) -> f64 {
    if commits == 0 {
        0.0
    } else {
        overturns as f64 / commits as f64
    }
}

/// 95% bootstrap CI of the overturn rate, resampling clauses with replacement.
fn bootstrap_ci(stats: &[ClauseStat], pick: fn(&ClauseStat) -> (usize, usize)) -> (f64, f64) {
    if stats.is_empty() {
        return (0.0, 0.0);
    }
    let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
    let n = stats.len() as u64;
    let mut rates = Vec::with_capacity(BOOTSTRAP_ROUNDS);
    for _ in 0..BOOTSTRAP_ROUNDS {
        let (mut o, mut c) = (0usize, 0usize);
        for _ in 0..stats.len() {
            let (so, sc) = pick(&stats[(rng.next() % n) as usize]);
            o += so;
            c += sc;
        }
        rates.push(rate(o, c));
    }
    rates.sort_by(f64::total_cmp);
    (
        rates[BOOTSTRAP_ROUNDS * 25 / 1000],
        rates[BOOTSTRAP_ROUNDS * 975 / 1000],
    )
}

/// Print one corpus block; returns `(rate, ci_lo, ci_hi)` for the
/// pre-registered arm and for the matched arm.
fn report(c: &Corpus) -> (Rate, Rate) {
    let sum = |f: fn(&ClauseStat) -> usize| c.stats.iter().map(f).sum::<usize>();
    let commits = sum(|s| s.left_commits);
    let overturns = sum(|s| s.overturns);
    let r = rate(overturns, commits);
    let (lo, hi) = bootstrap_ci(&c.stats, |s| (s.overturns, s.left_commits));
    let m_commits = sum(|s| s.matched_commits);
    let m_overturns = sum(|s| s.matched_overturns);
    let mr = rate(m_overturns, m_commits);
    let (mlo, mhi) = bootstrap_ci(&c.stats, |s| (s.matched_overturns, s.matched_commits));
    println!("== {} ==", c.label);
    println!("  clauses: {}", c.stats.len());
    println!(
        "  clauses with >=1 hypothesis (right read): {}",
        c.stats.iter().filter(|s| s.has_hypothesis).count()
    );
    println!("  left commitments: {commits}");
    println!("  overturns: {overturns}");
    println!("  overturn rate: {r:.4}  (95% bootstrap CI {lo:.4} .. {hi:.4})");
    println!(
        "  matched arm (post hoc): {m_overturns}/{m_commits} = {mr:.4}  (95% CI {mlo:.4} .. {mhi:.4})"
    );
    println!("  right-only lanes: {}", sum(|s| s.right_only));
    println!("  right abstained total: {}", sum(|s| s.right_abstained));
    println!(
        "  ambiguous_wins total (right read): {}",
        sum(|s| s.right_ambiguous_wins)
    );
    let mut top: Vec<(&String, &usize)> = c.forms.iter().collect();
    top.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    println!("  top forms opening an overturned left commitment:");
    for (form, n) in top.into_iter().take(10) {
        println!("    {form:12} {n}");
    }
    ((r, lo, hi), (mr, mlo, mhi))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: tekamolo_corners TEKAMOLO.tsv GSD.conllu BIBLE.json [BIBLE.json ...]");
        std::process::exit(2);
    }
    let tsv = std::fs::read_to_string(&args[0]).unwrap_or_else(|e| panic!("{}: {e}", args[0]));
    let h = GrammarHeuristics::parse(&tsv);
    let (forms, rows) = h.sizes();
    println!(
        "heuristics: {forms} forms, {rows} rows, {} ambiguous; lanes {}",
        h.ambiguous(),
        LANE_NAMES.join("/")
    );

    let gsd = measure_gsd(&h, &args[1]);
    let bibles: Vec<Corpus> = args[2..].iter().map(|p| measure_bible(&h, p)).collect();

    let ((gsd_rate, _gsd_lo, gsd_hi), (gsd_m, gsd_mlo, gsd_mhi)) = report(&gsd);
    let mut first: Option<(Rate, Rate)> = None;
    for b in &bibles {
        let r = report(b);
        first.get_or_insert(r);
    }
    let ((b_rate, b_lo, _b_hi), (b_m, b_mlo, b_mhi)) =
        first.unwrap_or_else(|| panic!("no Bible corpus given"));
    println!(
        "comparison: {} rate {b_rate:.4} (CI lower {b_lo:.4}) vs GSD rate {gsd_rate:.4} (CI upper {gsd_hi:.4})",
        bibles[0].label
    );
    println!(
        "matched arm (post hoc): {} {b_m:.4} (CI {b_mlo:.4} .. {b_mhi:.4}) vs GSD {gsd_m:.4} (CI {gsd_mlo:.4} .. {gsd_mhi:.4})",
        bibles[0].label
    );
    println!("verdict: {}", if b_lo > gsd_hi { "PASS" } else { "KILL" });
}
