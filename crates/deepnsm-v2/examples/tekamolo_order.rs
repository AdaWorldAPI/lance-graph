//! `tekamolo_order` — do edited German books keep the TEKAMOLO order
//! (Temporal < Kausal < Modal < Lokal) more often than the Bible translations?
//!
//! The reading is by **position only**, on raw text:
//! - a clause is the span between punctuation marks;
//! - in each clause, the first occurrence of each lane's cue word gives that
//!   lane's position;
//! - every pair of two different lanes counts as **in order** when the
//!   earlier lane comes first in Te < Ka < Mo < Lo.
//!
//! Cue words are UNAMBIGUOUS lane markers only: no prepositions (*in*, *an*
//! read both time and place), no modal particles (*wohl*), no words with a
//! second common reading (*da*, *so*, *noch*).
//!
//! Pre-registered prediction (fixed before the run): every Bible translation
//! has a lower in-order rate than the pooled edited literature, with 95 %
//! Wilson intervals that do not overlap.
//!
//! Inputs: Project Gutenberg `.txt`, getbible-style `.json`, UD `.conllu`.
//!
//! ```text
//! cargo run --release --example tekamolo_order -- --lit A.txt B.txt -- --bible X.json Y.json
//! ```

const TE: [&str; 17] = [
    "heute",
    "gestern",
    "morgen",
    "damals",
    "jetzt",
    "bald",
    "später",
    "früher",
    "immer",
    "nie",
    "niemals",
    "oft",
    "manchmal",
    "bereits",
    "neulich",
    "zuvor",
    "anschließend",
];
const KA: [&str; 10] = [
    "deshalb", "deswegen", "darum", "daher", "wegen", "aufgrund", "trotz", "infolge", "dank",
    "folglich",
];
const MO: [&str; 16] = [
    "gern",
    "gerne",
    "schnell",
    "langsam",
    "plötzlich",
    "allmählich",
    "sorgfältig",
    "gemeinsam",
    "leise",
    "laut",
    "ruhig",
    "heimlich",
    "mühsam",
    "eilig",
    "vorsichtig",
    "zusammen",
];
const LO: [&str; 17] = [
    "hier", "dort", "drinnen", "draußen", "oben", "unten", "vorn", "vorne", "hinten", "links",
    "rechts", "überall", "nirgends", "daheim", "dorthin", "hierher", "dahin",
];

fn lane(w: &str) -> Option<usize> {
    [&TE[..], &KA[..], &MO[..], &LO[..]]
        .iter()
        .position(|set| set.contains(&w))
}

fn text_of(path: &str) -> String {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    if path.ends_with(".json") {
        let v: serde_json::Value = serde_json::from_str(&raw).expect("json");
        let mut out = String::new();
        for b in v["books"].as_array().into_iter().flatten() {
            for c in b["chapters"].as_array().into_iter().flatten() {
                for vs in c["verses"].as_array().into_iter().flatten() {
                    out.push_str(vs["text"].as_str().unwrap_or(""));
                    out.push('\n');
                }
            }
        }
        out
    } else if path.ends_with(".conllu") {
        raw.lines()
            .filter_map(|l| {
                let c: Vec<&str> = l.split('\t').collect();
                (!l.starts_with('#') && c.len() > 1 && !c[0].contains(['-', '.'])).then(|| c[1])
            })
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        let start = raw
            .find("*** START")
            .and_then(|i| raw[i..].find('\n').map(|j| i + j))
            .unwrap_or(0);
        let end = raw.find("*** END").unwrap_or(raw.len());
        raw[start..end.max(start)].to_string()
    }
}

/// (pairs in order, pairs total, clauses with >= 2 lanes, [lane counts],
/// lane-in-Vorfeld counts).
#[derive(Default)]
struct Count {
    in_order: usize,
    pairs: usize,
    clauses: usize,
    lanes: [usize; 4],
    vorfeld: [usize; 4],
    tokens: usize,
}

fn count(text: &str) -> Count {
    let mut c = Count::default();
    for clause in text.split([
        '.', ',', ';', ':', '!', '?', '(', ')', '"', '«', '»', '„', '“',
    ]) {
        let words: Vec<String> = clause
            .split(|ch: char| !ch.is_alphabetic())
            .filter(|w| !w.is_empty())
            .map(str::to_lowercase)
            .collect();
        c.tokens += words.len();
        let mut first: [Option<usize>; 4] = [None; 4];
        for (i, w) in words.iter().enumerate() {
            if let Some(l) = lane(w) {
                c.lanes[l] += 1;
                if i == 0 {
                    c.vorfeld[l] += 1;
                }
                first[l].get_or_insert(i);
            }
        }
        let present: Vec<(usize, usize)> =
            (0..4).filter_map(|l| first[l].map(|p| (l, p))).collect();
        if present.len() < 2 {
            continue;
        }
        c.clauses += 1;
        for a in 0..present.len() {
            for b in a + 1..present.len() {
                // `present` is in lane order, so lane a < lane b.
                c.pairs += 1;
                c.in_order += usize::from(present[a].1 < present[b].1);
            }
        }
    }
    c
}

/// Wilson 95 % interval of k / n.
fn wilson(k: usize, n: usize) -> (f64, f64, f64) {
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let (k, n, z) = (k as f64, n as f64, 1.96);
    let p = k / n;
    let d = 1.0 + z * z / n;
    let c = (p + z * z / (2.0 * n)) / d;
    let h = z * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt()) / d;
    (p, c - h, c + h)
}

fn line(name: &str, c: &Count) {
    let (p, lo, hi) = wilson(c.in_order, c.pairs);
    let per_10k = |x: usize| 1e4 * x as f64 / c.tokens.max(1) as f64;
    println!(
        "  {name:<32} in order {:5.1}% [{:4.1}, {:4.1}]  pairs {:6}  clauses {:6}  per 10k tokens Te {:5.1} Ka {:4.1} Mo {:5.1} Lo {:5.1}  Te in Vorfeld {:4.1}%",
        100.0 * p,
        100.0 * lo,
        100.0 * hi,
        c.pairs,
        c.clauses,
        per_10k(c.lanes[0]),
        per_10k(c.lanes[1]),
        per_10k(c.lanes[2]),
        per_10k(c.lanes[3]),
        100.0 * c.vorfeld[0] as f64 / c.lanes[0].max(1) as f64
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let split = args.iter().position(|a| a == "--").unwrap_or(args.len());
    let (lit, bible) = args.split_at(split);
    let lit: Vec<&String> = lit.iter().filter(|a| *a != "--lit").collect();
    let bible: Vec<&String> = bible
        .iter()
        .filter(|a| *a != "--" && *a != "--bible")
        .collect();
    if lit.is_empty() || bible.is_empty() {
        eprintln!("usage: tekamolo_order --lit TEXT... -- --bible TEXT...");
        std::process::exit(2);
    }
    let name = |p: &str| {
        std::path::Path::new(p)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(p)
            .to_string()
    };
    println!("edited literature / contemporary text:");
    let mut pooled = Count::default();
    for p in &lit {
        let c = count(&text_of(p));
        line(&name(p), &c);
        pooled.in_order += c.in_order;
        pooled.pairs += c.pairs;
        pooled.clauses += c.clauses;
        pooled.tokens += c.tokens;
        for l in 0..4 {
            pooled.lanes[l] += c.lanes[l];
            pooled.vorfeld[l] += c.vorfeld[l];
        }
    }
    line("POOLED", &pooled);
    let (_, lit_lo, _) = wilson(pooled.in_order, pooled.pairs);
    println!("Bible translations:");
    let mut all_below = true;
    for p in &bible {
        let c = count(&text_of(p));
        line(&name(p), &c);
        let (_, _, hi) = wilson(c.in_order, c.pairs);
        all_below &= hi < lit_lo;
    }
    println!(
        "prediction (every Bible below the pooled literature, non-overlapping 95% intervals): {}",
        if all_below { "PASS" } else { "KILL" }
    );
}
