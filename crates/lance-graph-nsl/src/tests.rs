use super::*;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

const FIXTURES: &str = include_str!("../data/fixtures.txt");
const GOLDEN: &str = include_str!("../data/nsl_golden.txt");

struct Fixture {
    id: String,
    cap: Option<usize>,
    undirected: bool,
    seeds: Vec<Example>,
    unlabeled: Vec<Example>,
    tsv: String,
}

fn fixtures() -> Vec<Fixture> {
    let mut out = Vec::new();
    let mut cur: Option<Fixture> = None;
    for line in FIXTURES
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let tok: Vec<&str> = line.split(' ').collect();
        match tok[0] {
            "fixture" => {
                let opt = |k: &str| {
                    tok[2..]
                        .iter()
                        .find_map(|t| t.strip_prefix(k))
                        .expect("option present")
                };
                cur = Some(Fixture {
                    id: tok[1].into(),
                    cap: match opt("cap=") {
                        "none" => None,
                        n => Some(n.parse().unwrap()),
                    },
                    undirected: opt("undirected=") == "1",
                    seeds: vec![],
                    unlabeled: vec![],
                    tsv: String::new(),
                });
            }
            "seed" | "unlabeled" => {
                let f = cur.as_mut().unwrap();
                let e = Example {
                    id: tok[1].into(),
                    tag: tok[2].into(),
                    x: tok[3].parse().unwrap(),
                };
                if tok[0] == "seed" {
                    f.seeds.push(e)
                } else {
                    f.unlabeled.push(e)
                }
            }
            "edge" => {
                let f = cur.as_mut().unwrap();
                f.tsv.push_str(&tok[1..].join("\t"));
                f.tsv.push('\n');
            }
            "end" => out.push(cur.take().unwrap()),
            other => panic!("bad fixture line {other}"),
        }
    }
    out
}

fn fixture(id: &str) -> Fixture {
    fixtures()
        .into_iter()
        .find(|f| f.id == id)
        .expect("fixture exists")
}

fn golden(id: &str) -> Vec<String> {
    GOLDEN
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter(|l| l.split(' ').nth(1) == Some(id))
        .map(String::from)
        .collect()
}

fn run(f: &Fixture) -> Result<(FeatureTable, Vec<NeighborBatch>), NslError> {
    let mut g = Graph::from_tsv(&f.tsv)?;
    if f.undirected {
        g.symmetrize();
    }
    let t = FeatureTable::new(&f.seeds, &f.unlabeled)?;
    let b = select(&Adjacency::build(&g, &t), f.cap);
    Ok((t, b))
}

#[test]
fn every_fixture_matches_the_frozen_nsl_oracle() {
    let mut compared = 0;
    for f in fixtures() {
        let want = golden(&f.id);
        assert!(
            !want.is_empty(),
            "{}: no oracle output; regenerate the golden",
            f.id
        );
        if f.id == "G12b" {
            continue; // NaN: refused, see the dedicated test
        }
        let (t, b) = run(&f).unwrap_or_else(|e| panic!("{}: {e:?}", f.id));
        assert_eq!(compat_lines(&f.id, &t, &b), want, "{}", f.id);
        compared += 1;
    }
    assert_eq!(compared, 18, "fixtures compared: 19 minus the refused G12b");
}

#[test]
fn g03b_orders_by_the_parsed_f64_not_the_emitted_f32() {
    let (t, b) = run(&fixture("G03b")).unwrap();
    let ids: Vec<&str> = b[0].selected.iter().map(|s| t.id(s.row)).collect();
    assert_eq!(ids, ["B", "C"]);
    assert_eq!(
        b[0].selected[0].weight as f32,
        b[0].selected[1].weight as f32
    );
}

#[test]
fn g12c_signed_zeros_tie_and_the_id_decides() {
    let (t, b) = run(&fixture("G12c")).unwrap();
    let ids: Vec<&str> = b[0].selected.iter().map(|s| t.id(s.row)).collect();
    assert_eq!(ids, ["C", "B"]);
}

#[test]
fn g12b_nan_weight_is_refused_and_nsl_order_is_input_dependent() {
    assert_eq!(
        run(&fixture("G12b")).err(),
        Some(NslError::NanWeight {
            src: "A".into(),
            dst: "B".into()
        })
    );
    // The oracle put NaN first only because it was read first; nothing to match.
    assert_eq!(golden("G12b")[1].split(' ').nth(4), Some("B"));
}

#[test]
fn g05_seed_features_win_over_the_unlabeled_copy() {
    let (t, b) = run(&fixture("G05")).unwrap();
    let row = b[0].selected[0].row;
    assert_eq!((t.tag(row), t.x(row)), ("seed-B", 5.0));
    assert_eq!(t.rows(), 2, "the unlabeled B is not stored");
}

#[test]
fn g06_g07_zero_neighbors_keep_an_explicit_empty_batch() {
    for (id, seed) in [("G06", "A"), ("G07", "B")] {
        let (t, b) = run(&fixture(id)).unwrap();
        let batch = b.iter().find(|x| t.id(x.seed) == seed).unwrap();
        assert!(batch.selected.is_empty(), "{id}");
        let line = compat_lines(id, &t, std::slice::from_ref(batch));
        assert_eq!(line, [format!("seed {id} {seed} 0 NL_num_nbrs,id,tag,x")]);
    }
}

#[test]
fn g08_a_shared_neighbor_is_stored_once_and_referenced_three_times() {
    let (t, b) = run(&fixture("G08")).unwrap();
    let c = t.row("C").unwrap();
    assert_eq!(t.rows(), 4);
    assert!(b
        .iter()
        .all(|x| x.selected.len() == 1 && x.selected[0].row == c));
    // What NSL copies: id, tag, x per selected neighbor (weight is per edge on both sides).
    let row_bytes = t.id(c).len() + t.tag(c).len() + 4;
    let packed = 3 * row_bytes;
    let referenced = row_bytes + 3 * 4; // one row + three u32 ordinals
    eprintln!("G08: NSL copies {packed} feature bytes, references cost {referenced}");
}

#[test]
fn g11_caps_truncate_without_inventing_rows() {
    let lens: Vec<usize> = ["G11a", "G11b", "G11c"]
        .iter()
        .map(|id| run(&fixture(id)).unwrap().1[0].selected.len())
        .collect();
    assert_eq!(lens, [0, 1, 3]);
    let unlimited = run(&fixture("G11c")).unwrap().1;
    let mut big = fixture("G11c");
    big.cap = Some(usize::MAX);
    assert_eq!(
        run(&big).unwrap().1,
        unlimited,
        "a cap above the degree is no cap"
    );
}

const S1: SourceRef = SourceRef {
    dataset_id: 7,
    version: 3,
    index_seal: 1,
};

fn g08_table() -> FeatureTable {
    let f = fixture("G08");
    FeatureTable::new(&f.seeds, &f.unlabeled).unwrap()
}

#[test]
fn g09_a_changed_seal_is_refused_before_any_feature_read() {
    let lease = FocusLease::default();
    lease.get(S1, g08_table).unwrap();
    let stale = SourceRef {
        index_seal: 2,
        ..S1
    };
    let loads = AtomicUsize::new(0);
    let r = lease.get(stale, || {
        loads.fetch_add(1, AtomicOrdering::SeqCst);
        g08_table()
    });
    assert_eq!(
        r.err(),
        Some(NslError::StaleSeal {
            loaded: S1,
            requested: stale
        })
    );
    assert_eq!(loads.load(AtomicOrdering::SeqCst), 0);
    assert!(
        lease.get(S1, g08_table).is_ok(),
        "the original seal still reads"
    );
}

#[test]
fn g10_eight_concurrent_readers_share_one_load() {
    let lease = FocusLease::default();
    let loads = AtomicUsize::new(0);
    let tables: Vec<Arc<FeatureTable>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..8)
            .map(|_| {
                s.spawn(|| {
                    lease
                        .get(S1, || {
                            loads.fetch_add(1, AtomicOrdering::SeqCst);
                            std::thread::sleep(std::time::Duration::from_micros(200));
                            g08_table()
                        })
                        .unwrap()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(loads.load(AtomicOrdering::SeqCst), 1);
    assert!(tables.iter().all(|t| Arc::ptr_eq(t, &tables[0])));
}

#[test]
fn weight_text_follows_python_float_or_is_refused() {
    assert_eq!(parse_weight(" 0.5 "), Some(0.5));
    assert_eq!(parse_weight("inf"), Some(f64::INFINITY));
    assert_eq!(parse_weight("-Infinity"), Some(f64::NEG_INFINITY));
    assert_eq!(parse_weight("1e-3"), Some(0.001));
    assert_eq!(
        parse_weight("1_0"),
        None,
        "Python reads 10.0; refused, not misread"
    );
    assert_eq!(parse_weight("abc"), None);
    assert_eq!(
        Graph::from_tsv("A\tB\t1_0\n").err(),
        Some(NslError::BadWeight {
            line: 1,
            text: "1_0".into()
        })
    );
    assert_eq!(
        Graph::from_tsv("A\n").err(),
        Some(NslError::MalformedLine { line: 1 })
    );
}

#[test]
fn duplicate_examples_are_refused() {
    let e = |id: &str| Example {
        id: id.into(),
        tag: "t".into(),
        x: 0.0,
    };
    assert!(FeatureTable::new(&[e("A"), e("A")], &[]).is_err());
    assert!(FeatureTable::new(&[e("A")], &[e("B"), e("B")]).is_err());
    assert!(
        FeatureTable::new(&[e("A")], &[e("A")]).is_ok(),
        "seed and unlabeled may share"
    );
}

/// Cost probe (not a gate). Power-law out-degree graph, `cargo test --release
/// -- --ignored --nocapture cost_probe`.
#[test]
#[ignore]
fn cost_probe() {
    let n = 2000usize;
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let ex = |i: usize| Example {
        id: format!("n{i:05}"),
        tag: format!("t{i}"),
        x: i as f32,
    };
    let seeds: Vec<Example> = (0..n / 2).map(ex).collect();
    let unl: Vec<Example> = (n / 2..n).map(ex).collect();
    let mut tsv = String::new();
    let mut edges = 0;
    for s in 0..n / 2 {
        let deg = 1 + (64 >> (next() % 7)) as usize; // 2..65, heavy at the low end
        for _ in 0..deg {
            let d = (next() % n as u64) as usize;
            let w = (next() % 1000) as f64 / 1000.0;
            tsv.push_str(&format!("n{s:05}\tn{d:05}\t{w}\n"));
            edges += 1;
        }
    }
    let t0 = std::time::Instant::now();
    let g = Graph::from_tsv(&tsv).unwrap();
    let parse = t0.elapsed();
    let t = FeatureTable::new(&seeds, &unl).unwrap();
    let reps: usize = 20;
    let t1 = std::time::Instant::now();
    let adj = Adjacency::build(&g, &t);
    let build = t1.elapsed();
    let t2 = std::time::Instant::now();
    let mut sel = 0;
    for _ in 0..reps {
        for seed in 0..adj.seeds() {
            sel += std::hint::black_box(adj.neighbors(seed, Some(16))).len();
        }
    }
    let per_seed = t2.elapsed().as_nanos() / (reps * n / 2) as u128;
    eprintln!(
        "edges {edges}: cold parse {} us, cold freeze {} us (reported, not gates); \
         neighbors cap 16: {per_seed} ns per seed ({} selected per pass)",
        parse.as_micros(),
        build.as_micros(),
        sel / reps
    );
}
