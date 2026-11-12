//! Timing harness for D-DR-SELECT-K-1. Prints one TSV table on stdout and
//! nothing else there; progress goes to stderr. Run with
//! `cargo run --release --example report`.
//!
//! Key ranges: the fixture is built with `dup = 1`, so every key value is
//! held by exactly one row, for both patterns. The range `0..=K-1` therefore
//! selects exactly K rows (rows `0..K` when clustered, a scattered set of K
//! rows when random).
//!
//! Every cell is checked once against the independent oracle before it is
//! timed; a mismatch is reported on stderr and the process exits with 1.

use std::hint::black_box;
use std::io::Write;
use std::process;
use std::time::Instant;

use dr_select_k_lab::fixture::{self, Pattern};
use dr_select_k_lab::oracle::{self, Query};
use dr_select_k_lab::{
    admissible, ops, Bound, Counters, KeyRange, Op, Order, PreparedFolds, Residual, Selection,
    Source,
};

const N: usize = 1_000_000;
const SEED: u64 = 0x5EED_0001;
const WARMUP: usize = 3;
const RUNS: usize = 41;
const KS: [usize; 3] = [256, 65_536, 750_000];
const CONTROL_K: usize = 750_000;
/// The one K at which the `KeyThenGuid` producer is also run.
const GUID_K: usize = 65_536;
const COUNT_T: i32 = 500;
const TOP_LIMIT: usize = 10;

const HEADER: &str = "pattern\tk\tcontrol\tproducer\tresidual\tview\top\tadmissible\tm\t\
produce_ns\tbind_ns\tprepare_ns\tconvert_ns\tfold_ns\ttotal_r1_ns\ttotal_r100_ns\t\
bytes_copied\tretained_bytes\tstorage_after_bind\tfold_path";

#[derive(Clone, Copy)]
enum Producer {
    Filter,
    /// The index order with ties by source row (`Order::KeyThenRow`).
    Ordered,
    /// The index order with ties by guid (`Order::KeyThenGuid`).
    OrderedGuid,
}

impl Producer {
    fn name(self) -> &'static str {
        match self {
            Producer::Filter => "filter",
            Producer::Ordered => "ordered",
            Producer::OrderedGuid => "ordered_guid",
        }
    }

    /// The order the binding made from this producer witnesses.
    fn order(self) -> Order {
        match self {
            Producer::Filter => Order::Source,
            Producer::Ordered => Order::KeyThenRow,
            Producer::OrderedGuid => Order::KeyThenGuid,
        }
    }

    /// The order the `top_by_key` request names. A filter binding is asked
    /// for `KeyThenRow` (the existing contract) and is refused.
    fn want(self) -> Order {
        match self {
            Producer::Filter | Producer::Ordered => Order::KeyThenRow,
            Producer::OrderedGuid => Order::KeyThenGuid,
        }
    }

    fn select(self, src: &Source, range: KeyRange, c: &Counters) -> Selection {
        match self {
            Producer::Filter => src.select_filter(range, c),
            Producer::Ordered | Producer::OrderedGuid => src.select_ordered(range, self.order(), c),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    All,
    Positions,
    Mask,
}

impl Form {
    fn name(self) -> &'static str {
        match self {
            Form::All => "all",
            Form::Positions => "positions",
            Form::Mask => "mask",
        }
    }

    /// An empty residual of the wanted form, used as the `like` argument.
    fn like(self) -> Residual {
        match self {
            Form::All => Residual::All,
            Form::Positions => Residual::Positions(Vec::new()),
            Form::Mask => Residual::Mask(Vec::new()),
        }
    }
}

/// Median in nanoseconds of `RUNS` timed runs after `WARMUP` untimed ones.
/// `setup` runs outside the timed region and feeds `run`; the value `run`
/// returns is dropped after the clock stops.
fn median<T, R>(mut setup: impl FnMut() -> T, mut run: impl FnMut(T) -> R) -> u64 {
    let mut samples = [0u64; RUNS];
    for i in 0..WARMUP + RUNS {
        let input = black_box(setup());
        let t0 = Instant::now();
        let result = run(input);
        let dt = t0.elapsed().as_nanos() as u64;
        black_box(&result);
        drop(result);
        if i >= WARMUP {
            samples[i - WARMUP] = dt;
        }
    }
    samples.sort_unstable();
    samples[RUNS / 2]
}

fn mismatch(ctx: &str, what: &str) -> ! {
    eprintln!("ORACLE MISMATCH: {ctx}: {what}");
    process::exit(1);
}

/// The oracle's answers for one query: count, sum, and the top-T under each
/// declared key order.
struct Expected {
    count: u64,
    sum: i64,
    top_row: Vec<u128>,
    top_guid: Vec<u128>,
}

impl Expected {
    /// The oracle top-T for a requested key order.
    fn top(&self, order: Order) -> &[u128] {
        match order {
            Order::KeyThenGuid => &self.top_guid,
            Order::KeyThenRow => &self.top_row,
            Order::Source => unreachable!("no report cell requests the source order"),
        }
    }
}

fn expected(src: &Source, q: Query) -> Expected {
    Expected {
        count: oracle::count_b_gt(src, q, COUNT_T),
        sum: oracle::sum_a(src, q),
        top_row: oracle::top_by_key(src, q, TOP_LIMIT, Order::KeyThenRow),
        top_guid: oracle::top_by_key(src, q, TOP_LIMIT, Order::KeyThenGuid),
    }
}

/// The N-bit plane of the resident baseline: bit r set iff the key of row r
/// is in range and, when there is a threshold, `a >= t_a`.
fn build_plane(src: &Source, range: KeyRange, t_a: Option<i32>) -> Vec<u64> {
    let (_, key, a, _) = src.lanes();
    let mut plane = vec![0u64; key.len().div_ceil(64)];
    for (r, (&k, &av)) in key.iter().zip(a).enumerate() {
        if range.has(k) && t_a.is_none_or(|t| av >= t) {
            plane[r / 64] |= 1u64 << (r % 64);
        }
    }
    plane
}

fn popcount(words: &[u64]) -> usize {
    words.iter().map(|w| w.count_ones() as usize).sum()
}

fn residual_rows(r: &Residual, k: usize) -> usize {
    match r {
        Residual::All => k,
        Residual::Positions(p) => p.len(),
        Residual::Mask(m) => popcount(m),
    }
}

fn residual_label(t_a: Option<i32>) -> String {
    match t_a {
        None => "all".to_string(),
        Some(t) => format!("t_a={t}"),
    }
}

fn emit(
    out: &mut dyn Write,
    prefix: &[String],
    op: &str,
    adm: &str,
    m: &str,
    nums: [String; 10],
    fold_path: &str,
) {
    let mut fields: Vec<&str> = prefix.iter().map(String::as_str).collect();
    fields.extend([op, adm, m]);
    fields.extend(nums.iter().map(String::as_str));
    fields.push(fold_path);
    writeln!(out, "{}", fields.join("\t")).unwrap();
}

/// Which execution path a fold takes, read from `ops.rs`: `count_b_gt` and
/// `sum_a` go through mask-RISC for `All` and `Mask` and through a plain loop
/// for `Positions`; `top_by_key` is a scalar walk for every view.
fn fold_path(op: Op, form: Form) -> &'static str {
    match op {
        Op::TopByKey { .. } => "scalar-walk",
        Op::CountBGt { .. } | Op::SumA => {
            if form == Form::Positions {
                "scalar-loop"
            } else {
                "mask-risc"
            }
        }
    }
}

fn empty_nums() -> [String; 10] {
    std::array::from_fn(|_| String::new())
}

/// Everything that is fixed for one (pattern, K, threshold) group.
struct Ctx<'a> {
    src: &'a Source,
    pattern: &'static str,
    k: usize,
    range: KeyRange,
    t_a: Option<i32>,
    exp: &'a Expected,
}

impl Ctx<'_> {
    fn control(&self) -> &'static str {
        if self.k == CONTROL_K {
            "control75"
        } else {
            "-"
        }
    }

    fn prefix(&self, producer: &str, view: &str) -> Vec<String> {
        vec![
            self.pattern.to_string(),
            self.k.to_string(),
            self.control().to_string(),
            producer.to_string(),
            residual_label(self.t_a),
            view.to_string(),
        ]
    }

    /// The resident baseline: all N rows folded with an N-bit plane.
    fn baseline(&self, out: &mut dyn Write) {
        let scratch = Counters::new();
        let plane = build_plane(self.src, self.range, self.t_a);
        let m = popcount(&plane);
        let produce_ns = median(|| (), |()| build_plane(self.src, self.range, self.t_a));
        let prefix = self.prefix("none", "resident_n");
        let retained = plane.len() * 8;

        for op in ["count_b_gt", "sum_a"] {
            let ctx = format!(
                "{} k={} {} resident_n {op}",
                self.pattern,
                self.k,
                residual_label(self.t_a)
            );
            let (fold_ns, delta) = if op == "count_b_gt" {
                let got = ops::baseline::count_b_gt(self.src, &plane, COUNT_T, &scratch);
                if got != self.exp.count {
                    mismatch(&ctx, &format!("got {got}, oracle {}", self.exp.count));
                }
                let s0 = scratch.storage.get();
                let ns = median(
                    || (),
                    |()| ops::baseline::count_b_gt(self.src, &plane, COUNT_T, &scratch),
                );
                (ns, scratch.storage.get() - s0)
            } else {
                let got = ops::baseline::sum_a(self.src, &plane, &scratch);
                if got != self.exp.sum {
                    mismatch(&ctx, &format!("got {got}, oracle {}", self.exp.sum));
                }
                let s0 = scratch.storage.get();
                let ns = median(|| (), |()| ops::baseline::sum_a(self.src, &plane, &scratch));
                (ns, scratch.storage.get() - s0)
            };
            let mut nums = empty_nums();
            nums[0] = produce_ns.to_string();
            nums[4] = fold_ns.to_string();
            nums[5] = (produce_ns + fold_ns).to_string();
            nums[6] = (produce_ns + 100 * fold_ns).to_string();
            nums[8] = retained.to_string();
            nums[9] = delta.to_string();
            // The baseline folds through mask-RISC over all N rows.
            emit(out, &prefix, op, "yes", &m.to_string(), nums, "mask-risc");
        }
    }

    /// One producer and one stored form of the residual, all three operations.
    fn cell(&self, out: &mut dyn Write, producer: Producer, form: Form) {
        let src = self.src;
        let range = self.range;
        let prefix = self.prefix(producer.name(), form.name());

        // One produce-bind-convert sequence: the binding the folds read, and
        // the source of `bytes_copied`.
        let seq = Counters::new();
        let selection = producer.select(src, range, &seq);
        let bound = Bound::bind(src, selection, &seq).expect("bound at the source's own version");
        if bound.k() != self.k {
            mismatch(&prefix.join(" "), "the selection is not K rows");
        }
        let like = form.like();
        let t = self.t_a.unwrap_or(0);
        let residual = ops::residual_a_ge(&bound, t, &like, &seq);
        let bytes_copied = seq.snapshot()[4];
        let m = residual_rows(&residual, bound.k());
        let retained = residual.retained_bytes() + 4 * bound.k() + 28 * bound.k();

        // Once-per-version terms, each on its own fresh input.
        let scratch = Counters::new();
        let produce_ns = median(|| (), |()| producer.select(src, range, &scratch));
        let bind_ns = median(
            || producer.select(src, range, &scratch),
            |sel| Bound::bind(src, sel, &scratch),
        );
        let prepare_ns = median(|| (), |()| PreparedFolds::new(&bound));
        let convert_ns = if form == Form::All {
            0
        } else {
            median(|| (), |()| ops::residual_a_ge(&bound, t, &like, &scratch))
        };

        let mut prepared = PreparedFolds::new(&bound);
        let view = residual.view();
        let once = produce_ns + bind_ns + prepare_ns + convert_ns;

        let op_list = [
            ("count_b_gt", Op::CountBGt { t: COUNT_T }),
            ("sum_a", Op::SumA),
            (
                "top_by_key",
                Op::TopByKey {
                    limit: TOP_LIMIT,
                    order: producer.want(),
                },
            ),
        ];
        for (name, op) in op_list {
            if admissible(op, &bound, view).is_err() {
                // Untimed: nothing ran, so there is no fold path either.
                emit(out, &prefix, name, "no", &m.to_string(), empty_nums(), "");
                continue;
            }
            let ctx = format!("{} {name}", prefix.join(" "));
            let (fold_ns, delta) = match op {
                Op::CountBGt { t } => {
                    let got = prepared.count_b_gt(&bound, view, t, &seq);
                    if got != self.exp.count {
                        mismatch(&ctx, &format!("got {got}, oracle {}", self.exp.count));
                    }
                    let s0 = seq.storage.get();
                    let ns = median(|| (), |()| prepared.count_b_gt(&bound, view, t, &seq));
                    (ns, seq.storage.get() - s0)
                }
                Op::SumA => {
                    let got = prepared.sum_a(&bound, view, &seq);
                    if got != self.exp.sum {
                        mismatch(&ctx, &format!("got {got}, oracle {}", self.exp.sum));
                    }
                    let s0 = seq.storage.get();
                    let ns = median(|| (), |()| prepared.sum_a(&bound, view, &seq));
                    (ns, seq.storage.get() - s0)
                }
                Op::TopByKey { limit, order } => {
                    let mut top: Vec<u128> = Vec::new();
                    if prepared
                        .top_by_key(&bound, view, limit, order, &mut top, &seq)
                        .is_err()
                    {
                        mismatch(&ctx, "refused although admissible() accepted");
                    }
                    if top.as_slice() != self.exp.top(order) {
                        mismatch(&ctx, "guid sequence differs from the oracle");
                    }
                    let s0 = seq.storage.get();
                    let ns = median(
                        || (),
                        |()| prepared.top_by_key(&bound, view, limit, order, &mut top, &seq),
                    );
                    (ns, seq.storage.get() - s0)
                }
            };
            let nums = [
                produce_ns.to_string(),
                bind_ns.to_string(),
                prepare_ns.to_string(),
                convert_ns.to_string(),
                fold_ns.to_string(),
                (once + fold_ns).to_string(),
                (once + 100 * fold_ns).to_string(),
                bytes_copied.to_string(),
                retained.to_string(),
                delta.to_string(),
            ];
            emit(
                out,
                &prefix,
                name,
                "yes",
                &m.to_string(),
                nums,
                fold_path(op, form),
            );
        }
    }
}

fn main() {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(
        out,
        "# N={N} runs={RUNS} (after {WARMUP} warm-up) ns, medians; \
produce/bind/convert/prepare are once per version, reported not gated; fold is the hot term"
    )
    .unwrap();
    writeln!(out, "{HEADER}").unwrap();

    for (pattern, pattern_name) in [
        (Pattern::Clustered, "clustered"),
        (Pattern::Random, "random"),
    ] {
        let src = fixture::source(N, pattern, 1, SEED);
        for k in KS {
            let range = KeyRange {
                lo: 0,
                hi: k as i32 - 1,
            };
            for t_a in [None, Some(99), Some(50)] {
                eprintln!("{pattern_name} k={k} {}", residual_label(t_a));
                let exp = expected(&src, Query { range, t_a });
                let ctx = Ctx {
                    src: &src,
                    pattern: pattern_name,
                    k,
                    range,
                    t_a,
                    exp: &exp,
                };
                ctx.baseline(&mut out);
                let forms: &[Form] = if t_a.is_none() {
                    &[Form::All]
                } else {
                    &[Form::Positions, Form::Mask]
                };
                let mut producers = vec![Producer::Filter, Producer::Ordered];
                if k == GUID_K {
                    producers.push(Producer::OrderedGuid);
                }
                for producer in producers {
                    for &form in forms {
                        ctx.cell(&mut out, producer, form);
                    }
                }
            }
        }
    }
}
