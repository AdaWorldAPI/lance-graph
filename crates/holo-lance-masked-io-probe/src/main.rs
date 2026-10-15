//! D-HOLO H14c-4 and -5: Lance 13 masked I/O and coordinate identity.
//!
//! Pre-registered in `.claude/plans/d-holo-h14c-thought-first-v1.md` (§ 7, § 8)
//! before any run.
//!
//! One Lance dataset per tile: an explicit `coord: u32` column (the Cartesian
//! address, the logical identity), an `epoch: u32` column (rewritten by
//! updates), and `LANES` lane columns `FixedSizeBinary(16)`, nullable (null =
//! absent). A [`CoordMap`] translates coordinates into the dataset's `_rowid`
//! space at one named version; a masked read refuses any other version
//! ([`ReadError::StaleMapping`]). Values are attributed by their stored
//! `coord`, never by output position.
//!
//! Strategies measured: unmasked projection (the I/O baseline); combined
//! (one mask = rows where every requested lane is present, one projection);
//! independent (one masked read per lane with that lane's own mask, aligned
//! by `coord`). Counters come from `Scanner::scan_stats_callback`
//! (`ExecutionSummaryCounts`).
//!
//! Run: `cargo run --release --manifest-path
//! crates/holo-lance-masked-io-probe/Cargo.toml`. Tests: `cargo test --release
//! --manifest-path crates/holo-lance-masked-io-probe/Cargo.toml`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arrow_array::builder::FixedSizeBinaryBuilder;
use arrow_array::{
    Array, FixedSizeBinaryArray, RecordBatch, RecordBatchIterator, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use lance::dataset::scanner::{MaterializationStyle, RowAddrMask, RowAddrTreeMap};
use lance::dataset::{Dataset, WriteMode, WriteParams};

const LANE: usize = 16;
/// Lane columns in the fixture (the grid goes to k = 8).
const LANES: usize = 8;
const N: u32 = 65_536;

type Val = [u8; LANE];

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn lane_name(l: usize) -> String {
    format!("lane_{l}")
}

/// The value of lane `l` at coordinate `c` in a given epoch (never all zero).
fn value(l: usize, c: u32, epoch: u32) -> Val {
    let a = mix(((l as u64) << 40) ^ ((epoch as u64) << 32) ^ c as u64);
    let b = mix(a ^ 0x77);
    let mut v = [0u8; LANE];
    v[..8].copy_from_slice(&a.to_le_bytes());
    v[8..].copy_from_slice(&b.to_le_bytes());
    v[0] |= 1;
    v
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pattern {
    Uniform,
    Clustered,
}

impl Pattern {
    fn name(self) -> &'static str {
        match self {
            Pattern::Uniform => "uniform",
            Pattern::Clustered => "clustered",
        }
    }
}

/// Whether lane `l` holds a value at coordinate `c` (independent per lane).
fn present(p: Pattern, d: f64, l: usize, c: u32) -> bool {
    if d >= 1.0 {
        return true;
    }
    let t = (d * u64::MAX as f64) as u64;
    match p {
        Pattern::Uniform => mix(0x1A9E ^ ((l as u64) << 40) ^ c as u64) < t,
        Pattern::Clustered => mix(0xC1u64 ^ ((l as u64) << 40) ^ (c / 64) as u64) < t,
    }
}

fn schema() -> Arc<Schema> {
    let mut f = vec![
        Field::new("coord", DataType::UInt32, false),
        Field::new("epoch", DataType::UInt32, false),
    ];
    for l in 0..LANES {
        f.push(Field::new(
            lane_name(l),
            DataType::FixedSizeBinary(LANE as i32),
            true,
        ));
    }
    Arc::new(Schema::new(f))
}

/// Rows for the given coordinates, in the given order.
fn batch(coords: &[u32], p: Pattern, d: f64, epoch: u32) -> RecordBatch {
    let mut cols: Vec<Arc<dyn Array>> = vec![
        Arc::new(UInt32Array::from(coords.to_vec())),
        Arc::new(UInt32Array::from(vec![epoch; coords.len()])),
    ];
    for l in 0..LANES {
        let mut b = FixedSizeBinaryBuilder::with_capacity(coords.len(), LANE as i32);
        for &c in coords {
            if present(p, d, l, c) {
                b.append_value(value(l, c, epoch)).unwrap();
            } else {
                b.append_null();
            }
        }
        cols.push(Arc::new(b.finish()));
    }
    RecordBatch::try_new(schema(), cols).unwrap()
}

async fn write(
    uri: &str,
    batches: Vec<RecordBatch>,
    mode: WriteMode,
    stable: bool,
    rows_per_file: usize,
) -> Dataset {
    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema());
    let params = WriteParams {
        mode,
        enable_stable_row_ids: stable,
        max_rows_per_file: rows_per_file,
        ..Default::default()
    };
    Dataset::write(reader, uri, Some(params)).await.unwrap()
}

// ------------------------------------------------------------- identity

/// Coordinate -> `_rowid` at one dataset version. Valid only for that version.
#[derive(Clone, Debug)]
struct CoordMap {
    version: u64,
    id_of: BTreeMap<u32, u64>,
}

#[derive(Debug, PartialEq, Eq)]
enum ReadError {
    /// The map was built at another dataset version.
    StaleMapping { map: u64, dataset: u64 },
    /// Lance refused the read (missing column, missing dataset, …).
    Lance(String),
}

async fn build_map(ds: &Dataset) -> CoordMap {
    let mut s = ds.scan();
    s.project(&["coord"]).unwrap();
    s.with_row_id();
    let b = s.try_into_batch().await.unwrap();
    let coord = b
        .column_by_name("coord")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let id = b
        .column_by_name("_rowid")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let mut id_of = BTreeMap::new();
    for j in 0..b.num_rows() {
        let prev = id_of.insert(coord.value(j), id.value(j));
        assert!(prev.is_none(), "coord {} twice", coord.value(j));
    }
    CoordMap {
        version: ds.version().version,
        id_of,
    }
}

#[derive(Clone, Debug, Default)]
struct Io {
    bytes_read: usize,
    iops: usize,
    requests: usize,
    rows: usize,
    counts: BTreeMap<String, usize>,
}

/// One masked read: `coords` (by the map) into `_rowid`s, then a projection of
/// `coord` plus `cols`. Returns, per returned row, its stored coord and the
/// values of `cols` (None = null).
async fn masked_read(
    ds: &Dataset,
    map: &CoordMap,
    coords: &[u32],
    cols: &[usize],
    style: MaterializationStyle,
    check_version: bool,
) -> Result<(BTreeMap<u32, Vec<Option<Val>>>, Io), ReadError> {
    if check_version && ds.version().version != map.version {
        return Err(ReadError::StaleMapping {
            map: map.version,
            dataset: ds.version().version,
        });
    }
    let ids = coords.iter().filter_map(|c| map.id_of.get(c).copied());
    let mask = RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(ids));
    read_with(ds, Some(mask), cols, style).await
}

async fn read_with(
    ds: &Dataset,
    mask: Option<RowAddrMask>,
    cols: &[usize],
    style: MaterializationStyle,
) -> Result<(BTreeMap<u32, Vec<Option<Val>>>, Io), ReadError> {
    let mut names = vec!["coord".to_string()];
    names.extend(cols.iter().map(|&l| lane_name(l)));
    let stats: Arc<Mutex<Option<Io>>> = Arc::new(Mutex::new(None));
    let sink = stats.clone();
    let mut s = ds.scan();
    s.project(&names)
        .map_err(|e| ReadError::Lance(e.to_string()))?;
    if let Some(m) = mask {
        s.with_row_addr_prefilter(m);
    }
    s.materialization_style(style);
    s.scan_stats_callback(Arc::new(move |c| {
        *sink.lock().unwrap() = Some(Io {
            bytes_read: c.bytes_read,
            iops: c.iops,
            requests: c.requests,
            rows: 0,
            counts: c.all_counts.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        });
    }));
    let b = s
        .try_into_batch()
        .await
        .map_err(|e| ReadError::Lance(e.to_string()))?;
    let mut io = stats.lock().unwrap().clone().unwrap_or_default();
    io.rows = b.num_rows();
    Ok((align(&b, cols), io))
}

/// Values keyed by the row's stored coordinate (never by position).
fn align(b: &RecordBatch, cols: &[usize]) -> BTreeMap<u32, Vec<Option<Val>>> {
    let coord = b
        .column_by_name("coord")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let lanes: Vec<&FixedSizeBinaryArray> = cols
        .iter()
        .map(|&l| {
            b.column_by_name(&lane_name(l))
                .unwrap()
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap()
        })
        .collect();
    let mut m = BTreeMap::new();
    for j in 0..b.num_rows() {
        let v = lanes
            .iter()
            .map(|a| (!a.is_null(j)).then(|| a.value(j).try_into().unwrap()))
            .collect();
        m.insert(coord.value(j), v);
    }
    m
}

// ------------------------------------------------------------- benchmark

fn style_name(s: &MaterializationStyle) -> &'static str {
    match s {
        MaterializationStyle::Heuristic => "heuristic",
        MaterializationStyle::AllLate => "all_late",
        MaterializationStyle::AllEarly => "all_early",
        MaterializationStyle::AllEarlyExcept(_) => "all_early_except",
    }
}

fn styles() -> [MaterializationStyle; 3] {
    [
        MaterializationStyle::Heuristic,
        MaterializationStyle::AllLate,
        MaterializationStyle::AllEarly,
    ]
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    println!("lance: 13.0.0 (resolved; see Cargo.lock of this probe crate)");
    println!(
        "tier: {}",
        if cfg!(target_feature = "avx512f") {
            "v4 (avx512f)"
        } else {
            "below v4: timings not reportable"
        }
    );
    println!("csv,strategy,pattern,density,k,style,rows_emitted,bytes_read,iops,requests,rows_scanned,ranges_scanned,wall_us");
    let dir = tempfile::tempdir().unwrap();
    let mut printed_keys = false;
    for p in [Pattern::Uniform, Pattern::Clustered] {
        for &d in &[0.001, 0.01, 0.1, 0.5, 1.0] {
            let uri = dir.path().join(format!("{}_{d}", p.name()));
            let uri = uri.to_str().unwrap();
            let coords: Vec<u32> = (0..N).collect();
            let ds = write(
                uri,
                vec![batch(&coords, p, d, 1)],
                WriteMode::Create,
                false,
                1 << 20,
            )
            .await;
            let map = build_map(&ds).await;
            for &k in &[1usize, 2, 4, 8] {
                let lanes: Vec<usize> = (0..k).collect();
                for style in styles() {
                    // unmasked baseline
                    let mut walls = Vec::new();
                    let mut io = Io::default();
                    for _ in 0..5 {
                        let t = Instant::now();
                        let (_, i) = read_with(&ds, None, &lanes, style.clone()).await.unwrap();
                        walls.push(t.elapsed().as_secs_f64() * 1e6);
                        io = i;
                    }
                    if !printed_keys {
                        println!(
                            "all_counts keys: {:?}",
                            io.counts.keys().collect::<Vec<_>>()
                        );
                        printed_keys = true;
                    }
                    report("unmasked", p, d, k, &style, &io, median(walls));
                    // combined: every requested lane present
                    let joint: Vec<u32> = coords
                        .iter()
                        .copied()
                        .filter(|&c| lanes.iter().all(|&l| present(p, d, l, c)))
                        .collect();
                    let mut walls = Vec::new();
                    for _ in 0..5 {
                        let t = Instant::now();
                        let (_, i) = masked_read(&ds, &map, &joint, &lanes, style.clone(), true)
                            .await
                            .unwrap();
                        walls.push(t.elapsed().as_secs_f64() * 1e6);
                        io = i;
                    }
                    report("combined", p, d, k, &style, &io, median(walls));
                    // independent: each lane with its own mask, aligned by coord
                    let mut walls = Vec::new();
                    let mut sum = Io::default();
                    for _ in 0..5 {
                        sum = Io::default();
                        let t = Instant::now();
                        for &l in &lanes {
                            let own: Vec<u32> = coords
                                .iter()
                                .copied()
                                .filter(|&c| present(p, d, l, c))
                                .collect();
                            let (_, i) = masked_read(&ds, &map, &own, &[l], style.clone(), true)
                                .await
                                .unwrap();
                            sum.bytes_read += i.bytes_read;
                            sum.iops += i.iops;
                            sum.requests += i.requests;
                            sum.rows += i.rows;
                            for (key, v) in i.counts {
                                *sum.counts.entry(key).or_default() += v;
                            }
                        }
                        walls.push(t.elapsed().as_secs_f64() * 1e6);
                    }
                    report("independent", p, d, k, &style, &sum, median(walls));
                }
            }
        }
    }
}

fn report(
    strategy: &str,
    p: Pattern,
    d: f64,
    k: usize,
    s: &MaterializationStyle,
    io: &Io,
    wall_us: f64,
) {
    println!(
        "csv,{strategy},{},{d},{k},{},{},{},{},{},{},{},{wall_us:.1}",
        p.name(),
        style_name(s),
        io.rows,
        io.bytes_read,
        io.iops,
        io.requests,
        io.counts.get("rows_scanned").copied().unwrap_or(0),
        io.counts.get("ranges_scanned").copied().unwrap_or(0)
    );
}

// ------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    const TN: u32 = 4_096;

    fn coords(r: std::ops::Range<u32>) -> Vec<u32> {
        r.collect()
    }

    /// Expected value of lane `l` at `c` (epoch `e`), as the reference.
    fn want(p: Pattern, d: f64, l: usize, c: u32, e: u32) -> Option<Val> {
        present(p, d, l, c).then(|| value(l, c, e))
    }

    fn sample(n: u32) -> Vec<u32> {
        (0..64).map(|j| (mix(0x5A ^ j) % n as u64) as u32).collect()
    }

    async fn check(ds: &Dataset, map: &CoordMap, cs: &[u32], epoch_of: impl Fn(u32) -> u32) {
        let lanes = [0usize, 3];
        let (got, _) = masked_read(ds, map, cs, &lanes, MaterializationStyle::Heuristic, true)
            .await
            .unwrap();
        for (&c, vals) in &got {
            assert!(cs.contains(&c), "coord {c} was not requested");
            for (j, &l) in lanes.iter().enumerate() {
                assert_eq!(
                    vals[j],
                    want(Pattern::Uniform, 0.5, l, c, epoch_of(c)),
                    "lane {l} at {c}"
                );
            }
        }
        for &c in cs {
            if map.id_of.contains_key(&c) {
                assert!(got.contains_key(&c), "coord {c} missing from the read");
            }
        }
    }

    /// Oracle 8: combined and independent masked reads equal the reference.
    #[tokio::test]
    async fn masked_reads_equal_the_reference() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("t");
        let uri = uri.to_str().unwrap();
        let ds = write(
            uri,
            vec![batch(&coords(0..TN), Pattern::Uniform, 0.5, 1)],
            WriteMode::Create,
            false,
            1 << 20,
        )
        .await;
        let map = build_map(&ds).await;
        let lanes = [1usize, 2];
        let joint: Vec<u32> = (0..TN)
            .filter(|&c| lanes.iter().all(|&l| present(Pattern::Uniform, 0.5, l, c)))
            .collect();
        assert!(
            joint.len() > 500 && joint.len() < 2000,
            "joint population {}",
            joint.len()
        );
        let (got, io) = masked_read(
            &ds,
            &map,
            &joint,
            &lanes,
            MaterializationStyle::Heuristic,
            true,
        )
        .await
        .unwrap();
        assert_eq!(io.rows, joint.len());
        for &c in &joint {
            for (j, &l) in lanes.iter().enumerate() {
                assert_eq!(got[&c][j], want(Pattern::Uniform, 0.5, l, c, 1));
                assert!(got[&c][j].is_some());
            }
        }
        for &l in &lanes {
            let own: Vec<u32> = (0..TN)
                .filter(|&c| present(Pattern::Uniform, 0.5, l, c))
                .collect();
            let (got, _) = masked_read(&ds, &map, &own, &[l], MaterializationStyle::AllLate, true)
                .await
                .unwrap();
            assert_eq!(got.len(), own.len());
            for &c in &own {
                assert_eq!(got[&c][0], want(Pattern::Uniform, 0.5, l, c, 1));
            }
        }
    }

    /// Oracle 9: attribution is by stored coordinate; reversing the batch
    /// leaves it unchanged, while positional attribution would not.
    #[tokio::test]
    async fn read_order_cannot_change_identity() {
        let cs = coords(0..256);
        let b = batch(&cs, Pattern::Uniform, 0.5, 1);
        // the same rows in reverse order
        let rev = batch(
            &cs.iter().rev().copied().collect::<Vec<_>>(),
            Pattern::Uniform,
            0.5,
            1,
        );
        assert_eq!(align(&b, &[0, 1]), align(&rev, &[0, 1]));
        // positional attribution of the reversed batch disagrees
        let lane0 = rev
            .column_by_name("lane_0")
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        let positional_wrong = (0..256)
            .filter(|&j| {
                let by_pos: Option<Val> =
                    (!lane0.is_null(j)).then(|| lane0.value(j).try_into().unwrap());
                by_pos != want(Pattern::Uniform, 0.5, 0, j as u32, 1)
            })
            .count();
        assert!(
            positional_wrong > 100,
            "the fixture must make position and coord disagree"
        );
    }

    /// Oracle 10 and § 8: every identity scenario returns the requested
    /// coordinates or refuses; none returns a wrong coordinate.
    #[tokio::test]
    async fn identity_survives_or_refuses_across_rewrites() {
        for stable in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let uri = dir.path().join("id");
            let uri = uri.to_str().unwrap();
            // initial append (two fragments of 1,024 rows)
            let mut ds = write(
                uri,
                vec![batch(&coords(0..2048), Pattern::Uniform, 0.5, 1)],
                WriteMode::Create,
                stable,
                1024,
            )
            .await;
            let m1 = build_map(&ds).await;
            check(&ds, &m1, &sample(2048), |_| 1).await;
            // second append
            ds = write(
                uri,
                vec![batch(&coords(2048..TN), Pattern::Uniform, 0.5, 1)],
                WriteMode::Append,
                stable,
                1024,
            )
            .await;
            assert!(matches!(
                masked_read(&ds, &m1, &[1], &[0], MaterializationStyle::Heuristic, true).await,
                Err(ReadError::StaleMapping { .. })
            ));
            let m2 = build_map(&ds).await;
            check(&ds, &m2, &sample(TN), |_| 1).await;
            // delete
            ds.delete("coord % 7 = 0").await.unwrap();
            let m3 = build_map(&ds).await;
            assert!(!m3.id_of.contains_key(&7));
            let (got, _) = masked_read(
                &ds,
                &m3,
                &[7, 8],
                &[0],
                MaterializationStyle::Heuristic,
                true,
            )
            .await
            .unwrap();
            assert!(!got.contains_key(&7) && got.contains_key(&8));
            check(&ds, &m3, &sample(TN), |_| 1).await;
            // update: epoch 2 for coord < 100, lane values rewritten in place
            let upd = lance::dataset::UpdateBuilder::new(Arc::new(ds.clone()))
                .update_where("coord < 100")
                .unwrap()
                .set("epoch", "2")
                .unwrap()
                .build()
                .unwrap()
                .execute()
                .await
                .unwrap();
            ds = (*upd.new_dataset).clone();
            let m4 = build_map(&ds).await;
            let moved = (0..100u32)
                .filter(|c| m3.id_of.contains_key(c) && m3.id_of.get(c) != m4.id_of.get(c))
                .count();
            println!("stable={stable}: update moved {moved} of the updated rows' _rowid");
            check(&ds, &m4, &sample(TN), |_| 1).await; // lane values unchanged by the epoch update
                                                       // reopen: same version, the map stays valid
            let reopened = Dataset::open(uri).await.unwrap();
            assert_eq!(reopened.version().version, m4.version);
            check(&reopened, &m4, &sample(TN), |_| 1).await;
            // compaction
            lance::dataset::optimize::compact_files(&mut ds, Default::default(), None)
                .await
                .unwrap();
            assert!(matches!(
                masked_read(&ds, &m4, &[8], &[0], MaterializationStyle::Heuristic, true).await,
                Err(ReadError::StaleMapping { .. })
            ));
            let m5 = build_map(&ds).await;
            let changed = m4
                .id_of
                .iter()
                .filter(|(c, id)| m5.id_of.get(c) != Some(id))
                .count();
            println!(
                "stable={stable}: compaction changed {changed} of {} _rowid",
                m4.id_of.len()
            );
            if stable {
                assert_eq!(changed, 0, "stable row ids must survive compaction");
            }
            check(&ds, &m5, &sample(TN), |_| 1).await;
            // version transition: the old version with its own map
            let old = ds.checkout_version(m2.version).await.unwrap();
            check(&old, &m2, &sample(TN), |_| 1).await;
            // same-row-count rewrite: overwrite with the coordinates reversed
            let rev: Vec<u32> = (0..TN).rev().collect();
            ds = write(
                uri,
                vec![batch(&rev, Pattern::Uniform, 0.5, 1)],
                WriteMode::Overwrite,
                stable,
                1024,
            )
            .await;
            assert!(matches!(
                masked_read(&ds, &m5, &[8], &[0], MaterializationStyle::Heuristic, true).await,
                Err(ReadError::StaleMapping { .. })
            ));
            let m6 = build_map(&ds).await;
            check(&ds, &m6, &sample(TN), |_| 1).await;
        }
    }

    /// The version check is load-bearing: after a reordering rewrite the old
    /// map, used without the check, attributes rows to the wrong coordinates.
    #[tokio::test]
    async fn an_unchecked_stale_map_returns_a_wrong_answer() {
        let asked: Vec<u32> = (0..TN).step_by(17).collect();
        let mut outcomes = Vec::new();
        for (name, stable) in [
            ("overwrite-reversed", false),
            ("delete+compact", false),
            ("update", false),
            ("overwrite-reversed", true),
            ("delete+compact", true),
            ("update", true),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let uri = dir.path().join("stale");
            let uri = uri.to_str().unwrap();
            let mut ds = write(
                uri,
                vec![batch(&coords(0..TN), Pattern::Uniform, 0.5, 1)],
                WriteMode::Create,
                stable,
                1024,
            )
            .await;
            let m1 = build_map(&ds).await;
            match name {
                "overwrite-reversed" => {
                    let rev: Vec<u32> = (0..TN).rev().collect();
                    ds = write(
                        uri,
                        vec![batch(&rev, Pattern::Uniform, 0.5, 1)],
                        WriteMode::Overwrite,
                        stable,
                        1024,
                    )
                    .await;
                }
                "delete+compact" => {
                    ds.delete("coord % 3 = 0").await.unwrap();
                    lance::dataset::optimize::compact_files(&mut ds, Default::default(), None)
                        .await
                        .unwrap();
                }
                _ => {
                    let upd = lance::dataset::UpdateBuilder::new(Arc::new(ds.clone()))
                        .update_where("coord < 2000")
                        .unwrap()
                        .set("epoch", "2")
                        .unwrap()
                        .build()
                        .unwrap()
                        .execute()
                        .await
                        .unwrap();
                    ds = (*upd.new_dataset).clone();
                }
            }
            let fresh = build_map(&ds).await;
            let want: Vec<u32> = asked
                .iter()
                .copied()
                .filter(|c| fresh.id_of.contains_key(c))
                .collect();
            let (got, _) = masked_read(
                &ds,
                &m1,
                &asked,
                &[0],
                MaterializationStyle::Heuristic,
                false,
            )
            .await
            .unwrap();
            let wrong = got.keys().filter(|c| !asked.contains(c)).count();
            let missing = want.iter().filter(|c| !got.contains_key(c)).count();
            let extra_deleted = got
                .keys()
                .filter(|c| asked.contains(c) && !fresh.id_of.contains_key(c))
                .count();
            println!("stale map, {name}, stable={stable}: asked {}, live {}, returned {}, wrong coord {wrong}, silently missing {missing}, deleted returned {extra_deleted}", asked.len(), want.len(), got.len());
            outcomes.push(wrong + missing + extra_deleted);
            assert!(matches!(
                masked_read(
                    &ds,
                    &m1,
                    &asked,
                    &[0],
                    MaterializationStyle::Heuristic,
                    true
                )
                .await,
                Err(ReadError::StaleMapping { .. })
            ));
        }
        assert!(
            outcomes.iter().any(|&n| n > 0),
            "the hazard must be real for the version check to matter"
        );
    }

    /// A missing lane column or a missing lane dataset fails explicitly.
    #[tokio::test]
    async fn a_missing_lane_fails_explicitly() {
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().join("m");
        let uri = uri.to_str().unwrap();
        let ds = write(
            uri,
            vec![batch(&coords(0..64), Pattern::Uniform, 0.5, 1)],
            WriteMode::Create,
            false,
            1 << 20,
        )
        .await;
        let map = build_map(&ds).await;
        assert!(matches!(
            masked_read(
                &ds,
                &map,
                &[1],
                &[LANES + 3],
                MaterializationStyle::Heuristic,
                true
            )
            .await,
            Err(ReadError::Lance(_))
        ));
        assert!(Dataset::open(dir.path().join("absent").to_str().unwrap())
            .await
            .is_err());
    }
}
