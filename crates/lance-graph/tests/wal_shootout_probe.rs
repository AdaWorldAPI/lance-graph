//! D-WAL-1 — validation of `LanceCycleWriter` (arm L) against the lancedb 0.40
//! MemWAL write path (arm M). Plan: `.claude/plans/wal-shootout-v1.md` §2.
//!
//! - **V1** SIGKILL a child mid-write: recovered rows per cycle ∈ {0, all}.
//! - **V2** every cycle the child acknowledged before the kill is readable.
//! - **V3** resubmitting the same frozen cycle adds no rows.
//! - **V4** what a second reader sees while a cycle is acknowledged but not
//!   flushed (observed behaviour, printed, not a pass/fail gate).
//!
//! M is read two ways after a crash: (a) a reader that has not written, (b) the
//! same reader after its first write, which opens the shard writer and replays
//! the WAL. Lance local filesystem only; a process crash, not a power loss.
//!
//! Run: `cargo test -p lance-graph --features lancedb-sdk --test wal_shootout_probe -- --nocapture --test-threads=1`

mod wal_common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

use lance_graph::graph::cycle_sink::LanceCycleWriter;
use lance_graph_planner::persist_sink::{CommitOutcome, WalSink};
use wal_common::*;

const CHILD_ENV: &str = "WAL_SHOOTOUT_CHILD";
/// V1 cycle size: 4,096 casts → 8,193 rows, about 2 MB of payload.
const V1_N: usize = 4096;
/// V2 cycle size.
const V2_N: usize = 1024;
/// WAL flush interval for M in these tests.
const FLUSH_MS: u64 = 10;

fn l_path(dir: &str) -> String {
    format!("{dir}/cycles.lance")
}

// ───────────────────────────── child mode ─────────────────────────────

/// Child body. `WAL_SHOOTOUT_CHILD=<v1|v2>:<L|M>:<dir>`; a no-op otherwise.
#[tokio::test(flavor = "multi_thread")]
async fn wal_child_body() {
    let Ok(spec) = std::env::var(CHILD_ENV) else {
        return;
    };
    let mut it = spec.splitn(3, ':');
    let (mode, arm, dir) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
    let cycles: u64 = if mode == "v1" { 1 } else { 10_000 };
    let n = if mode == "v1" { V1_N } else { V2_N };
    let mut out = std::io::stdout();
    match arm {
        "L" => {
            let mut w = LanceCycleWriter::open(l_path(dir)).await.expect("open");
            for c in 1..=cycles {
                l_commit(&mut w, c, n).await;
                writeln!(out, "ACK {c}").unwrap();
                out.flush().unwrap();
            }
        }
        "M" => {
            let t = m_open(dir).await;
            for c in 1..=cycles {
                m_put(&t, &frozen(c, 0, n)).await;
                writeln!(out, "ACK {c}").unwrap();
                out.flush().unwrap();
            }
        }
        other => panic!("arm {other}"),
    }
    std::thread::sleep(Duration::from_secs(30));
}

fn spawn_child(mode: &str, arm: &str, dir: &str, stdout: Stdio) -> std::process::Child {
    Command::new(std::env::current_exe().expect("current_exe"))
        .args(["--exact", "wal_child_body", "--nocapture"])
        .env(CHILD_ENV, format!("{mode}:{arm}:{dir}"))
        .stdout(stdout)
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn child")
}

/// After a crash: (L) rows per cycle; (M) rows per cycle before and after the
/// reader's first write (sentinel cycle `u64::MAX >> 8`, excluded from the map).
async fn recovered(arm: &str, dir: &str) -> (std::collections::BTreeMap<u64, usize>, Option<std::collections::BTreeMap<u64, usize>>) {
    match arm {
        "L" => (l_rows_by_cycle(&l_path(dir)).await.unwrap_or_default(), None),
        _ => {
            let t = m_open(dir).await;
            let before = m_rows_by_cycle(&t, true).await;
            let sentinel = u64::MAX >> 8;
            m_put(&t, &frozen(sentinel, 0, 1)).await;
            let mut after = m_rows_by_cycle(&t, true).await;
            after.remove(&sentinel);
            (before, Some(after))
        }
    }
}

async fn prepare(arm: &str, dir: &str) {
    if arm == "M" {
        m_create(dir, FLUSH_MS).await;
    }
}

// ───────────────────────────── V1 ─────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn v1_sigkill_mid_write_leaves_all_or_nothing() {
    if std::env::var(CHILD_ENV).is_ok() {
        return;
    }
    let full = rows_per_cycle(V1_N);
    for arm in ["L", "M"] {
        let mut observed = Vec::new();
        for delay_ms in [5u64, 30, 60, 120, 250, 500, 1000, 2000] {
            let tmp = tempfile::tempdir().expect("tempdir");
            let dir = tmp.path().to_string_lossy().to_string();
            prepare(arm, &dir).await;
            let mut child = spawn_child("v1", arm, &dir, Stdio::null());
            std::thread::sleep(Duration::from_millis(delay_ms));
            child.kill().expect("kill");
            let _ = child.wait();
            let (a, b) = recovered(arm, &dir).await;
            let ra = a.get(&1).copied().unwrap_or(0);
            let rb = b.as_ref().map(|m| m.get(&1).copied().unwrap_or(0));
            eprintln!("V1 {arm} kill@{delay_ms}ms recovered: read={ra} after-replay={rb:?} (full {full})");
            observed.push((delay_ms, ra, rb));
        }
        let partial: Vec<_> = observed
            .iter()
            .filter(|(_, a, b)| {
                (*a != 0 && *a != full) || b.is_some_and(|b| b != 0 && b != full)
            })
            .collect();
        let landed = observed.iter().any(|(_, a, b)| *a > 0 || b.is_some_and(|b| b > 0));
        let empty = observed.iter().any(|(_, a, b)| *a == 0 && b.unwrap_or(0) == 0);
        if !landed || !empty {
            eprintln!(
                "V1 {arm} INCONCLUSIVE: the sweep did not land on both sides of the \
                 durability boundary (landed={landed}, empty={empty}); widen it."
            );
        }
        assert!(partial.is_empty(), "V1 {arm}: partial cycle after SIGKILL: {partial:?}");
    }
}

// ───────────────────────────── V2 ─────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn v2_acknowledged_cycles_survive_sigkill() {
    if std::env::var(CHILD_ENV).is_ok() {
        return;
    }
    let full = rows_per_cycle(V2_N);
    for arm in ["L", "M"] {
        for target in [2u64, 5, 9] {
            let tmp = tempfile::tempdir().expect("tempdir");
            let dir = tmp.path().to_string_lossy().to_string();
            prepare(arm, &dir).await;
            let mut child = spawn_child("v2", arm, &dir, Stdio::piped());
            let mut acked = 0u64;
            {
                let reader = BufReader::new(child.stdout.take().expect("stdout"));
                for line in reader.lines() {
                    let line = line.expect("line");
                    if let Some(k) = line.strip_prefix("ACK ") {
                        acked = k.trim().parse().expect("ack");
                        if acked >= target {
                            break;
                        }
                    }
                }
            }
            child.kill().expect("kill");
            let _ = child.wait();
            let (a, b) = recovered(arm, &dir).await;
            let lost = |m: &std::collections::BTreeMap<u64, usize>| -> Vec<u64> {
                (1..=acked).filter(|c| m.get(c).copied() != Some(full)).collect()
            };
            let partial = |m: &std::collections::BTreeMap<u64, usize>| -> Vec<(u64, usize)> {
                m.iter().filter(|(_, &v)| v != full).map(|(&k, &v)| (k, v)).collect()
            };
            eprintln!(
                "V2 {arm} acked={acked}: read sees cycles {:?} (missing acked {:?}); after replay {:?}",
                a.keys().collect::<Vec<_>>(),
                lost(&a),
                b.as_ref().map(|m| (m.keys().copied().collect::<Vec<_>>(), lost(m))),
            );
            assert!(partial(&a).is_empty(), "V2 {arm}: partial cycle {:?}", partial(&a));
            match &b {
                None => assert!(lost(&a).is_empty(), "V2 L lost acknowledged cycles {:?}", lost(&a)),
                Some(b) => {
                    assert!(partial(b).is_empty(), "V2 M: partial cycle after replay {:?}", partial(b));
                    assert!(lost(b).is_empty(), "V2 M lost acknowledged cycles after replay {:?}", lost(b));
                }
            }
        }
    }
}

// ───────────────────────────── V3 ─────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn v3_resubmitting_a_frozen_cycle_adds_no_rows() {
    if std::env::var(CHILD_ENV).is_ok() {
        return;
    }
    let n = 64;
    // L
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_string_lossy().to_string();
    let mut w = LanceCycleWriter::open(l_path(&dir)).await.unwrap();
    let batch = frozen(1, w.head().0, n);
    let first = w.commit_cycle(batch.clone()).await.unwrap();
    let second = w.commit_cycle(batch).await.unwrap();
    let rows = l_rows_by_cycle(&l_path(&dir)).await.unwrap();
    eprintln!("V3 L first={first:?} second={second:?} rows={rows:?}");
    assert!(matches!(first, CommitOutcome::Committed { .. }));
    assert!(matches!(second, CommitOutcome::Reconciled { .. }));
    assert_eq!(rows.get(&1), Some(&rows_per_cycle(n)));
    drop(w);
    // M
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_string_lossy().to_string();
    let t = m_create(&dir, FLUSH_MS).await;
    let batch = frozen(1, 0, n);
    m_put(&t, &batch).await;
    m_put(&t, &batch).await;
    let rows = m_rows_by_cycle(&t, true).await;
    eprintln!("V3 M rows after two puts of one cycle={rows:?}");
    assert_eq!(rows.get(&1), Some(&rows_per_cycle(n)));
}

// ───────────────────────────── V4 ─────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn v4_read_horizon_while_a_cycle_is_unflushed() {
    if std::env::var(CHILD_ENV).is_ok() {
        return;
    }
    let n = 64;
    // L: a reader opened at cycle 1 stays there; a fresh reader sees cycle 2.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_string_lossy().to_string();
    let path = l_path(&dir);
    let mut w = LanceCycleWriter::open(path.clone()).await.unwrap();
    l_commit(&mut w, 1, n).await;
    let pinned = lance::dataset::Dataset::open(&path).await.unwrap();
    l_commit(&mut w, 2, n).await;
    let pinned_rows = pinned.count_rows(None).await.unwrap();
    let fresh = l_rows_by_cycle(&path).await.unwrap();
    eprintln!("V4 L pinned reader rows={pinned_rows} fresh reader cycles={:?}", fresh.keys().collect::<Vec<_>>());
    assert_eq!(pinned_rows, rows_per_cycle(n));
    assert_eq!(fresh.len(), 2);
    drop(w);
    // M: writer handle A, second handle B in the same process (own writer cache).
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_string_lossy().to_string();
    let a = m_create(&dir, FLUSH_MS).await;
    m_put(&a, &frozen(1, 0, n)).await;
    m_put(&a, &frozen(2, 0, n)).await;
    let b = m_open(&dir).await;
    let a_lsm = m_rows_by_cycle(&a, true).await;
    let b_lsm = m_rows_by_cycle(&b, true).await;
    let b_base = m_rows_by_cycle(&b, false).await;
    eprintln!(
        "V4 M acked 2 cycles: writer LSM {:?} | second handle LSM {:?} | base-only {:?}",
        a_lsm, b_lsm, b_base
    );
    a.close_lsm_writers().await.unwrap();
    let b2 = m_open(&dir).await;
    eprintln!(
        "V4 M after close_lsm_writers: second handle LSM {:?} | base-only {:?}",
        m_rows_by_cycle(&b2, true).await,
        m_rows_by_cycle(&b2, false).await
    );
    assert_eq!(a_lsm.len(), 2, "the writer's own LSM read must see what it acknowledged");
}
