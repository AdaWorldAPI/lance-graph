//! The log stored in Lance: one batch per append, restored in one read,
//! a stale writer refused, a torn batch refused at restore.

use lance_graph_dir_sim::image::{RuleRegistry, StoredImage};
use lance_graph_dir_sim::log::{LogState, ReplayError};
use lance_graph_dir_sim::{Observation, ObservedNode, VersionStore};
use lance_graph_dir_sim_lance::{LanceLog, StoreError};
use ogar_dir_core::{DirectoryScope, Guid128};
use ogar_dir_sim::VersionId;

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));

fn g(n: u8) -> Guid128 {
    Guid128([n; 16])
}

fn observation(users: &[(u8, &str)]) -> Observation {
    Observation {
        scope: SCOPE,
        nodes: users
            .iter()
            .map(|&(n, a)| (g(n), ObservedNode::user(a, a)))
            .collect(),
        members: vec![],
    }
}

/// A store with one observation, and the versions it holds.
fn first() -> (VersionStore, Vec<VersionId>) {
    let mut st = VersionStore::new();
    let v = st
        .observe("lab", 1, observation(&[(0xA1, "alice@x.test")]))
        .unwrap();
    (st, vec![v])
}

fn second(st: &mut VersionStore, vs: &mut Vec<VersionId>) {
    let v = st
        .observe(
            "lab",
            2,
            observation(&[(0xA1, "alice@x.test"), (0xB0, "bob@x.test")]),
        )
        .unwrap();
    vs.push(v);
}

fn image(st: &VersionStore) -> StoredImage {
    StoredImage::from(st.image())
}

fn uri(dir: &tempfile::TempDir) -> String {
    dir.path().join("log.lance").to_str().unwrap().to_string()
}

// Two saves, each one append; the dataset restores to a store that reads
// the same, and the physical version is the commit count, not a logical
// version (the store holds two logical versions after the second save,
// but one save could carry any number).
#[tokio::test]
async fn two_saves_restore_to_the_same_reads() {
    let dir = tempfile::tempdir().unwrap();
    let (mut st, mut vs) = first();
    let mut log = LogState::new();
    let mut w = LanceLog::new(uri(&dir));

    let b = log.batch(&image(&st)).unwrap().unwrap();
    assert_eq!(w.append(&mut log, &b).await.unwrap(), 1);
    second(&mut st, &mut vs);
    let b = log.batch(&image(&st)).unwrap().unwrap();
    assert_eq!(w.append(&mut log, &b).await.unwrap(), 2);

    let (stored, state, w2) = LanceLog::restore(uri(&dir)).await.unwrap();
    assert_eq!(w2.version(), Some(2));
    assert_eq!(state, log, "the restored log continues where the writer is");
    let back = VersionStore::restore_stored(stored, &RuleRegistry::new()).unwrap();
    assert_eq!(back.diff(vs[0], vs[1]), st.diff(vs[0], vs[1]));
    assert_eq!(back.version(vs[1]), st.version(vs[1]));
    // Nothing new: no batch, no append.
    assert!(log.batch(&image(&st)).unwrap().is_none());
}

// A missing dataset is an empty log.
#[tokio::test]
async fn a_missing_dataset_is_an_empty_log() {
    let dir = tempfile::tempdir().unwrap();
    let (stored, state, w) = LanceLog::restore(uri(&dir)).await.unwrap();
    assert!(stored.versions.is_empty());
    assert_eq!(state, LogState::new());
    assert_eq!(w.version(), None);
}

// A writer whose view of the dataset is out of date writes nothing: not on
// create (the dataset exists) and not on append (it moved on).
#[tokio::test]
async fn a_stale_writer_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (mut st, mut vs) = first();
    let mut log = LogState::new();
    let mut a = LanceLog::new(uri(&dir));
    let b0 = log.batch(&image(&st)).unwrap().unwrap();

    // A second writer that also believes the log is empty.
    let mut late = LanceLog::new(uri(&dir));
    let mut late_log = LogState::new();
    a.append(&mut log, &b0).await.unwrap();
    assert!(matches!(
        late.append(&mut late_log, &b0).await,
        Err(StoreError::StaleWriter {
            expected: None,
            found: Some(1)
        })
    ));
    assert_eq!(
        late_log,
        LogState::new(),
        "the refused writer's log did not move"
    );

    // A writer restored at version 1, overtaken by `a` at version 2.
    let (_, mut old_log, mut old) = LanceLog::restore(uri(&dir)).await.unwrap();
    second(&mut st, &mut vs);
    let b1 = log.batch(&image(&st)).unwrap().unwrap();
    let old_b1 = old_log.batch(&image(&st)).unwrap().unwrap();
    a.append(&mut log, &b1).await.unwrap();
    assert!(matches!(
        old.append(&mut old_log, &old_b1).await,
        Err(StoreError::StaleWriter {
            expected: Some(1),
            found: Some(2)
        })
    ));

    // Only `a`'s two batches are stored.
    let (stored, _, w) = LanceLog::restore(uri(&dir)).await.unwrap();
    assert_eq!(w.version(), Some(2));
    assert_eq!(stored.versions.len(), 2);
}

// A batch stored with a record missing is refused at restore, naming the
// batch; nothing is returned.
#[tokio::test]
async fn a_torn_batch_is_refused_at_restore() {
    let dir = tempfile::tempdir().unwrap();
    let (st, _) = first();
    let mut log = LogState::new();
    let mut w = LanceLog::new(uri(&dir));
    let mut b = log.batch(&image(&st)).unwrap().unwrap();
    b.records.pop();
    w.append(&mut log, &b).await.unwrap();
    assert!(matches!(
        LanceLog::restore(uri(&dir)).await,
        Err(StoreError::Replay(ReplayError::RecordCount {
            batch: 0,
            ..
        }))
    ));
}

// A batch built against another log state is refused before anything is
// written.
#[tokio::test]
async fn a_batch_out_of_order_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let (st, _) = first();
    let mut log = LogState::new();
    let mut w = LanceLog::new(uri(&dir));
    let b0 = log.batch(&image(&st)).unwrap().unwrap();
    w.append(&mut log, &b0).await.unwrap();
    assert!(matches!(
        w.append(&mut log, &b0).await,
        Err(StoreError::Log(_))
    ));
    assert_eq!(w.version(), Some(1));
}
