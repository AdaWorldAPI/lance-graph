//! `VersionedAppend` over a caller's own records (EXO-483): one batch per Lance commit, a
//! stale writer refused before it writes, the logical version kept in the caller's records.
//!
//! The records here are a document version log, the shape report-rs's versioned documents
//! need (`report-egress-convergence-v1.md` §12.1): document, logical version `n`, content
//! hash, parent `n`. Nothing in the writer knows that shape.

use std::sync::Arc;

use arrow_array::{Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;
use lance_graph_dir_sim_lance::{AppendError, VersionedAppend};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("doc", DataType::Utf8, false),
        Field::new("n", DataType::UInt64, false),
        Field::new("sha", DataType::Utf8, false),
        Field::new("parent", DataType::UInt64, true),
    ]))
}

/// One version row of document `doc`.
fn version(doc: &str, n: u64, sha: &str, parent: Option<u64>) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(vec![doc])),
            Arc::new(UInt64Array::from(vec![n])),
            Arc::new(StringArray::from(vec![sha])),
            Arc::new(UInt64Array::from(vec![parent])),
        ],
    )
    .unwrap()
}

fn uri(dir: &tempfile::TempDir) -> String {
    dir.path().join("docs.lance").to_string_lossy().into_owned()
}

/// The logical versions stored, in commit order.
async fn stored_ns(uri: &str) -> Vec<u64> {
    let ds = Dataset::open(uri).await.unwrap();
    let all = ds.scan().try_into_batch().await.unwrap();
    let n = all
        .column_by_name("n")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    (0..n.len()).map(|i| n.value(i)).collect()
}

/// EXO-483's test: two writers at the same version; the first lands, the second is refused
/// and writes nothing; after reopening, the second lands as the next version.
#[tokio::test]
async fn a_second_writer_at_the_same_version_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let mut seed = VersionedAppend::at(&uri, None);
    assert_eq!(seed.append(version("K01-a", 1, "aa", None)).await.unwrap(), 1);

    // Both writers read the dataset at version 1.
    let mut first = VersionedAppend::open(&uri).await.unwrap();
    let mut second = VersionedAppend::open(&uri).await.unwrap();
    assert_eq!((first.version(), second.version()), (Some(1), Some(1)));

    assert_eq!(first.append(version("K01-a", 2, "bb", Some(1))).await.unwrap(), 2);
    match second.append(version("K01-a", 2, "cc", Some(1))).await {
        Err(AppendError::StaleWriter { expected, found }) => {
            assert_eq!((expected, found), (Some(1), Some(2)));
        }
        other => panic!("the second writer must be refused: {other:?}"),
    }
    assert_eq!(second.version(), Some(1), "a refused writer does not advance");
    assert_eq!(stored_ns(&uri).await, vec![1, 2], "nothing of the refused append landed");

    // Reopened, the second writer sees version 2 and lands as version 3.
    let mut second = VersionedAppend::open(&uri).await.unwrap();
    assert_eq!(second.append(version("K01-a", 3, "cc", Some(2))).await.unwrap(), 3);
    assert_eq!(stored_ns(&uri).await, vec![1, 2, 3]);
}

/// A writer that has seen no dataset creates it; a second such writer is refused.
#[tokio::test]
async fn only_one_writer_creates_the_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let mut a = VersionedAppend::at(&uri, None);
    let mut b = VersionedAppend::at(&uri, None);
    assert_eq!(a.append(version("K01-a", 1, "aa", None)).await.unwrap(), 1);
    assert!(matches!(
        b.append(version("K02-a", 1, "dd", None)).await,
        Err(AppendError::StaleWriter {
            expected: None,
            found: Some(1)
        })
    ));
    assert_eq!(stored_ns(&uri).await, vec![1]);
}

/// The two clocks stay apart: the writer returns the physical commit counter, while the
/// logical version is whatever the caller wrote. Two documents both at logical version 1
/// are two commits, 1 and 2.
#[tokio::test]
async fn the_logical_version_lives_in_the_records() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let mut w = VersionedAppend::at(&uri, None);
    assert_eq!(w.append(version("K01-a", 1, "aa", None)).await.unwrap(), 1);
    assert_eq!(w.append(version("K02-a", 1, "dd", None)).await.unwrap(), 2);
    assert_eq!(stored_ns(&uri).await, vec![1, 1]);
}

/// A batch with another schema than the dataset's is refused by Lance and nothing lands.
#[tokio::test]
async fn a_batch_of_another_shape_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let uri = uri(&dir);
    let mut w = VersionedAppend::at(&uri, None);
    w.append(version("K01-a", 1, "aa", None)).await.unwrap();
    let other = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::UInt64, false)])),
        vec![Arc::new(UInt64Array::from(vec![7u64]))],
    )
    .unwrap();
    assert!(matches!(w.append(other).await, Err(AppendError::Lance(_))));
    assert_eq!(w.version(), Some(1));
    assert_eq!(stored_ns(&uri).await, vec![1]);
}
