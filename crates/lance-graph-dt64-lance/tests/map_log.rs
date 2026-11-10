//! A DR map table survives a round trip through its Lance dataset exactly:
//! every past route version answers as before, one advance is one commit,
//! a past commit reads the table as of then, a stale writer is refused, and
//! route versions are never confused with dataset versions.

use lance_graph_dt64::dr::{FrameRefusal, LocalMap, MapRead, MapTable, RouteVersion, Width};
use lance_graph_dt64_lance::{LanceMapLog, StoreError};

const TILES: [u32; 3] = [1, 2, 3];

fn map(w: Width, ids: &[u64]) -> Option<LocalMap> {
    Some(LocalMap::prepare(w, ids).unwrap())
}

/// What a table answers for `tile` at `version`: `(width, globals)`, or the
/// refusal.
fn answer(t: &MapTable, version: u64, tile: u32) -> Result<(u16, Vec<u64>), MapRead> {
    let d = t.diagnose_at(RouteVersion(version), tile)?;
    let width = match d.width() {
        Width::W64 => 64,
        Width::W256 => 256,
    };
    let globals = (0..d.len())
        .map(|l| d.global(u8::try_from(l).unwrap()).unwrap())
        .collect();
    Ok((width, globals))
}

/// Every version up to `upto`, every tile: the two tables answer alike.
fn assert_same(a: &MapTable, b: &MapTable, upto: u64) {
    for v in 0..=upto {
        for tile in TILES {
            assert_eq!(
                answer(a, v, tile),
                answer(b, v, tile),
                "version {v} tile {tile}"
            );
        }
    }
}

/// Six advances covering every kind of history event.
async fn history(log: &mut LanceMapLog, t: &mut MapTable) {
    let steps: Vec<Vec<(u32, Option<LocalMap>)>> = vec![
        vec![
            (1, map(Width::W64, &[10, 11, 12])),
            (2, map(Width::W256, &[7, 3, 9, 1])),
        ],
        vec![(1, map(Width::W64, &[10, 11, 12, 13]))], // extension
        vec![(1, map(Width::W64, &[13, 10, 11, 12]))], // replacement (reorder)
        vec![(2, None)],                               // removal
        vec![],                                        // advance with no change
        vec![(2, map(Width::W64, &[7])), (3, map(Width::W256, &[]))], // re-add, empty map
    ];
    for (k, changes) in steps.into_iter().enumerate() {
        let v = log.advance(t, 1, 1 + k as u64, changes).await.unwrap();
        assert_eq!(v, RouteVersion(k as u64 + 1));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_restored_table_answers_every_past_version_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("map.lance").to_string_lossy().into_owned();
    let (mut t, mut log) = LanceMapLog::create(&uri, 1, 1).await.unwrap();
    history(&mut log, &mut t).await;

    let (r, rlog) = LanceMapLog::restore(&uri).await.unwrap();
    assert_eq!(r.current_version(), t.current_version());
    assert_eq!(r.revisions(), t.revisions());
    assert_same(&t, &r, t.current_version().0);
    for tile in TILES {
        assert_eq!(
            r.generation_count(tile),
            t.generation_count(tile),
            "tile {tile}"
        );
        assert_eq!(r.event_count(tile), t.event_count(tile), "tile {tile}");
    }
    // The extension is stored as an extension: tile 1 has two generations
    // (the original, extended in place, and the reorder), not three.
    assert_eq!(r.generation_count(1), 2);
    // Genesis plus six advances: seven commits.
    assert_eq!(log.dataset_version(), Some(7));
    assert_eq!(rlog.dataset_version(), Some(7));
}

/// Lance's own version history is the DR history: dataset version `k` holds
/// genesis and the first `k - 1` advances, and answers each of them as the
/// live table does.
#[tokio::test(flavor = "multi_thread")]
async fn each_past_commit_reads_the_table_as_of_then() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("map.lance").to_string_lossy().into_owned();
    let (mut t, mut log) = LanceMapLog::create(&uri, 1, 1).await.unwrap();
    history(&mut log, &mut t).await;
    for k in 1..=7u64 {
        let past = LanceMapLog::restore_at(&uri, k).await.unwrap();
        assert_eq!(
            past.current_version(),
            RouteVersion(k - 1),
            "dataset version {k}"
        );
        assert_same(&t, &past, k - 1);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stale_writer_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("map.lance").to_string_lossy().into_owned();
    let (mut t, mut log) = LanceMapLog::create(&uri, 1, 1).await.unwrap();
    let (mut t2, mut log2) = LanceMapLog::restore(&uri).await.unwrap();
    log.advance(&mut t, 1, 1, vec![(1, map(Width::W64, &[1, 2]))])
        .await
        .unwrap();

    let before = t2.current_version();
    let err = log2
        .advance(&mut t2, 1, 1, vec![(1, map(Width::W64, &[5]))])
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        StoreError::StaleWriter {
            expected: Some(1),
            found: Some(2)
        }
    ));
    assert_eq!(t2.current_version(), before, "the table was not advanced");
    let (r, _) = LanceMapLog::restore(&uri).await.unwrap();
    assert_eq!(
        answer(&r, 1, 1),
        Ok((64, vec![1, 2])),
        "only the first writer's advance is stored"
    );
}

/// A table whose clock is shared (here: ticked from a clone, as a RouteLog
/// sharing it would) issues route versions with gaps. They are stored and
/// replayed as route versions, independent of the dataset version.
#[tokio::test(flavor = "multi_thread")]
async fn route_versions_with_gaps_replay_as_route_versions() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("map.lance").to_string_lossy().into_owned();
    let (mut t, mut log) = LanceMapLog::create(&uri, 1, 1).await.unwrap();
    let shared = t.clock();
    for _ in 0..5 {
        shared.tick();
    }
    let v = log
        .advance(&mut t, 1, 1, vec![(1, map(Width::W64, &[42]))])
        .await
        .unwrap();
    assert_eq!(v, RouteVersion(6));
    assert_eq!(log.dataset_version(), Some(2), "two commits, not six");

    let (r, _) = LanceMapLog::restore(&uri).await.unwrap();
    assert_eq!(r.current_version(), RouteVersion(6));
    assert_eq!(answer(&r, 5, 1), Err(MapRead::NoSuchTile(1)));
    assert_eq!(answer(&r, 6, 1), Ok((64, vec![42])));
}

/// Frames are witnesses of one table instance, so a frame sealed before a
/// restore is foreign to the restored table.
#[tokio::test(flavor = "multi_thread")]
async fn a_frame_from_before_a_restore_is_foreign() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("map.lance").to_string_lossy().into_owned();
    let (mut t, mut log) = LanceMapLog::create(&uri, 1, 1).await.unwrap();
    log.advance(&mut t, 1, 1, vec![(1, map(Width::W64, &[1]))])
        .await
        .unwrap();
    let frame = t.seal();
    assert!(t.map(&frame, 1).is_ok());
    let (r, _) = LanceMapLog::restore(&uri).await.unwrap();
    assert!(matches!(
        r.map(&frame, 1),
        Err(MapRead::Stale(FrameRefusal::ForeignFrame))
    ));
    assert!(r.map(&r.seal(), 1).is_ok());
}

#[tokio::test(flavor = "multi_thread")]
async fn create_refuses_an_existing_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().join("map.lance").to_string_lossy().into_owned();
    LanceMapLog::create(&uri, 1, 1).await.unwrap();
    assert!(matches!(
        LanceMapLog::create(&uri, 1, 1).await.unwrap_err(),
        StoreError::StaleWriter {
            expected: None,
            found: Some(1)
        }
    ));
}
