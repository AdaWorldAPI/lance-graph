//! Licences as DR64 planes over a version's users (`license::LicensePlanes`).

use lance_graph_dir_sim::license::{LicenseError, LicensePlanes, Via, MAX_LICENSE_STATES};
use lance_graph_dir_sim::{Observation, ObservedNode, VersionStore};
use ogar_az::license::{ingest_licenses_page, ingest_skus_page, LicensesPage, PathState};
use ogar_dir_core::{DirectoryScope, Guid128};

const SCOPE: DirectoryScope = DirectoryScope(Guid128([0x5C; 16]));
const SKU: &str = "c7df2760-2c81-4ef7-b578-5b5392b571df";
const SKU2: &str = "4b9405b0-7788-4568-add1-99614e613b69";
const GROUP: &str = "3f2504e0-4f89-11d3-9a0c-0305e82c3301";

fn user(u: u8) -> Guid128 {
    let mut b = [0u8; 16];
    b[0] = 1;
    b[1] = u;
    Guid128(b)
}

fn store(users: u8) -> VersionStore {
    let mut o = Observation {
        scope: SCOPE,
        nodes: Vec::new(),
        members: Vec::new(),
    };
    for u in 0..users {
        let mail = format!("u{u}@example.test");
        o.nodes.push((user(u), ObservedNode::user(&mail, &mail)));
    }
    let mut st = VersionStore::new();
    st.observe("ad", 0, o).unwrap();
    st
}

fn g(s: &str) -> Guid128 {
    Guid128::parse(s).unwrap()
}

fn page(entries: &[String]) -> LicensesPage {
    ingest_licenses_page(&format!(r#"{{"value":[{}]}}"#, entries.join(","))).unwrap()
}

fn path(sku: &str, state: &str, group: Option<&str>) -> String {
    match group {
        Some(gr) => format!(r#"{{"skuId":"{sku}","state":"{state}","assignedByGroup":"{gr}"}}"#),
        None => format!(r#"{{"skuId":"{sku}","state":"{state}"}}"#),
    }
}

fn entry(u: u8, paths: Option<&[String]>) -> String {
    match paths {
        Some(p) => format!(
            r#"{{"id":"{}","licenseAssignmentStates":[{}]}}"#,
            user(u),
            p.join(",")
        ),
        None => format!(r#"{{"id":"{}"}}"#, user(u)),
    }
}

fn build(users: u8, entries: &[String]) -> Result<LicensePlanes, LicenseError> {
    let st = store(users);
    let v = st.view(st.tag("observed").unwrap()).unwrap();
    LicensePlanes::build(&v, &[page(entries)], &[])
}

/// Falsifier: a not-read user counted as read (or as holding nothing that was
/// read) moves a row between the two sets.
#[test]
fn not_read_is_not_none() {
    // u0 read with none, u1 property absent, u2 not on the page, u3 holds SKU.
    let p = build(
        4,
        &[
            entry(0, Some(&[])),
            entry(1, None),
            entry(3, Some(&[path(SKU, "Active", None)])),
        ],
    )
    .unwrap();
    assert_eq!(p.read().rows(), vec![0, 3]);
    assert_eq!(p.not_read().rows(), vec![1, 2]);
    assert_eq!(p.holders(&g(SKU)).rows(), vec![3]);
    assert_eq!(p.holder_count(&g(SKU)), 1);
    // A SKU no code names has no holder, and is not an error.
    assert!(p.holders(&g(SKU2)).rows().is_empty());
    assert_eq!(p.holder_count(&g(SKU2)), 0);
}

/// Falsifier: counting paths as holders makes u0 count twice.
#[test]
fn two_paths_are_one_holder() {
    let p = build(
        2,
        &[
            entry(
                0,
                Some(&[path(SKU, "Active", None), path(SKU, "Active", Some(GROUP))]),
            ),
            entry(1, Some(&[path(SKU, "Active", Some(GROUP))])),
        ],
    )
    .unwrap();
    assert_eq!(p.states().len(), 2, "direct and group are two codes");
    assert_eq!(p.states()[0].via, Via::Direct);
    assert_eq!(p.states()[1].via, Via::Group);
    assert_eq!(p.holders(&g(SKU)).rows(), vec![0, 1]);
    assert_eq!(p.holder_count(&g(SKU)), 2);
    assert_eq!(p.with_state(0).unwrap().rows(), vec![0]);
    assert_eq!(p.with_state(1).unwrap().rows(), vec![0, 1]);
    assert!(p.with_state(2).is_none());
}

/// Falsifier: merging states (or coercing unknown text to Active) collapses
/// the three codes.
#[test]
fn states_stay_apart_and_unknown_text_is_kept() {
    let p = build(
        3,
        &[
            entry(0, Some(&[path(SKU, "Active", None)])),
            entry(1, Some(&[path(SKU, "Disabled", None)])),
            entry(2, Some(&[path(SKU, "ActiveWithError", None)])),
        ],
    )
    .unwrap();
    let states: Vec<&PathState> = p.states().iter().map(|s| &s.state).collect();
    assert_eq!(
        states,
        [
            &PathState::Active,
            &PathState::Disabled,
            &PathState::Other("ActiveWithError".into())
        ]
    );
    // A disabled path is still a held SKU: the planes do not decide entitlement.
    assert_eq!(p.holders(&g(SKU)).rows(), vec![0, 1, 2]);
}

fn sku(i: usize) -> String {
    format!("00000000-0000-4000-8000-{i:012x}")
}

/// Falsifier: silently dropping or merging the 65th code builds planes.
#[test]
fn a_65th_state_is_refused_and_64_fit() {
    let fits: Vec<String> = (0..MAX_LICENSE_STATES)
        .map(|i| path(&sku(i), "Active", None))
        .collect();
    let p = build(1, &[entry(0, Some(&fits))]).unwrap();
    assert_eq!(p.states().len(), 64);
    assert_eq!(p.holder_count(&g(&sku(63))), 1);

    let over: Vec<String> = (0..=MAX_LICENSE_STATES)
        .map(|i| path(&sku(i), "Active", None))
        .collect();
    assert_eq!(
        build(1, &[entry(0, Some(&over))]),
        Err(LicenseError::TooManyStates {
            found: 65,
            limit: 64
        })
    );
}

#[test]
fn unknown_or_repeated_entries_are_refused() {
    assert_eq!(
        build(1, &[entry(5, Some(&[]))]),
        Err(LicenseError::UnknownUser(user(5)))
    );
    assert_eq!(
        build(1, &[entry(0, Some(&[])), entry(0, Some(&[]))]),
        Err(LicenseError::DuplicateUser(user(0)))
    );
    let st = store(1);
    let v = st.view(st.tag("observed").unwrap()).unwrap();
    let skus = ingest_skus_page(&format!(
        r#"{{"value":[
        {{"skuId":"{SKU}","consumedUnits":1,"prepaidUnits":{{"enabled":1,"suspended":0,"warning":0}}}},
        {{"skuId":"{SKU}","consumedUnits":1,"prepaidUnits":{{"enabled":1,"suspended":0,"warning":0}}}}]}}"#
    ))
    .unwrap();
    assert_eq!(
        LicensePlanes::build(&v, &[], &[skus]),
        Err(LicenseError::DuplicateSku(g(SKU)))
    );
}

#[test]
fn capacity_is_held_exact_beside_the_planes() {
    let st = store(1);
    let v = st.view(st.tag("observed").unwrap()).unwrap();
    let skus = ingest_skus_page(&format!(
        r#"{{"value":[{{"skuId":"{SKU}","consumedUnits":1200,"prepaidUnits":{{"enabled":1000,"suspended":0,"warning":0}}}}]}}"#
    ))
    .unwrap();
    let p = LicensePlanes::build(&v, &[], &[skus]).unwrap();
    let c = p.capacity(&g(SKU)).unwrap();
    assert_eq!((c.enabled, c.consumed, c.free()), (1000, 1200, -200));
    assert!(p.capacity(&g(SKU2)).is_none());
    // Capacity read, licences not: every user is not read.
    assert_eq!(p.not_read().rows(), vec![0]);
}

/// A user deleted in the version keeps its slot but is not a user: it is
/// neither read nor unread. Falsifier: the unread set taken against every
/// slot instead of the version's existing users.
#[test]
fn a_deleted_user_is_not_unread() {
    let mut st = store(3);
    let mut o = Observation {
        scope: SCOPE,
        nodes: Vec::new(),
        members: Vec::new(),
    };
    for u in [0u8, 2] {
        let mail = format!("u{u}@example.test");
        o.nodes.push((user(u), ObservedNode::user(&mail, &mail)));
    }
    let gone = st
        .view(st.tag("observed").unwrap())
        .unwrap()
        .user_ordinal(&user(1))
        .unwrap();
    let v1 = st.reobserve("ad", 1, o).unwrap();
    let v = st.view(v1).unwrap();
    assert_eq!(v.users_len(), 3, "the deleted user keeps its slot");
    assert_eq!(v.user_guid(gone), None, "and is not a user of this version");
    let p = LicensePlanes::build(&v, &[page(&[entry(0, Some(&[]))])], &[]).unwrap();
    assert_eq!(p.read().rows(), vec![0], "u0 read");
    assert_eq!(
        p.not_read().rows(),
        vec![2],
        "u2 unread; the deleted slot is in neither set"
    );
}

/// A user listed twice is refused in either order, whether or not its
/// licences were read the first time. Falsifier: duplicates tracked by the
/// read plane, which an unread first entry leaves clear.
#[test]
fn a_repeated_user_is_refused_in_either_order() {
    let read = entry(0, Some(&[path(SKU, "Active", None)]));
    let unread = entry(0, None);
    for entries in [
        [unread.clone(), read.clone()],
        [read.clone(), unread.clone()],
        [unread.clone(), unread.clone()],
    ] {
        assert_eq!(
            build(2, &entries).err(),
            Some(LicenseError::DuplicateUser(user(0))),
            "{entries:?}"
        );
    }
    // Silence half: one entry per user is admitted.
    assert!(build(2, &[unread, entry(1, None)]).is_ok());
}
