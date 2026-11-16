//! Licences as DR64 planes over a version's users (`license::LicensePlanes`).

use lance_graph_dir_sim::license::{
    Coverage, Evidence, ExplainRefusal, LicenseError, LicensePlanes, Via, MAX_LICENSE_STATES,
};
use lance_graph_dir_sim::{Observation, ObservedNode, UserOrdinal, VersionStore};
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
    let ver = st.tag("observed").unwrap();
    let v = st.view(ver).unwrap();
    LicensePlanes::build(&v, ver, vec![page(entries)], Vec::new())
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
    let ver = st.tag("observed").unwrap();
    let v = st.view(ver).unwrap();
    let skus = ingest_skus_page(&format!(
        r#"{{"value":[
        {{"skuId":"{SKU}","consumedUnits":1,"prepaidUnits":{{"enabled":1,"suspended":0,"warning":0}}}},
        {{"skuId":"{SKU}","consumedUnits":1,"prepaidUnits":{{"enabled":1,"suspended":0,"warning":0}}}}]}}"#
    ))
    .unwrap();
    assert_eq!(
        LicensePlanes::build(&v, ver, Vec::new(), vec![skus]),
        Err(LicenseError::DuplicateSku(g(SKU)))
    );
}

#[test]
fn capacity_is_held_exact_beside_the_planes() {
    let st = store(1);
    let ver = st.tag("observed").unwrap();
    let v = st.view(ver).unwrap();
    let skus = ingest_skus_page(&format!(
        r#"{{"value":[{{"skuId":"{SKU}","consumedUnits":1200,"prepaidUnits":{{"enabled":1000,"suspended":0,"warning":0}}}}]}}"#
    ))
    .unwrap();
    let p = LicensePlanes::build(&v, ver, Vec::new(), vec![skus]).unwrap();
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
    let p = LicensePlanes::build(&v, v1, vec![page(&[entry(0, Some(&[]))])], Vec::new()).unwrap();
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

// ---- D-DIRSIM-LICENSE-DT64-1, Gate 0: the source contract --------------------

const GROUP2: &str = "9a1b2c3d-0000-4000-8000-00000000c0de";

fn build_pages(users: u8, pages: Vec<LicensesPage>) -> LicensePlanes {
    let st = store(users);
    let ver = st.tag("observed").unwrap();
    let v = st.view(ver).unwrap();
    LicensePlanes::build(&v, ver, pages, Vec::new()).unwrap()
}

/// Reordering the pages, and the users inside them, keeps every code at the
/// same index with the same holders. Falsifier: codes in first-seen order
/// (the order before this correction) put SKU2/Disabled first in one build
/// and last in the other.
#[test]
fn reordered_pages_and_users_keep_every_code_in_place() {
    let a = [
        entry(0, Some(&[path(SKU2, "Disabled", None)])),
        entry(1, Some(&[path(SKU, "Active", Some(GROUP))])),
    ];
    let b = [
        entry(2, Some(&[path(SKU, "Active", None)])),
        entry(3, Some(&[path(SKU, "Error", None), path(SKU2, "Disabled", None)])),
    ];
    let forward = build_pages(4, vec![page(&a), page(&b)]);
    let mut b_rev = b.to_vec();
    b_rev.reverse();
    let mut a_rev = a.to_vec();
    a_rev.reverse();
    let backward = build_pages(4, vec![page(&b_rev), page(&a_rev)]);
    assert_eq!(forward.states(), backward.states());
    assert_eq!(forward.states().len(), 4);
    for code in 0..forward.states().len() {
        assert_eq!(
            forward.predicate_iri(code),
            backward.predicate_iri(code),
            "code {code}"
        );
        assert_eq!(
            forward.with_state(code).unwrap().rows(),
            backward.with_state(code).unwrap().rows(),
            "code {code}"
        );
    }
    // The order is the content's: SKU (c7df…) sorts after SKU2 (4b94…).
    assert_eq!(forward.states()[0].sku, g(SKU2));
    // Identity resolves to the index, and an unknown identity to nothing.
    let iri = forward.predicate_iri(0).unwrap().to_string();
    assert_eq!(forward.code_of(&iri), Some(0));
    assert_eq!(forward.code_of("urn:graph:licenseAssignmentState:none"), None);
}

/// A user whose property is JSON `null` the first time and read the second
/// is still a duplicate.
#[test]
fn a_null_first_entry_is_still_a_duplicate() {
    let null = format!(r#"{{"id":"{}","licenseAssignmentStates":null}}"#, user(0));
    let read = entry(0, Some(&[path(SKU, "Active", None)]));
    assert_eq!(
        build(2, &[null, read]).err(),
        Some(LicenseError::DuplicateUser(user(0)))
    );
}

/// Not listed, listed-not-read, read-empty and read-holding are four answers,
/// and the evidence keeps them apart.
#[test]
fn missing_empty_and_held_paths_stay_distinct() {
    let p = build(
        4,
        &[
            entry(0, None),
            entry(1, Some(&[])),
            entry(2, Some(&[path(SKU, "Active", None)])),
        ],
    )
    .unwrap();
    assert_eq!(p.paths(UserOrdinal(0)), Evidence::NotRead);
    assert_eq!(p.paths(UserOrdinal(1)), Evidence::Read(&[][..]));
    assert!(matches!(p.paths(UserOrdinal(2)), Evidence::Read(x) if x.len() == 1));
    assert_eq!(p.paths(UserOrdinal(3)), Evidence::NotListed);
}

/// Two groups assigning one SKU in one state set one bit; the evidence still
/// names both groups. Holders count once, paths count every path.
/// Falsifier: an explanation read from the plane alone could not name either
/// group, and counting paths as holders makes u0 count three times.
#[test]
fn the_evidence_keeps_both_assigning_groups_behind_one_bit() {
    let p = build(
        2,
        &[
            entry(
                0,
                Some(&[
                    path(SKU, "Active", None),
                    path(SKU, "Active", Some(GROUP)),
                    path(SKU, "Active", Some(GROUP2)),
                ]),
            ),
            entry(1, Some(&[path(SKU, "Active", Some(GROUP2))])),
        ],
    )
    .unwrap();
    assert_eq!(p.states().len(), 2, "direct and group: two codes, not three");
    let group_code = p
        .states()
        .iter()
        .position(|s| s.via == Via::Group)
        .unwrap();
    assert_eq!(p.with_state(group_code).unwrap().rows(), vec![0, 1]);
    let groups: Vec<Guid128> = p
        .explain(UserOrdinal(0), group_code)
        .unwrap()
        .map(|x| x.assigned_by_group.unwrap())
        .collect();
    assert_eq!(groups, vec![g(GROUP), g(GROUP2)]);
    assert_eq!(p.holder_count(&g(SKU)), 2);
    assert_eq!(p.path_count(&g(SKU)), 4);
    assert_eq!(
        p.explain(UserOrdinal(0), 9).err(),
        Some(ExplainRefusal::NoSuchCode(9))
    );
}

/// `assignedLicenses` and `licenseAssignmentStates` are separate properties.
/// A user with effective SKUs and unread paths is not read for the planes,
/// and the effective SKUs are still there as decoded.
#[test]
fn effective_skus_are_not_assignment_paths() {
    let only_effective = format!(
        r#"{{"id":"{}","assignedLicenses":[{{"skuId":"{SKU2}"}}]}}"#,
        user(0)
    );
    let p = build(1, &[only_effective]).unwrap();
    assert_eq!(p.paths(UserOrdinal(0)), Evidence::NotRead);
    assert!(matches!(p.effective(UserOrdinal(0)), Evidence::Read(e) if e.len() == 1 && e[0].sku == g(SKU2)));
    assert_eq!(p.not_read().rows(), vec![0]);
    assert!(p.holders(&g(SKU2)).rows().is_empty());
    assert_eq!(
        p.explain(UserOrdinal(0), 0).err(),
        Some(ExplainRefusal::NoSuchCode(0))
    );
}

fn linked(entries: &[String], next: Option<&str>) -> LicensesPage {
    let mut p = page(entries);
    p.next_link = next.map(str::to_string);
    p
}

/// Exactly one page ends a collection, in any page order. A link nobody
/// followed is partial; two ends are not one collection.
#[test]
fn coverage_reads_the_page_links_in_any_order() {
    let first = || linked(&[entry(0, Some(&[]))], Some("https://graph.example/next"));
    let last = || linked(&[entry(1, Some(&[]))], None);
    assert_eq!(build_pages(2, vec![first(), last()]).coverage(), Coverage::Complete);
    assert_eq!(build_pages(2, vec![last(), first()]).coverage(), Coverage::Complete);
    assert_eq!(build_pages(2, vec![first()]).coverage(), Coverage::Partial);
    assert_eq!(
        build_pages(2, vec![last(), linked(&[entry(0, Some(&[]))], None)]).coverage(),
        Coverage::Ambiguous
    );
    assert_eq!(build_pages(2, Vec::new()).coverage(), Coverage::NotCollected);
}
