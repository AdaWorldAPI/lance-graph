//! Makor citations addressed like the KJV: NiblePath from the outline.

use std::collections::HashSet;

use deepnsm_v2::toc::{TocEntry, TocLevel};
use lance_graph_contract::hhtl::NiblePath;
use mistaber_kripke_probe::makor::{
    MAKOR_BASIN, MakorHydration, hydrate_makor, parse_makor, path_of,
};

const FIXTURE: &str = include_str!("../fixtures/makor.tsv");

fn rows() -> Vec<(String, u16, u16)> {
    parse_makor(FIXTURE).expect("fixture parses")
}

fn seifim(rows: &[(String, u16, u16)]) -> Vec<(u16, u16)> {
    let mut out: Vec<(u16, u16)> = Vec::new();
    for (_, s, f) in rows {
        if !out.contains(&(*s, *f)) {
            out.push((*s, *f));
        }
    }
    out
}

fn node(h: &MakorHydration, level: TocLevel, chapter: u16) -> &TocEntry {
    h.hydration
        .entries
        .iter()
        .find(|e| {
            e.level == level && e.book == 0 && (level == TocLevel::Book || e.chapter == chapter)
        })
        .expect("node exists")
}

#[test]
fn every_citation_is_addressed() {
    let rows = rows();
    assert_eq!(rows.len(), 141);
    let h = hydrate_makor(&rows).expect("hydrates");
    assert_eq!(
        h.rules,
        rows.iter().map(|r| &r.0).collect::<HashSet<_>>().len()
    );
    assert_eq!(h.hydration.unaddressed, 0);
    assert_eq!(h.hydration.triples.len(), 141);
    let distinct: HashSet<NiblePath> = h.hydration.triples.iter().map(|t| t.path).collect();
    assert_eq!(distinct.len(), 10);
    for (t, (_, s, f)) in h.hydration.triples.iter().zip(&rows) {
        assert_eq!(Some(t.path), path_of(&h, *s, *f), "{s}:{f}");
    }
}

#[test]
fn containment_is_prefix_ancestry() {
    let h = hydrate_makor(&rows()).expect("hydrates");
    let p = |s, f| path_of(&h, s, f).expect("addressed");
    let (a, b, c) = (p(87, 1), p(87, 3), p(89, 1));
    let book = node(&h, TocLevel::Book, 0).path;
    let ch87 = node(&h, TocLevel::Chapter, 87).path;
    let ch89 = node(&h, TocLevel::Chapter, 89).path;
    // NiblePath::is_ancestor_of (hhtl.rs): prefix test.
    assert!(ch87.is_ancestor_of(a) && ch87.is_ancestor_of(b));
    assert!(!ch87.is_ancestor_of(c) && ch89.is_ancestor_of(c));
    // NiblePath::common_ancestor (hhtl.rs) is nibble-granular: two seifim
    // whose low-nibble-pair prefixes agree share an interior nibble below
    // the level node, so assert on level nodes, not on exact equality.
    let ab = a.common_ancestor(b).expect("same basin");
    assert!(
        ch87.is_ancestor_of(ab) || ab == ch87,
        "87:1/87:3 share the siman-87 node"
    );
    let ac = a.common_ancestor(c).expect("same basin");
    assert!(book.is_ancestor_of(ac) || ab == book);
    assert!(
        ac.depth() < ch87.depth(),
        "87:x and 89:1 share no chapter node"
    );
    assert!(ac.depth() >= book.depth(), "...but do share the book node");
    assert!(a.depth() > ch87.depth() && ch87.depth() > book.depth());
    for (s, f) in seifim(&rows()) {
        assert!(book.is_ancestor_of(p(s, f)), "{s}:{f}");
    }
    assert_eq!(h.hydration.toc.book_count(), 1);
}

#[test]
fn distinct_seifim_get_distinct_paths() {
    let rows = rows();
    let h = hydrate_makor(&rows).expect("hydrates");
    let paths: Vec<NiblePath> = seifim(&rows)
        .iter()
        .map(|&(s, f)| path_of(&h, s, f).expect("addressed"))
        .collect();
    assert_eq!(paths.len(), 10);
    assert_eq!(paths.iter().collect::<HashSet<_>>().len(), 10);
    assert!(paths.iter().all(|p| p.basin() == Some(MAKOR_BASIN)));
}

#[test]
fn a_siman_beyond_level_capacity_is_refused_not_folded() {
    let mut rows = rows();
    let before = hydrate_makor(&rows).expect("hydrates");
    rows.push(("r_far".to_owned(), 403, 1));
    let h = hydrate_makor(&rows).expect("hydrates");
    assert_eq!(h.hydration.unaddressed, 1);
    assert_eq!(h.hydration.triples.len(), 141);
    assert_eq!(path_of(&h, 403, 1), None);
    for (a, b) in h.hydration.triples.iter().zip(&before.hydration.triples) {
        assert_eq!((a.path, a.spo), (b.path, b.spo));
    }
}

#[test]
fn reading_order_is_the_fixture_order() {
    let rows = rows();
    let h = hydrate_makor(&rows).expect("hydrates");
    let t = &h.hydration.triples;
    assert!(t.iter().enumerate().all(|(i, x)| x.order as usize == i));
    let want = seifim(&rows);
    let got: Vec<NiblePath> = h.hydration.scanpath();
    let want: Vec<NiblePath> = want
        .iter()
        .map(|&(s, f)| path_of(&h, s, f).unwrap())
        .collect();
    assert_eq!(got, want);
    assert_eq!(h.hydration.chain().len(), 140);
}

#[test]
fn a_seif_can_have_many_citing_rules() {
    let rows = rows();
    let h = hydrate_makor(&rows).expect("hydrates");
    let p = path_of(&h, 87, 6).unwrap();
    let n = h.hydration.triples.iter().filter(|t| t.path == p).count();
    assert!(n > 1, "87:6 cited by {n}");
    assert_eq!(n, rows.iter().filter(|r| (r.1, r.2) == (87, 6)).count());
}

#[test]
fn malformed_lines_are_refused() {
    use mistaber_kripke_probe::Refusal;
    for bad in ["r\t87", "r\tx\t1", "\t87\t1", "r\t87\t1\textra"] {
        assert!(
            matches!(parse_makor(bad), Err(Refusal::Format(_))),
            "{bad:?}"
        );
    }
}
