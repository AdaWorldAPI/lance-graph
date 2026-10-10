//! A filter wider than the node budget is refused before its children are
//! held in memory. Its own test binary: the counting allocator is global,
//! and this file has a single test, so nothing else allocates while it
//! counts.

use lance_graph_dir_sim::ad::{project, Entry, Source};
use lance_graph_dir_sim::ldap::{Authority, Directory, Server, Session, MAX_PDU};
use lance_graph_dir_sim::observe::from_graph;
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128, OuDictionary, ValuePool};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Counting;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(l.size(), Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from `alloc` above, with the same layout.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(new, Ordering::Relaxed);
        }
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const TENANT: &str = "c0ffee00-1234-4abc-8def-000000000042";
const NC: &str = "DC=example,DC=de";

fn directory() -> Directory {
    let (mut dict, mut pool) = (OuDictionary::new(), ValuePool::new());
    let body = r#"{"value":[{"id":"7a3c9e11-2b44-4c6d-9e8f-0123456789ac","userPrincipalName":"cloud@example.de","accountEnabled":true}]}"#;
    let tenant = Guid128::parse(TENANT).unwrap();
    let recs = ogar_az::ingest_page(body, tenant, &mut dict, &mut pool, 0)
        .unwrap()
        .records;
    let out = from_graph(DirectoryScope(tenant), &recs, &pool, &mut dict).unwrap();
    let mut st = VersionStore::new();
    let v = st.observe("graph", 0, out.observation).unwrap();
    let p = project(
        &st.view(v).unwrap(),
        Source::Cloud {
            synthetic: &out.synthetic,
        },
        NC,
        &dict,
    )
    .unwrap();
    Directory::new(p, NC).unwrap()
}

struct Anyone;
impl Authority for Anyone {
    type Actor = ();
    fn bind(&self, _: &str, _: &[u8]) -> Option<()> {
        Some(())
    }
    fn visible(&self, _: &(), _: &Entry) -> bool {
        true
    }
}

fn tlv(tag: u8, v: &[u8]) -> Vec<u8> {
    let mut o = vec![tag];
    let n = v.len();
    if n < 0x80 {
        o.push(n as u8);
    } else {
        let b = n.to_be_bytes();
        let skip = b.iter().take_while(|&&x| x == 0).count();
        o.push(0x80 | (b.len() - skip) as u8);
        o.extend_from_slice(&b[skip..]);
    }
    o.extend_from_slice(v);
    o
}

#[test]
fn a_wide_filter_is_refused_before_its_children_are_held() {
    let dir = directory();
    let srv = Server::new(&dir, &Anyone);
    // An unauthenticated rootDSE search whose filter is an OR of tens of
    // thousands of presence items, just under the PDU limit.
    let leaf = tlv(0x87, b"objectClass");
    let n = (MAX_PDU - 1024) / leaf.len();
    let filter = tlv(0xa1, &leaf.repeat(n));
    let mut b = tlv(0x04, b"");
    b.extend(tlv(0x0a, &[0]));
    b.extend(tlv(0x0a, &[0]));
    b.extend(tlv(0x02, &[0]));
    b.extend(tlv(0x02, &[0]));
    b.extend(tlv(0x01, &[0]));
    b.extend(filter);
    b.extend(tlv(0x30, &[]));
    let mut m = tlv(0x02, &[1]);
    m.extend(tlv(0x63, &b));
    let pdu = tlv(0x30, &m);
    assert!(pdu.len() <= MAX_PDU && n > 50_000, "{} / {n}", pdu.len());

    let mut s = Session::default();
    ALLOCATED.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let out = srv.handle(&mut s, &pdu);
    COUNTING.store(false, Ordering::Relaxed);
    let allocated = ALLOCATED.load(Ordering::Relaxed);

    // Refused with a notice of disconnection, as any too-wide filter is.
    assert!(s.closed());
    assert_eq!(out.len(), 1);
    // Bounded by the node budget, not by the number of children: holding
    // every child would take more than a byte per child of the request.
    assert!(
        allocated < 64 * 1024,
        "{allocated} bytes allocated for a refused filter of {n} children"
    );
}
