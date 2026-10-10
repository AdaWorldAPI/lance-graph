//! The read-only LDAP handler over a projected cloud read, driven by raw
//! BER requests. Access is decided by a test authority.

use lance_graph_dir_sim::ad::{project, Entry, Source};
use lance_graph_dir_sim::ldap::{frame, Authority, Directory, LdapError, Server, Session};
use lance_graph_dir_sim::observe::from_graph;
use lance_graph_dir_sim::*;
use ogar_dir_core::{DirectoryScope, Guid128, OuDictionary, ValuePool};

const TENANT: &str = "c0ffee00-1234-4abc-8def-000000000042";
const ERIKA: &str = "7a3c9e11-2b44-4c6d-9e8f-0123456789ab";
const CLOUD: &str = "7a3c9e11-2b44-4c6d-9e8f-0123456789ac";
const NC: &str = "DC=example,DC=de";

fn g(s: &str) -> Guid128 {
    Guid128::parse(s).unwrap()
}

fn directory() -> Directory {
    let mut dict = OuDictionary::new();
    let mut pool = ValuePool::new();
    let body = format!(
        r#"{{"value":[
        {{"id":"{ERIKA}","userPrincipalName":"erika@example.de","accountEnabled":true,
          "onPremisesDistinguishedName":"CN=Erika,OU=Staff,DC=example,DC=de",
          "proxyAddresses":["SMTP:erika@example.de"]}},
        {{"id":"{CLOUD}","userPrincipalName":"cloud@example.de","accountEnabled":true}}]}}"#
    );
    let recs = ogar_az::ingest_page(&body, g(TENANT), &mut dict, &mut pool, 0)
        .unwrap()
        .records;
    let out = from_graph(DirectoryScope(g(TENANT)), &recs, &pool, &mut dict).unwrap();
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

/// `admin` sees everything; `erika` sees the domain, the OUs except the
/// synthetic container, and her own user,
/// and never reads `proxyAddresses`.
struct Iam;
impl Authority for Iam {
    type Actor = &'static str;
    fn bind(&self, name: &str, password: &[u8]) -> Option<&'static str> {
        match (name, password) {
            ("admin@example.de", b"pw") => Some("admin"),
            ("erika@example.de", b"pw") => Some("erika"),
            _ => None,
        }
    }
    fn visible(&self, a: &&'static str, e: &Entry) -> bool {
        *a == "admin"
            || (e.node.is_none() && !e.dn.starts_with("OU=Cloud Only"))
            || e.node == Some(g(ERIKA))
    }
    fn readable(&self, a: &&'static str, _: &Entry, name: &str) -> bool {
        *a == "admin" || !name.eq_ignore_ascii_case("proxyAddresses")
    }
}

// ── a minimal BER client ──

fn tlv(tag: u8, v: &[u8]) -> Vec<u8> {
    let mut o = vec![tag];
    if v.len() < 0x80 {
        o.push(v.len() as u8);
    } else {
        o.push(0x82);
        o.extend((v.len() as u16).to_be_bytes());
    }
    o.extend_from_slice(v);
    o
}
fn int(tag: u8, v: u8) -> Vec<u8> {
    tlv(tag, &[v])
}
fn msg(id: u8, op: Vec<u8>) -> Vec<u8> {
    let mut b = int(0x02, id);
    b.extend(op);
    tlv(0x30, &b)
}
fn bind_req(id: u8, version: u8, name: &str, auth_tag: u8, pw: &[u8]) -> Vec<u8> {
    let mut b = int(0x02, version);
    b.extend(tlv(0x04, name.as_bytes()));
    b.extend(tlv(auth_tag, pw));
    msg(id, tlv(0x60, &b))
}
fn bind(id: u8, name: &str, pw: &[u8]) -> Vec<u8> {
    bind_req(id, 3, name, 0x80, pw)
}
fn search(id: u8, base: &str, scope: u8, size: u8, filter: Vec<u8>, attrs: &[&str]) -> Vec<u8> {
    let mut b = tlv(0x04, base.as_bytes());
    b.extend(int(0x0a, scope));
    b.extend(int(0x0a, 0));
    b.extend(int(0x02, size));
    b.extend(int(0x02, 0));
    b.extend(int(0x01, 0));
    b.extend(filter);
    let mut a = Vec::new();
    for s in attrs {
        a.extend(tlv(0x04, s.as_bytes()));
    }
    b.extend(tlv(0x30, &a));
    msg(id, tlv(0x63, &b))
}
fn eq(a: &str, v: &str) -> Vec<u8> {
    let mut b = tlv(0x04, a.as_bytes());
    b.extend(tlv(0x04, v.as_bytes()));
    tlv(0xa3, &b)
}
fn present(a: &str) -> Vec<u8> {
    tlv(0x87, a.as_bytes())
}
fn and(fs: &[Vec<u8>]) -> Vec<u8> {
    tlv(0xa0, &fs.concat())
}
fn not(f: Vec<u8>) -> Vec<u8> {
    tlv(0xa2, &f)
}

fn read(b: &[u8]) -> (u8, &[u8], &[u8]) {
    let tag = b[0];
    let (len, h) = if b[1] < 0x80 {
        (b[1] as usize, 2)
    } else {
        let n = (b[1] & 0x7f) as usize;
        (
            b[2..2 + n].iter().fold(0usize, |a, &x| a << 8 | x as usize),
            2 + n,
        )
    };
    (tag, &b[h..h + len], &b[h + len..])
}
fn kids(mut b: &[u8]) -> Vec<(u8, &[u8])> {
    let mut o = Vec::new();
    while !b.is_empty() {
        let (t, v, r) = read(b);
        o.push((t, v));
        b = r;
    }
    o
}

#[derive(Debug, PartialEq)]
enum Resp {
    Entry {
        dn: String,
        attrs: Vec<(String, Vec<String>)>,
    },
    Done {
        op: u8,
        code: u8,
    },
}

fn decode(pdu: &[u8]) -> (u8, Resp) {
    let (t, v, rest) = read(pdu);
    assert_eq!((t, rest.len()), (0x30, 0));
    let k = kids(v);
    let id = k[0].1[0];
    let (op, body) = k[1];
    if op == 0x64 {
        let k = kids(body);
        let attrs = kids(k[1].1)
            .into_iter()
            .map(|(_, a)| {
                let a = kids(a);
                let vals = kids(a[1].1)
                    .into_iter()
                    .map(|(_, v)| String::from_utf8_lossy(v).into_owned())
                    .collect();
                (String::from_utf8(a[0].1.to_vec()).unwrap(), vals)
            })
            .collect();
        (
            id,
            Resp::Entry {
                dn: String::from_utf8(k[0].1.to_vec()).unwrap(),
                attrs,
            },
        )
    } else {
        (
            id,
            Resp::Done {
                op,
                code: kids(body)[0].1[0],
            },
        )
    }
}

fn run(srv: &Server<'_, Iam>, s: &mut Session<&'static str>, pdu: Vec<u8>) -> Vec<Resp> {
    assert_eq!(frame(&pdu), Ok(Some(pdu.len())));
    srv.handle(s, &pdu).iter().map(|r| decode(r).1).collect()
}
fn dns(r: &[Resp]) -> Vec<&str> {
    r.iter()
        .filter_map(|r| match r {
            Resp::Entry { dn, .. } => Some(dn.as_str()),
            Resp::Done { .. } => None,
        })
        .collect()
}
fn done(r: &[Resp]) -> u8 {
    match r.last().unwrap() {
        Resp::Done { code, .. } => *code,
        Resp::Entry { .. } => panic!("no result"),
    }
}

#[test]
fn the_root_dse_is_public_and_the_tree_is_not() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    let r = run(
        &srv,
        &mut s,
        search(1, "", 0, 0, present("objectClass"), &[]),
    );
    let Resp::Entry { dn, attrs } = &r[0] else {
        panic!()
    };
    assert_eq!(dn, "");
    assert!(attrs.contains(&("namingContexts".into(), vec![NC.into()])));
    assert!(attrs.contains(&("isReadOnly".into(), vec!["TRUE".into()])));
    assert_eq!(done(&r), 0);
    // Anonymous: nothing below the rootDSE.
    let r = run(
        &srv,
        &mut s,
        search(2, NC, 2, 0, present("objectClass"), &[]),
    );
    assert_eq!((dns(&r).len(), done(&r)), (0, 1));
}

#[test]
fn binds_are_checked() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    for (pdu, code) in [
        (bind(1, "admin@example.de", b"nope"), 49),
        (bind(2, "admin@example.de", b""), 53),
        (bind(3, "", b"pw"), 49),
        (bind_req(4, 2, "admin@example.de", 0x80, b"pw"), 2),
        (
            bind_req(5, 3, "admin@example.de", 0xa3, &tlv(0x04, b"GSSAPI")),
            7,
        ),
        (bind(6, "", b""), 0),
    ] {
        assert_eq!(done(&run(&srv, &mut s, pdu)), code);
        assert!(s.actor().is_none());
    }
    assert_eq!(
        done(&run(&srv, &mut s, bind(7, "admin@example.de", b"pw"))),
        0
    );
    assert_eq!(s.actor(), Some(&"admin"));
    // A failed rebind drops the earlier identity.
    assert_eq!(
        done(&run(&srv, &mut s, bind(8, "admin@example.de", b"x"))),
        49
    );
    assert!(s.actor().is_none());
}

#[test]
fn an_administrator_sees_the_emulated_tree_marked() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    let r = run(
        &srv,
        &mut s,
        search(2, NC, 2, 0, present("objectClass"), &["1.1"]),
    );
    // The domain, the two OUs, the two users.
    assert_eq!(dns(&r).len(), 5);
    assert_eq!(done(&r), 0);
    let r = run(
        &srv,
        &mut s,
        search(
            3,
            NC,
            2,
            0,
            and(&[eq("objectClass", "USER"), eq("dirSimOrigin", "synthetic")]),
            &["userPrincipalName", "dirSimOrigin"],
        ),
    );
    assert_eq!(
        r[0],
        Resp::Entry {
            dn: format!("CN={CLOUD},OU=Cloud Only (emulated),{NC}"),
            attrs: vec![
                ("userPrincipalName".into(), vec!["cloud@example.de".into()]),
                ("dirSimOrigin".into(), vec!["synthetic".into()]),
            ],
        }
    );
    assert_eq!(dns(&r).len(), 1);
    // Substring and NOT over the same tree.
    let mut sub = tlv(0x04, b"userPrincipalName");
    let mut parts = tlv(0x80, b"ERI");
    parts.extend(tlv(0x82, b"@example.de"));
    sub.extend(tlv(0x30, &parts));
    let r = run(&srv, &mut s, search(4, NC, 2, 0, tlv(0xa4, &sub), &["1.1"]));
    assert_eq!(dns(&r), [format!("CN={ERIKA},OU=Staff,{NC}")]);
    let r = run(
        &srv,
        &mut s,
        search(
            5,
            NC,
            2,
            0,
            and(&[
                eq("objectClass", "user"),
                not(eq("dirSimOrigin", "synthetic")),
            ]),
            &["1.1"],
        ),
    );
    assert_eq!(dns(&r), [format!("CN={ERIKA},OU=Staff,{NC}")]);
    // NOT of an Undefined filter is still Undefined: an extensible match
    // (unsupported) matches nothing, negated or not.
    // A well-formed MatchingRuleAssertion: type and matchValue.
    let ext = tlv(0xa9, &[tlv(0x82, b"cn"), tlv(0x83, b"x")].concat());
    for f in [ext.clone(), not(ext)] {
        let r = run(&srv, &mut s, search(8, NC, 2, 0, f, &["1.1"]));
        assert_eq!((dns(&r).len(), done(&r)), (0, 0));
    }
    // One level under the domain: the two top OUs.
    let r = run(
        &srv,
        &mut s,
        search(6, NC, 1, 0, present("objectClass"), &["1.1"]),
    );
    assert_eq!(dns(&r).len(), 2);
    // Size limit.
    let r = run(
        &srv,
        &mut s,
        search(7, NC, 2, 1, present("objectClass"), &["1.1"]),
    );
    assert_eq!((dns(&r).len(), done(&r)), (1, 4));
}

// A restricted actor sees only what IAM grants: a hidden entry is absent,
// even as a search base, and a hidden attribute neither shows nor matches.
#[test]
fn iam_filters_entries_and_attributes() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "erika@example.de", b"pw"));
    let r = run(
        &srv,
        &mut s,
        search(2, NC, 2, 0, eq("objectClass", "user"), &[]),
    );
    assert_eq!(dns(&r), [format!("CN={ERIKA},OU=Staff,{NC}")]);
    let Resp::Entry { attrs, .. } = &r[0] else {
        panic!()
    };
    assert!(!attrs.iter().any(|(n, _)| n == "proxyAddresses"));
    assert!(attrs.iter().any(|(n, _)| n == "userPrincipalName"));
    let r = run(
        &srv,
        &mut s,
        search(3, NC, 2, 0, present("proxyAddresses"), &["1.1"]),
    );
    assert_eq!(dns(&r).len(), 0);
    let hidden = format!("CN={CLOUD},OU=Cloud Only (emulated),{NC}");
    let r = run(
        &srv,
        &mut s,
        search(4, &hidden, 0, 0, present("objectClass"), &[]),
    );
    assert_eq!((dns(&r).len(), done(&r)), (0, 32));
    // A hidden container is missing too, as a base of any scope.
    let ou = format!("OU=Cloud Only (emulated),{NC}");
    for scope in [0, 2] {
        let r = run(
            &srv,
            &mut s,
            search(5, &ou, scope, 0, present("objectClass"), &[]),
        );
        assert_eq!((dns(&r).len(), done(&r)), (0, 32));
    }
}

#[test]
fn writes_are_refused_and_the_directory_does_not_change() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    for (op, resp) in [
        (0x66u8, 0x67u8),
        (0x68, 0x69),
        (0x6c, 0x6d),
        (0x6e, 0x6f),
        (0x77, 0x78),
    ] {
        let r = run(&srv, &mut s, msg(2, tlv(op, &write_body(op))));
        assert_eq!(r, [Resp::Done { op: resp, code: 53 }], "{op:#x}");
    }
    let r = run(&srv, &mut s, msg(3, tlv(0x4a, NC.as_bytes())));
    assert_eq!(r, [Resp::Done { op: 0x6b, code: 53 }]);
    let r = run(
        &srv,
        &mut s,
        search(4, NC, 2, 0, present("objectClass"), &["1.1"]),
    );
    assert_eq!(dns(&r).len(), 5);
}

#[test]
fn framing_unbind_and_malformed_requests() {
    let pdu = bind(1, "", b"");
    assert_eq!(frame(&pdu[..3]), Ok(None));
    assert_eq!(frame(&[]), Ok(None));
    assert_eq!(frame(&[0x04, 0]), Err(LdapError::Malformed));
    let mut two = pdu.clone();
    two.extend(&pdu);
    assert_eq!(frame(&two), Ok(Some(pdu.len())));
    assert_eq!(
        frame(&[0x30, 0x84, 0x7f, 0, 0, 0]),
        Err(LdapError::TooLarge)
    );

    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    assert!(srv.handle(&mut s, &msg(1, tlv(0x42, b""))).is_empty());
    assert!(s.closed());
    let mut s = Session::default();
    let out = srv.handle(&mut s, &[0x30, 0x03, 0x02, 0x01]);
    assert_eq!(out.len(), 1);
    assert_eq!(decode(&out[0]), (0, Resp::Done { op: 0x78, code: 2 }));
    assert!(s.closed());
}

fn ge(a: &str, v: &str) -> Vec<u8> {
    let mut b = tlv(0x04, a.as_bytes());
    b.extend(tlv(0x04, v.as_bytes()));
    tlv(0xa5, &b)
}
fn le(a: &str, v: &str) -> Vec<u8> {
    let mut b = tlv(0x04, a.as_bytes());
    b.extend(tlv(0x04, v.as_bytes()));
    tlv(0xa6, &b)
}

// Ordering follows the attribute: integers numerically (512 < 1000, which
// text order gets wrong), binary values and non-numbers Undefined.
#[test]
fn ordering_follows_the_attribute() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    for (f, n) in [
        (ge("userAccountControl", "1000"), 0),
        (ge("userAccountControl", "500"), 2),
        (le("userAccountControl", "1000"), 2),
        (le("userAccountControl", "99"), 0),
        (ge("userAccountControl", "abc"), 0),
        (not(ge("userAccountControl", "abc")), 0),
        (ge("objectGUID", "\0"), 0),
        (ge("userPrincipalName", "D"), 1),
    ] {
        let r = run(&srv, &mut s, search(2, NC, 2, 0, f, &["1.1"]));
        assert_eq!((dns(&r).len(), done(&r)), (n, 0));
    }
}

#[test]
fn deep_filters_are_refused_before_they_recurse() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let nest = |n: usize| (0..n).fold(present("objectClass"), |f, _| not(f));
    // Within the limit: answered normally (anonymous, so operationsError).
    let mut s = Session::default();
    let r = run(&srv, &mut s, search(1, NC, 2, 0, nest(31), &[]));
    assert_eq!(done(&r), 1);
    assert!(!s.closed());
    // Beyond it: malformed, notice of disconnection, session closed.
    let mut s = Session::default();
    let out = srv.handle(&mut s, &search(1, NC, 2, 0, nest(5000), &[]));
    assert_eq!(decode(&out[0]), (0, Resp::Done { op: 0x78, code: 2 }));
    assert!(s.closed());
}

fn with_control(pdu: Vec<u8>, critical: bool) -> Vec<u8> {
    let (_, body, _) = read(&pdu);
    let mut ctrl = tlv(0x04, b"1.2.840.113556.1.4.319");
    ctrl.extend(tlv(0x01, &[if critical { 0xff } else { 0 }]));
    let mut b = body.to_vec();
    b.extend(tlv(0xa0, &tlv(0x30, &ctrl)));
    tlv(0x30, &b)
}

#[test]
fn critical_controls_are_refused_and_others_ignored() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    let q = || search(2, NC, 2, 0, present("objectClass"), &["1.1"]);
    let r = run(&srv, &mut s, with_control(q(), true));
    assert_eq!(r, [Resp::Done { op: 0x65, code: 12 }]);
    let r = run(&srv, &mut s, with_control(q(), false));
    assert_eq!((dns(&r).len(), done(&r)), (5, 0));
    let r = run(
        &srv,
        &mut s,
        with_control(bind(3, "admin@example.de", b"pw"), true),
    );
    assert_eq!(r, [Resp::Done { op: 0x61, code: 12 }]);
}

/// Sees everything, slowly: each visibility check takes this many ms.
struct Slow(u64);
impl Authority for Slow {
    type Actor = ();
    fn bind(&self, _: &str, _: &[u8]) -> Option<()> {
        Some(())
    }
    fn visible(&self, _: &(), _: &Entry) -> bool {
        std::thread::sleep(std::time::Duration::from_millis(self.0));
        true
    }
}

#[test]
fn the_time_limit_is_honoured() {
    let dir = directory();
    let srv = Server::new(&dir, &Slow(400));
    let mut s = Session::default();
    srv.handle(&mut s, &bind(1, "x", b"y"));
    let mut q = search(2, NC, 2, 0, present("objectClass"), &["1.1"]);
    // sizeLimit, timeLimit, typesOnly: set timeLimit (the middle INTEGER) to 1 s.
    let pos = q
        .windows(9)
        .position(|w| w == [0x02, 1, 0, 0x02, 1, 0, 0x01, 1, 0])
        .unwrap();
    q[pos + 5] = 1;
    let r: Vec<Resp> = srv.handle(&mut s, &q).iter().map(|p| decode(p).1).collect();
    assert_eq!(done(&r), 3);
    assert!(dns(&r).len() < 5);
}

fn or(fs: &[Vec<u8>]) -> Vec<u8> {
    tlv(0xa1, &fs.concat())
}

// A shallow but wide filter is bounded too: every node counts.
#[test]
fn wide_filters_are_refused() {
    use lance_graph_dir_sim::ldap::MAX_FILTER_NODES;
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let leaves = |n: usize| or(&vec![eq("cn", "x"); n]);
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    // The `or` itself plus its leaves: exactly at the budget.
    let r = run(
        &srv,
        &mut s,
        search(2, NC, 2, 0, leaves(MAX_FILTER_NODES - 1), &["1.1"]),
    );
    assert_eq!((dns(&r).len(), done(&r)), (0, 0));
    assert!(!s.closed());
    let out = srv.handle(
        &mut s,
        &search(3, NC, 2, 0, leaves(MAX_FILTER_NODES), &["1.1"]),
    );
    assert_eq!(decode(&out[0]), (0, Resp::Done { op: 0x78, code: 2 }));
    assert!(s.closed());
}

/// A search with explicit size and time limits, encoded as given.
fn search_limits(size: &[u8], time: &[u8]) -> Vec<u8> {
    let mut b = tlv(0x04, NC.as_bytes());
    b.extend(int(0x0a, 2));
    b.extend(int(0x0a, 0));
    b.extend(tlv(0x02, size));
    b.extend(tlv(0x02, time));
    b.extend(int(0x01, 0));
    b.extend(present("objectClass"));
    b.extend(tlv(0x30, &tlv(0x04, b"1.1")));
    msg(2, tlv(0x63, &b))
}

// Limits outside 0..maxInt are a protocol error, never a panic, even
// before the session is bound.
#[test]
fn out_of_range_limits_are_refused_without_panicking() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let huge = [0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
    for pdu in [
        search_limits(&[0], &huge),
        search_limits(&huge, &[0]),
        search_limits(&[0], &[0xff]),
    ] {
        let mut s = Session::default();
        let out = srv.handle(&mut s, &pdu);
        assert_eq!(decode(&out[0]), (0, Resp::Done { op: 0x78, code: 2 }));
        assert!(s.closed());
    }
    // The largest legal limit is accepted.
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    let r = run(&srv, &mut s, search_limits(&[0], &[0x7f, 0xff, 0xff, 0xff]));
    assert_eq!((dns(&r).len(), done(&r)), (5, 0));
}

fn with_time_limit(mut q: Vec<u8>, secs: u8) -> Vec<u8> {
    let pos = q
        .windows(9)
        .position(|w| w == [0x02, 1, 0, 0x02, 1, 0, 0x01, 1, 0])
        .unwrap();
    q[pos + 5] = secs;
    q
}

// A base-scope search has one candidate: a limit that runs out while it is
// checked is still exceeded, not reported as success.
#[test]
fn the_time_limit_holds_for_the_last_candidate() {
    let dir = directory();
    let srv = Server::new(&dir, &Slow(600));
    let mut s = Session::default();
    srv.handle(&mut s, &bind(1, "x", b"y"));
    // The base is the last entry, so no later loop iteration rechecks.
    let last = format!("CN={CLOUD},OU=Cloud Only (emulated),{NC}");
    let q = with_time_limit(search(2, &last, 0, 0, present("objectClass"), &["1.1"]), 1);
    let r: Vec<Resp> = srv.handle(&mut s, &q).iter().map(|p| decode(p).1).collect();
    assert_eq!(done(&r), 3);
}

// A rebind refused for a critical control still drops the earlier identity.
#[test]
fn a_rebind_refused_for_a_control_drops_the_identity() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    assert!(s.actor().is_some());
    let r = run(
        &srv,
        &mut s,
        with_control(bind(2, "admin@example.de", b"pw"), true),
    );
    assert_eq!(r, [Resp::Done { op: 0x61, code: 12 }]);
    assert!(s.actor().is_none());
    let r = run(
        &srv,
        &mut s,
        search(3, NC, 2, 0, present("objectClass"), &["1.1"]),
    );
    assert_eq!((dns(&r).len(), done(&r)), (0, 1));
}

// A subtree search of the naming context with the four fixed-type fields
// after the scope given verbatim: derefAliases, sizeLimit, timeLimit,
// typesOnly.
fn search_fields(fields: [(u8, &[u8]); 4], filter: Vec<u8>) -> Vec<u8> {
    let mut b = tlv(0x04, NC.as_bytes());
    b.extend(int(0x0a, 2));
    for (t, v) in fields {
        b.extend(tlv(t, v));
    }
    b.extend(filter);
    b.extend(tlv(0x30, &tlv(0x04, b"1.1")));
    msg(2, tlv(0x63, &b))
}

// Every SearchRequest field has a fixed ASN.1 type (RFC 4511 §4.5.1); a
// request that breaks one is a protocol error, not a search.
#[test]
fn search_fields_are_type_checked() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let ok: [(u8, &[u8]); 4] = [(0x0a, &[0]), (0x02, &[0]), (0x02, &[0]), (0x01, &[0])];
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    let r = run(&srv, &mut s, search_fields(ok, present("objectClass")));
    assert_eq!((dns(&r).len(), done(&r)), (5, 0));
    let bad: [[(u8, &[u8]); 4]; 6] = [
        [(0x02, &[0]), ok[1], ok[2], ok[3]], // derefAliases not ENUMERATED
        [(0x0a, &[4]), ok[1], ok[2], ok[3]], // derefAliases out of range
        [ok[0], (0x04, &[0]), ok[2], ok[3]], // sizeLimit not INTEGER
        [ok[0], ok[1], (0x0a, &[0]), ok[3]], // timeLimit not INTEGER
        [ok[0], ok[1], ok[2], (0x04, &[1])], // typesOnly not BOOLEAN
        [ok[0], ok[1], ok[2], (0x01, &[1, 1])], // BOOLEAN of two octets
    ];
    for f in bad {
        let mut s = Session::default();
        run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
        let out = srv.handle(&mut s, &search_fields(f, present("objectClass")));
        assert_eq!(decode(&out[0]), (0, Resp::Done { op: 0x78, code: 2 }));
        assert!(s.closed());
    }
}

// Equality on an integer attribute is integerMatch: numeric, and Undefined
// for an assertion that is not an integer, so a NOT of it matches nothing.
#[test]
fn integer_equality_is_numeric_and_undefined_for_non_integers() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    let mut q = |f| dns(&run(&srv, &mut s, search(2, NC, 2, 0, f, &["1.1"]))).len();
    let plain = q(eq("userAccountControl", "512"));
    assert!(plain > 0, "the fixture must carry an enabled account");
    // RFC 4517 Integer: no leading zero, no "-0", any magnitude.
    for invalid in ["abc", "0512", "-0", "+512", ""] {
        assert_eq!(q(eq("userAccountControl", invalid)), 0, "{invalid}");
        assert_eq!(q(not(eq("userAccountControl", invalid))), 0, "{invalid}");
    }
    let huge = "92233720368547758080000";
    assert_eq!(q(le("userAccountControl", huge)), plain);
    // Every entry: those with the attribute differ from `huge`, the others
    // lack it.
    assert_eq!(
        q(not(eq("userAccountControl", huge))),
        q(present("objectClass"))
    );
    assert_eq!(
        q(ge("userAccountControl", &format!("-{huge}"))),
        q(present("userAccountControl"))
    );
    assert_eq!(q(ge("userAccountControl", huge)), 0);
}

struct SlowHidden(u64);
impl Authority for SlowHidden {
    type Actor = ();
    fn bind(&self, _: &str, _: &[u8]) -> Option<()> {
        Some(())
    }
    fn visible(&self, _: &(), _: &Entry) -> bool {
        std::thread::sleep(std::time::Duration::from_millis(self.0));
        false
    }
}

// Resolving a hidden base can itself use up the time limit; that is
// timeLimitExceeded, not noSuchObject.
#[test]
fn the_time_limit_holds_while_the_base_is_resolved() {
    let dir = directory();
    let srv = Server::new(&dir, &SlowHidden(1100));
    let mut s = Session::default();
    srv.handle(&mut s, &bind(1, "x", b"y"));
    let q = with_time_limit(search(2, NC, 0, 0, present("objectClass"), &["1.1"]), 1);
    let r: Vec<Resp> = srv.handle(&mut s, &q).iter().map(|p| decode(p).1).collect();
    assert_eq!(done(&r), 3);
}

// Attach one control, given as its raw SEQUENCE content.
fn with_raw_control(pdu: Vec<u8>, ctrl: Vec<u8>) -> Vec<u8> {
    let (_, body, _) = read(&pdu);
    let mut b = body.to_vec();
    b.extend(tlv(0xa0, &tlv(0x30, &ctrl)));
    tlv(0x30, &b)
}

// A search whose attribute list is given as raw TLVs.
fn search_attrs(attrs: Vec<u8>) -> Vec<u8> {
    let mut b = tlv(0x04, NC.as_bytes());
    for (t, v) in [(0x0a, 2), (0x0a, 0), (0x02, 0), (0x02, 0), (0x01, 0)] {
        b.extend(int(t, v));
    }
    b.extend(present("objectClass"));
    b.extend(tlv(0x30, &attrs));
    msg(2, tlv(0x63, &b))
}

// Every typed field inside a request is checked: control OID, criticality
// BOOLEAN and value; requested attribute names; the attribute and value of
// an assertion filter; a non-empty substrings sequence. A violation
// disconnects; a well-formed request still runs.
#[test]
fn nested_fields_are_type_checked() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let q = || search(2, NC, 2, 0, present("objectClass"), &["1.1"]);
    let oid = tlv(0x04, b"1.2.840.113556.1.4.319");
    let with = |extra: &[Vec<u8>]| [oid.clone(), extra.concat()].concat();
    let mut sub = tlv(0x04, b"cn");
    sub.extend(tlv(0x30, &[]));
    let mut int_attr = tlv(0x02, b"cn");
    int_attr.extend(tlv(0x04, b"x"));
    let mut int_value = tlv(0x04, b"cn");
    int_value.extend(tlv(0x02, b"x"));
    let bad = [
        with_raw_control(q(), with(&[tlv(0x01, &[])])), // empty BOOLEAN
        with_raw_control(q(), with(&[tlv(0x01, &[0, 0])])), // two octets
        with_raw_control(q(), [tlv(0x02, &[1]), tlv(0x01, &[0])].concat()), // OID tag
        with_raw_control(q(), with(&[tlv(0x02, &[1])])), // value tag
        with_raw_control(q(), with(&[tlv(0x04, b"v"), tlv(0x04, b"w")])), // extra field
        search_attrs(tlv(0x02, b"objectClass")),
        search(2, NC, 2, 0, tlv(0xa4, &sub), &["1.1"]),
        search(2, NC, 2, 0, tlv(0xa3, &int_attr), &["1.1"]),
        search(2, NC, 2, 0, tlv(0xa3, &int_value), &["1.1"]),
    ];
    for pdu in bad {
        let mut s = Session::default();
        run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
        let out = srv.handle(&mut s, &pdu);
        assert_eq!(decode(&out[0]), (0, Resp::Done { op: 0x78, code: 2 }));
        assert!(s.closed());
    }
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    for pdu in [
        with_raw_control(q(), with(&[])),
        with_raw_control(q(), with(&[tlv(0x04, b"v")])),
        with_raw_control(q(), with(&[tlv(0x01, &[0]), tlv(0x04, b"v")])),
        search_attrs(tlv(0x04, b"1.1")),
    ] {
        let r = run(&srv, &mut s, pdu);
        assert_eq!((dns(&r).len(), done(&r)), (5, 0));
    }
}

// A filter on an attribute type the server does not recognise is Undefined
// and stays Undefined under NOT; `present` on it is FALSE. Every type the
// server emits is recognised.
#[test]
fn unknown_attribute_types_are_undefined() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    let mut q = |base: &str, scope: u8, f| run(&srv, &mut s, search(2, base, scope, 0, f, &["*"]));
    assert_eq!(dns(&q(NC, 2, not(eq("unknownAttribute", "x")))).len(), 0);
    assert_eq!(dns(&q(NC, 2, present("unknownAttribute"))).len(), 0);
    assert_eq!(dns(&q(NC, 2, not(present("unknownAttribute")))).len(), 5);
    // Drift guard: a NOT of a non-matching equality on each emitted type
    // matches, so none of them is treated as unknown.
    let mut names: Vec<String> = Vec::new();
    for (base, scope) in [(NC, 2u8), ("", 0)] {
        for r in q(base, scope, present("objectClass")) {
            if let Resp::Entry { attrs, .. } = r {
                names.extend(attrs.into_iter().map(|(n, _)| n));
            }
        }
    }
    names.sort();
    names.dedup();
    assert!(names.len() > 8, "{names:?}");
    for n in names {
        let v = match n.to_lowercase().as_str() {
            "useraccountcontrol" | "supportedldapversion" => "999999",
            "member" | "distinguishedname" | "namingcontexts" | "defaultnamingcontext" => "CN=none",
            _ => "zz-no-such-value",
        };
        let (base, scope) = if [
            "namingContexts",
            "defaultNamingContext",
            "supportedLDAPVersion",
            "vendorName",
            "isReadOnly",
        ]
        .contains(&n.as_str())
        {
            ("", 0)
        } else {
            (NC, 2)
        };
        assert!(
            !dns(&q(base, scope, not(eq(&n, v)))).is_empty(),
            "{n} is not recognised"
        );
    }
}

fn substr_initial(a: &str, v: &str) -> Vec<u8> {
    let mut b = tlv(0x04, a.as_bytes());
    b.extend(tlv(0x30, &tlv(0x80, v.as_bytes())));
    tlv(0xa4, &b)
}

// distinguishedName and member compare as parsed DNs: a differently escaped
// spelling of the same name matches, a value that is not a DN is Undefined,
// and DNs have no substrings rule.
#[test]
fn dn_values_match_by_parsed_name() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    let mut q = |f| {
        dns(&run(&srv, &mut s, search(2, NC, 2, 0, f, &["1.1"])))
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
    };
    let plain = format!("CN={CLOUD},OU=Cloud Only (emulated),{NC}");
    let escaped = format!("cn={CLOUD},ou=Cloud Only \\28emulated\\29,dc=EXAMPLE,dc=de");
    assert_eq!(q(eq("distinguishedName", &plain)).len(), 1);
    assert_eq!(
        q(eq("distinguishedName", &escaped)),
        q(eq("distinguishedName", &plain))
    );
    assert_eq!(q(not(eq("distinguishedName", "not a dn"))).len(), 0);
    assert_eq!(q(not(substr_initial("distinguishedName", "CN="))).len(), 0);
}

// and / or with no members are malformed.
#[test]
fn empty_and_or_are_refused() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    for f in [tlv(0xa0, &[]), tlv(0xa1, &[]), not(tlv(0xa0, &[]))] {
        let mut s = Session::default();
        run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
        let out = srv.handle(&mut s, &search(2, NC, 2, 0, f, &["1.1"]));
        assert_eq!(decode(&out[0]), (0, Resp::Done { op: 0x78, code: 2 }));
        assert!(s.closed());
    }
}

// The server caps what one search returns, whatever the client asks.
#[test]
fn server_limits_cap_unlimited_searches() {
    let dir = directory();
    let q = |srv: &Server<'_, Iam>, size: u8| {
        let mut s = Session::default();
        run(srv, &mut s, bind(1, "admin@example.de", b"pw"));
        let r = run(
            srv,
            &mut s,
            search(2, NC, 2, size, present("objectClass"), &["*"]),
        );
        (dns(&r).len(), done(&r))
    };
    // The fixture has five entries: a cap of five is not reached.
    assert_eq!(
        q(&Server::new(&dir, &Iam).with_limits(5, usize::MAX), 0),
        (5, 0)
    );
    assert_eq!(
        q(&Server::new(&dir, &Iam).with_limits(2, usize::MAX), 0),
        (2, 11)
    );
    // The client's own smaller limit still reports sizeLimitExceeded.
    assert_eq!(
        q(&Server::new(&dir, &Iam).with_limits(2, usize::MAX), 1),
        (1, 4)
    );
    // The byte cap holds even when no entry fits.
    assert_eq!(q(&Server::new(&dir, &Iam).with_limits(100, 1), 0), (0, 11));
    // The default cap leaves a small directory whole.
    assert_eq!(q(&Server::new(&dir, &Iam), 0), (5, 0));
}

// Every message from a client carries a nonzero MessageID.
#[test]
fn message_id_zero_is_refused() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    let out = srv.handle(&mut s, &bind(0, "admin@example.de", b"pw"));
    assert_eq!(decode(&out[0]), (0, Resp::Done { op: 0x78, code: 2 }));
    assert!(s.closed());
    assert!(s.actor().is_none());
}

// An extensible match is decoded for shape: malformed ones disconnect,
// well-formed ones are Undefined.
#[test]
fn extensible_matches_are_shape_checked() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let f = |parts: &[Vec<u8>]| tlv(0xa9, &parts.concat());
    let (rule, ty, val, dn) = (
        tlv(0x81, b"1.2.840.113556.1.4.803"),
        tlv(0x82, b"userAccountControl"),
        tlv(0x83, b"2"),
        tlv(0x84, &[0xff]),
    );
    for bad in [
        f(&[]),                                        // empty
        f(std::slice::from_ref(&ty)),                  // no matchValue
        f(std::slice::from_ref(&val)),                 // neither rule nor type
        f(&[val.clone(), ty.clone()]),                 // out of order
        f(&[ty.clone(), ty.clone(), val.clone()]),     // duplicate
        f(&[ty.clone(), val.clone(), tlv(0x84, &[])]), // empty dnAttributes
        f(&[ty.clone(), tlv(0x04, b"2")]),             // wrong tag
    ] {
        let mut s = Session::default();
        run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
        let out = srv.handle(&mut s, &search(2, NC, 2, 0, bad, &["1.1"]));
        assert_eq!(decode(&out[0]), (0, Resp::Done { op: 0x78, code: 2 }));
        assert!(s.closed());
    }
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    for good in [
        f(&[ty.clone(), val.clone()]),
        f(&[rule.clone(), val.clone()]),
        f(&[rule, ty, val, dn]),
    ] {
        let r = run(&srv, &mut s, search(2, NC, 2, 0, not(good), &["1.1"]));
        assert_eq!((dns(&r).len(), done(&r)), (0, 0));
    }
}

// Selectors are deduplicated, so repeating one costs nothing, and the
// number of distinct ones is capped.
#[test]
fn attribute_selectors_are_bounded() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    let many = vec!["cn"; 10_000];
    let r = run(
        &srv,
        &mut s,
        search(2, NC, 2, 0, present("objectClass"), &many),
    );
    assert_eq!((dns(&r).len(), done(&r)), (5, 0));
    let distinct: Vec<String> = (0..300).map(|i| format!("a{i}")).collect();
    let distinct: Vec<&str> = distinct.iter().map(String::as_str).collect();
    let r = run(
        &srv,
        &mut s,
        search(3, NC, 2, 0, present("objectClass"), &distinct),
    );
    assert_eq!((dns(&r).len(), done(&r)), (0, 11));
}

// Substring items count against the filter budget.
#[test]
fn substring_items_count_against_the_budget() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let sub = |n: usize| {
        let mut b = tlv(0x04, b"cn");
        b.extend(tlv(0x30, &vec![tlv(0x81, b"a"); n].concat()));
        tlv(0xa4, &b)
    };
    let mut s = Session::default();
    run(&srv, &mut s, bind(1, "admin@example.de", b"pw"));
    let r = run(&srv, &mut s, search(2, NC, 2, 0, sub(10), &["1.1"]));
    assert_eq!(done(&r), 0);
    let out = srv.handle(&mut s, &search(3, NC, 2, 0, sub(5000), &["1.1"]));
    assert_eq!(decode(&out[0]), (0, Resp::Done { op: 0x78, code: 2 }));
}

/// Sees everything; `visible` of the entry named here takes this long.
struct SlowOn(&'static str, u64);
impl Authority for SlowOn {
    type Actor = ();
    fn bind(&self, _: &str, _: &[u8]) -> Option<()> {
        Some(())
    }
    fn visible(&self, _: &(), e: &Entry) -> bool {
        if e.dn.contains(self.0) {
            std::thread::sleep(std::time::Duration::from_millis(self.1));
        }
        true
    }
}

// A candidate whose checks cross the deadline is not returned.
#[test]
fn an_entry_found_after_the_deadline_is_not_returned() {
    let dir = directory();
    // Slow only for the last entry, and only in the loop (the base is the NC).
    let srv = Server::new(&dir, &SlowOn(CLOUD, 1100));
    let mut s = Session::default();
    srv.handle(&mut s, &bind(1, "x", b"y"));
    let q = with_time_limit(search(2, NC, 2, 0, present("objectClass"), &["1.1"]), 1);
    let r: Vec<Resp> = srv.handle(&mut s, &q).iter().map(|p| decode(p).1).collect();
    assert_eq!(done(&r), 3);
    assert!(!dns(&r).iter().any(|d| d.contains(CLOUD)), "{r:?}");
}

fn attr(t: &str, vals: &[&str]) -> Vec<u8> {
    let v: Vec<u8> = vals.iter().flat_map(|v| tlv(0x04, v.as_bytes())).collect();
    tlv(0x30, &[tlv(0x04, t.as_bytes()), tlv(0x31, &v)].concat())
}

// A well-formed body for each refused operation.
fn write_body(op: u8) -> Vec<u8> {
    let dn = tlv(0x04, NC.as_bytes());
    match op {
        0x66 => [
            dn,
            tlv(
                0x30,
                &tlv(0x30, &[int(0x0a, 2), attr("cn", &["x"])].concat()),
            ),
        ]
        .concat(),
        0x68 => [dn, tlv(0x30, &attr("cn", &["x"]))].concat(),
        0x6c => [dn, tlv(0x04, b"CN=x"), tlv(0x01, &[0xff])].concat(),
        0x6e => [dn, tlv(0x30, &[tlv(0x04, b"cn"), tlv(0x04, b"x")].concat())].concat(),
        0x77 => tlv(0x80, b"1.3.6.1.4.1.4203.1.11.3"),
        _ => unreachable!(),
    }
}

// Every request is shape-checked before it is answered: a malformed one
// disconnects even when the operation would be refused, or carries a
// critical control.
#[test]
fn every_request_is_shape_checked() {
    let dir = directory();
    let srv = Server::new(&dir, &Iam);
    let dn = || tlv(0x04, NC.as_bytes());
    let bad: Vec<Vec<u8>> = vec![
        msg(2, tlv(0x42, &[0])),                            // unbind with content
        msg(2, tlv(0x50, &[])),                             // abandon, empty
        msg(2, tlv(0x50, &[0xff])),                         // abandon, negative
        msg(2, tlv(0x50, &[0x7f, 0xff, 0xff, 0xff, 0xff])), // abandon, past maxInt
        msg(2, tlv(0x4a, &[0xff])),                         // delete, not UTF-8
        msg(2, tlv(0x66, &dn())),                           // modify, no changes
        msg(
            2,
            tlv(
                0x66,
                &[
                    dn(),
                    tlv(0x30, &tlv(0x30, &[int(0x0a, 9), attr("cn", &[])].concat())),
                ]
                .concat(),
            ),
        ), // bad op
        msg(2, tlv(0x68, &[dn(), tlv(0x30, &attr("cn", &[]))].concat())), // add, no values
        msg(2, tlv(0x6c, &[dn(), tlv(0x04, b"CN=x")].concat())), // modDN, no deleteoldrdn
        msg(
            2,
            tlv(
                0x6c,
                &[dn(), tlv(0x04, b"CN=x"), tlv(0x01, &[1]), tlv(0x04, b"x")].concat(),
            ),
        ), // bad newSuperior tag
        msg(2, tlv(0x6e, &[dn(), tlv(0x30, &tlv(0x04, b"cn"))].concat())), // compare, half an ava
        msg(2, tlv(0x77, &tlv(0x81, b"x"))),                // extended, no name
        bind_req(2, 3, "a", 0xa3, &[]),                     // SASL, no mechanism
        with_control(search_attrs(tlv(0x02, b"cn")), true), // bad search, critical control
        with_control(msg(2, tlv(0x42, &[0])), true),        // bad unbind, critical control
    ];
    for (i, pdu) in bad.into_iter().enumerate() {
        let mut s = Session::default();
        let out = srv.handle(&mut s, &pdu);
        assert_eq!(
            decode(&out[0]),
            (0, Resp::Done { op: 0x78, code: 2 }),
            "case {i}"
        );
        assert!(s.closed(), "case {i}");
    }
    // Well-formed: an unbind closes quietly, an abandon is silent.
    let mut s = Session::default();
    assert!(srv.handle(&mut s, &msg(2, tlv(0x50, &[1]))).is_empty());
    assert!(!s.closed());
    assert!(srv.handle(&mut s, &msg(3, tlv(0x42, &[]))).is_empty());
    assert!(s.closed());
}

/// Sees and reads everything, counting reads of `ou`: only building an OU
/// entry reads it under a filter on objectClass.
struct CountOu(std::cell::Cell<usize>);
impl Authority for CountOu {
    type Actor = ();
    fn bind(&self, _: &str, _: &[u8]) -> Option<()> {
        Some(())
    }
    fn visible(&self, _: &(), _: &Entry) -> bool {
        true
    }
    fn readable(&self, _: &(), _: &Entry, name: &str) -> bool {
        if name == "ou" {
            self.0.set(self.0.get() + 1);
        }
        true
    }
}

// The entry cap stops a search before the next entry is built: with a cap
// of one, the naming context is returned and the OU after it is never
// projected.
#[test]
fn the_entry_cap_is_checked_before_building_the_entry() {
    let dir = directory();
    let auth = CountOu(std::cell::Cell::new(0));
    let srv = Server::new(&dir, &auth).with_limits(1, usize::MAX);
    let mut s = Session::default();
    srv.handle(&mut s, &bind(1, "x", b"y"));
    let out = srv.handle(&mut s, &search(2, NC, 2, 0, present("objectClass"), &["*"]));
    let r: Vec<Resp> = out.iter().map(|p| decode(p).1).collect();
    assert_eq!((dns(&r), done(&r)), (vec![NC], 11));
    assert_eq!(auth.0.get(), 0);
    // Control: without the cap the OUs are built, and `ou` is read.
    let srv = Server::new(&dir, &auth);
    srv.handle(&mut s, &search(3, NC, 2, 0, present("objectClass"), &["*"]));
    assert!(auth.0.get() > 0);
}
