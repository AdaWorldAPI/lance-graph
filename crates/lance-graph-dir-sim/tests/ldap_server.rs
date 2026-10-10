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
        (bind_req(5, 3, "admin@example.de", 0xa3, b""), 7),
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
    let ext = tlv(0xa9, &tlv(0x82, b"cn"));
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
        let r = run(&srv, &mut s, msg(2, tlv(op, &tlv(0x04, NC.as_bytes()))));
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

/// Sees everything, slowly.
struct Slow;
impl Authority for Slow {
    type Actor = ();
    fn bind(&self, _: &str, _: &[u8]) -> Option<()> {
        Some(())
    }
    fn visible(&self, _: &(), _: &Entry) -> bool {
        std::thread::sleep(std::time::Duration::from_millis(400));
        true
    }
}

#[test]
fn the_time_limit_is_honoured() {
    let dir = directory();
    let srv = Server::new(&dir, &Slow);
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
