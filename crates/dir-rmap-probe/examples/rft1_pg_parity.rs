//! D-RFT-1: the typed composition against PostgreSQL as the semantic
//! reference.
//!
//! The fixture's membership, security set, nesting and grants are emitted in
//! `Prepared` ordinals. PostgreSQL computes the reflexive nesting closure
//! over security groups with a recursive CTE, joins membership to the role
//! carriers, and returns count(*) (Bag), count(DISTINCT (user, role))
//! (Distinct) and the per-coordinate multiplicities. Every seventh user also
//! gets a membership row with a NULL group, a NULL join key that must match
//! nothing in both engines.
//!
//! Needs `psql` and a server; connection from the usual `PG*` variables.
//! Without one it prints `SKIP` and exits 0.
//!
//! ```text
//! PGHOST=/var/tmp/k3pg PGPORT=54329 PGUSER=k3 PGDATABASE=postgres \
//!   cargo run --release --example rft1_pg_parity
//! ```
//! (from this crate's directory)

use dir_rmap_probe::World;
use dir_rmap_probe::compose::{Carriers, Law, Members, add_bag, compose};
use dir_rmap_probe::fixture::{Fixture, Spec};
use dir_rmap_probe::prep::{bit, ones};
use dir_rmap_probe::qr::Qr;
use std::fmt::Write as _;
use std::io::Write as _;
use std::process::{Command, Stdio};

const CTE: &str = "WITH RECURSIVE reach(g, a) AS (\
    SELECT g, g FROM sec \
    UNION \
    SELECT r.g, n.p FROM reach r JOIN nest n ON n.c = r.a JOIN sec s ON s.g = n.p), \
  carries AS (\
    SELECT DISTINCT r.g, gr.role FROM reach r JOIN grant_ gr ON gr.g = r.a \
    JOIN sec s ON s.g = r.g) ";

fn psql(script: String) -> Result<String, String> {
    let out = Command::new("psql")
        .args(["-X", "-q", "-tA", "-F,", "-v", "ON_ERROR_STOP=1", "-f", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            // Write on a separate thread: psql answers while it reads, so a
            // parent that wrote everything first would block once the stdout
            // pipe filled. A refused connection closes the pipe early; the
            // exit status and stderr say why, so a write error is moot.
            let mut stdin = child.stdin.take().expect("piped stdin");
            let writer = std::thread::spawn(move || {
                let _ = stdin.write_all(script.as_bytes());
            });
            let out = child.wait_with_output();
            let _ = writer.join();
            out
        })
        .map_err(|e| format!("could not run psql ({e})"))?;
    if !out.status.success() {
        return Err(format!(
            "psql failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8(out.stdout).expect("utf-8"))
}

fn main() {
    let w = World::new(Fixture::generate(Spec::small()));
    let b = w.prepare();
    let qr = Qr::bind(&b, &w.witness()).expect("fits 256 slots");
    let users = b.users;

    // Tables in Prepared ordinals.
    let mut s = String::from(
        "CREATE TEMP TABLE m (u int, g int);\n\
         CREATE TEMP TABLE sec (g int);\n\
         CREATE TEMP TABLE nest (c int, p int);\n\
         CREATE TEMP TABLE grant_ (g int, role int);\n",
    );
    s.push_str("INSERT INTO m VALUES ");
    let mut first = true;
    for u in 0..users {
        for &g in b.user_groups.row(u) {
            let _ = write!(s, "{}({u},{g})", if first { "" } else { "," });
            first = false;
        }
        if u % 7 == 0 {
            let _ = write!(s, "{}({u},NULL)", if first { "" } else { "," });
            first = false;
        }
    }
    s.push_str(";\nINSERT INTO sec VALUES ");
    let mut first = true;
    for g in 0..b.groups {
        if bit(&b.security, g) {
            let _ = write!(s, "{}({g})", if first { "" } else { "," });
            first = false;
        }
    }
    s.push_str(";\nINSERT INTO nest VALUES ");
    for (k, &(c, p)) in b.nested.iter().enumerate() {
        let _ = write!(s, "{}({c},{p})", if k == 0 { "" } else { "," });
    }
    s.push_str(";\nINSERT INTO grant_ VALUES ");
    let mut first = true;
    for g in 0..b.groups {
        for role in ones(&[b.grants[g]]) {
            let _ = write!(s, "{}({g},{role})", if first { "" } else { "," });
            first = false;
        }
    }
    s.push_str(";\n");
    for q in [
        "SELECT count(*) FROM m JOIN carries c ON c.g = m.g;",
        "SELECT count(DISTINCT (m.u, c.role)) FROM m JOIN carries c ON c.g = m.g;",
        "SELECT m.u, c.role, count(*) FROM m JOIN carries c ON c.g = m.g \
         GROUP BY 1,2 ORDER BY 1,2;",
    ] {
        s.push_str(CTE);
        s.push_str(q);
        s.push('\n');
    }

    let stdout = match psql(s) {
        Ok(o) => o,
        Err(e) => {
            println!("SKIP: {e}");
            return;
        }
    };
    let mut lines = stdout.lines().filter(|l| !l.is_empty());
    let pg_bag: u64 = lines.next().expect("count(*)").parse().expect("count");
    let pg_distinct: u64 = lines
        .next()
        .expect("count(distinct)")
        .parse()
        .expect("count");
    // PostgreSQL's per-coordinate rows are read straight into a lane indexed
    // `u * 64 + role` (an external reference's output, not a tuple answer).
    let mut pg_lane = vec![0u16; users * 64];
    for l in lines {
        let f: Vec<usize> = l.split(',').map(|x| x.parse().expect("int")).collect();
        pg_lane[f[0] * 64 + f[1]] = u16::try_from(f[2]).expect("count fits u16");
    }

    // Qr pages map user ordinal `u` to page `u / 256`, row `u % 256`
    // (`Qr::page` reads `b.user_groups.row(u)` for `u` in the page), so page
    // `p`, user `i` is ordinal `p * 256 + i`.
    let c = Carriers::of(&qr);
    let mut lane = vec![0u16; users * 64];
    let (mut bag, mut distinct) = (0u64, 0u64);
    for p in 0..qr.pages() {
        let n = users.saturating_sub(p * 256).min(256);
        let page = qr.page(&b, p).expect("page");
        let m = Members::of(&qr, &page, n);
        let out = compose(&m, &c, &w.witness(), Law::Count).expect("compose");
        lane[p * 256 * 64..][..n * 64].copy_from_slice(&out.counts);
        bag = add_bag(bag, out.bag).expect("no overflow");
        distinct = add_bag(distinct, out.distinct).expect("no overflow");
    }

    if lane != pg_lane {
        let at = lane.iter().zip(&pg_lane).position(|(a, b)| a != b);
        panic!("per-coordinate lane differs from PostgreSQL, first at {at:?}");
    }
    assert_eq!(bag, pg_bag, "bag vs count(*)");
    assert_eq!(distinct, pg_distinct, "distinct vs count(DISTINCT)");
    assert!(
        pg_bag > pg_distinct,
        "the run never reached the Bag stratum: {pg_bag} vs {pg_distinct}"
    );

    let version = Command::new("psql")
        .args(["-X", "-tA", "-c", "SHOW server_version"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "?".into());
    println!(
        "PostgreSQL {version}: lane equal over {users} users x 64 roles; \
         bag {bag} = count(*) {pg_bag}; distinct {distinct} = count(DISTINCT) {pg_distinct}"
    );
}
