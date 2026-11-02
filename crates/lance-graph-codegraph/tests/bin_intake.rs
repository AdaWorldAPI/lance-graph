//! Tests for the zero-copy `.rows.bin` reader (`artifact_bin`), over the
//! golden file copied from the ruff writer.

use std::io::Read;
use std::path::PathBuf;

use lance_graph_codegraph::artifact::digest;
use lance_graph_codegraph::artifact_bin::{tag, BinRefusal, CrateFile, Header, HEADER_LEN};

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/golden")
        .join(name)
}

fn golden_bin() -> Vec<u8> {
    std::fs::read(golden_path("cg_lib.rows.bin")).expect("read golden bin")
}

fn golden_tsv() -> String {
    std::fs::read_to_string(golden_path("cg_lib.rows.tsv")).expect("read golden tsv")
}

/// Recompute the header digest over `[32..]` so later checks are isolated.
fn reseal(mut b: Vec<u8>) -> Vec<u8> {
    let d = digest(&b[32..]);
    b[16..24].copy_from_slice(&d.to_le_bytes());
    b
}

fn rd32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().expect("4 bytes"))
}

/// `(directory position, tag, offset, len)` of every directory entry.
fn dir(b: &[u8]) -> Vec<(usize, u32, usize, usize)> {
    let n = rd32(b, 44) as usize;
    (0..n)
        .map(|i| {
            let at = HEADER_LEN + i * 16;
            (
                at,
                rd32(b, at),
                rd32(b, at + 8) as usize,
                rd32(b, at + 12) as usize,
            )
        })
        .collect()
}

fn entry(b: &[u8], t: u32) -> (usize, usize, usize) {
    let (at, _, off, len) = dir(b)
        .into_iter()
        .find(|e| e.1 == t)
        .expect("tag in directory");
    (at, off, len)
}

fn refusal(b: Vec<u8>) -> BinRefusal {
    CrateFile::parse(b).expect_err("must be refused")
}

/// Red if the digest folds a different length encoding than the ruff writer's.
#[test]
fn digest_matches_the_ruff_golden_vectors() {
    assert_eq!(digest(b""), 0xd0dd_3f66_b426_3ceb);
    assert_eq!(digest(b"a"), 0x9fc7_5a5c_9bce_51be);
    assert_eq!(digest(b"CHRNROWS digest golden v1"), 0xfe0a_b233_a400_27ce);
}

/// Red if any accessor disagrees with the TSV the same writer emitted, or if
/// the fixture stops containing a dispatch row or a siteless row.
#[test]
fn the_ruff_golden_parses_and_matches_its_tsv() {
    let f = CrateFile::parse(golden_bin()).expect("golden parses");
    let tsv = golden_tsv();
    let lines: Vec<&str> = tsv
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .collect();
    assert!(!lines.is_empty());
    assert_eq!(f.n_rows(), lines.len());

    let mut saw_may = false;
    let mut saw_no_site = false;
    for (row, line) in lines.iter().enumerate() {
        let c: Vec<&str> = line.split('\t').collect();
        assert_eq!(c.len(), 7, "row {row}");
        assert_eq!(f.predicate(row), c[0], "predicate row {row}");
        assert_eq!(f.symbol(f.subject(row)), c[1], "subject row {row}");
        assert_eq!(f.symbol(f.object(row)), c[2], "object row {row}");
        assert_eq!(f.grade(row), c[3], "grade row {row}");
        if c[4] == "-" {
            saw_no_site = true;
            assert_eq!(f.file(row), None, "file row {row}");
            assert_eq!(f.span(row), None, "span row {row}");
            assert_eq!((c[5], c[6]), ("-", "-"));
        } else {
            assert_eq!(f.file(row), Some(c[4]), "file row {row}");
            let want = (c[5].parse::<u32>().unwrap(), c[6].parse::<u32>().unwrap());
            assert_eq!(f.span(row), Some(want), "span row {row}");
        }
        if c[0] == "MAY_DISPATCH" {
            saw_may = true;
            assert!(f.may_target(row).is_some(), "may_target row {row}");
        } else {
            assert_eq!(f.may_target(row), None, "may_target row {row}");
        }
        assert_eq!(f.obj_kind(row), 1);
        assert!(f.predicate_id(row) >= 1);
    }
    assert!(saw_may, "golden must contain a MAY_DISPATCH row");
    assert!(saw_no_site, "golden must contain a row with no site");

    // Symbol side: hashes and owners are consistent with the labels.
    for id in 0..u32::try_from(f.n_symbols()).unwrap() {
        let label = f.symbol(id);
        assert_eq!(f.sym_hash(id), digest(label.as_bytes()), "hash {id}");
        match (label.find("::"), f.sym_owner(id)) {
            (Some(p), Some(o)) => assert_eq!(f.symbol(o), &label[..p]),
            (None, None) => {}
            other => panic!("owner mismatch for {label}: {other:?}"),
        }
    }
}

struct Counting<'a> {
    inner: &'a [u8],
    requested: usize,
}

impl Read for Counting<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.requested += buf.len();
        self.inner.read(buf)
    }
}

/// Red if `Header::read` asks for more than 64 bytes or returns another digest.
#[test]
fn the_header_alone_is_enough_for_the_unchanged_check() {
    let bytes = golden_bin();
    let mut r = Counting {
        inner: &bytes,
        requested: 0,
    };
    let h = Header::read(&mut r).expect("io").expect("header");
    assert_eq!(r.requested, HEADER_LEN);
    let full = CrateFile::parse(bytes.clone()).expect("parse");
    assert_eq!(h.digest, full.header().digest);
    assert_eq!(h, *full.header());
    assert_eq!(h.file_len, bytes.len() as u64);
}

/// Red if a flipped body byte is not caught by the digest before decoding.
#[test]
fn a_flipped_body_byte_is_a_digest_refusal() {
    let mut b = golden_bin();
    let last = b.len() - 9;
    b[last] ^= 1;
    assert!(matches!(refusal(b), BinRefusal::Digest { .. }));
}

/// Red if a cut file is accepted or reported as anything but Truncated.
#[test]
fn a_truncated_file_is_refused() {
    let mut b = golden_bin();
    b.truncate(b.len() - 8);
    assert!(matches!(refusal(b), BinRefusal::Truncated { .. }));
    assert!(matches!(
        refusal(golden_bin()[..10].to_vec()),
        BinRefusal::Truncated {
            header: 0,
            actual: 10
        }
    ));
}

/// Red if a newer major version is read.
#[test]
fn a_newer_major_is_refused() {
    let mut b = golden_bin();
    b[8] += 1;
    assert_eq!(refusal(reseal(b)), BinRefusal::Major(2));
}

/// Red if a newer minor version is read.
#[test]
fn a_newer_minor_is_refused() {
    let mut b = golden_bin();
    b[10] += 1;
    assert_eq!(refusal(reseal(b)), BinRefusal::Minor(1));
}

/// Red if a non-zero reserved header byte is ignored.
#[test]
fn a_reserved_byte_is_refused() {
    let mut b = golden_bin();
    b[50] = 1;
    assert_eq!(refusal(reseal(b)), BinRefusal::Reserved);
}

/// Red if a wrong magic is accepted.
#[test]
fn a_wrong_magic_is_refused() {
    let mut b = golden_bin();
    b[0] ^= 0xFF;
    assert_eq!(refusal(reseal(b)), BinRefusal::Magic);
}

/// Red if a subject id equal to `n_symbols` is not an IdRange refusal.
#[test]
fn a_subject_id_past_the_symbols_is_refused() {
    let mut b = golden_bin();
    let n_syms = rd32(&b, 36);
    let (_, off, _) = entry(&b, tag::SUBJ);
    b[off..off + 4].copy_from_slice(&n_syms.to_le_bytes());
    assert_eq!(
        refusal(reseal(b)),
        BinRefusal::IdRange {
            tag: tag::SUBJ,
            row: 0
        }
    );
}

/// Red if a section whose offset passes the end of the file is accepted.
#[test]
fn a_section_past_the_file_end_is_refused() {
    let mut b = golden_bin();
    let (at, _, _) = entry(&b, tag::OBJ);
    let past = u32::try_from(b.len() + 8).unwrap();
    b[at + 8..at + 12].copy_from_slice(&past.to_le_bytes());
    assert_eq!(refusal(reseal(b)), BinRefusal::Section { tag: tag::OBJ });
}

/// Red if an unknown must-understand section is skipped.
#[test]
fn an_unknown_required_tag_is_refused() {
    let mut b = golden_bin();
    let (at, _, _) = entry(&b, tag::PROV);
    let unknown = 0x8000_00FFu32;
    b[at..at + 4].copy_from_slice(&unknown.to_le_bytes());
    assert_eq!(
        refusal(reseal(b)),
        BinRefusal::UnknownRequired { tag: unknown }
    );
}

/// Red if a file without a SUBJ section parses. The entry is removed from the
/// directory (the later entries shift up, the count drops by one).
#[test]
fn a_missing_required_section_is_refused() {
    let mut b = golden_bin();
    let (at, _, _) = entry(&b, tag::SUBJ);
    let n = rd32(&b, 44);
    let dir_end = HEADER_LEN + n as usize * 16;
    b.copy_within(at + 16..dir_end, at);
    // The freed last slot is now dead space between the directory and the
    // first section; keep the file length by zeroing it.
    for x in &mut b[dir_end - 16..dir_end] {
        *x = 0;
    }
    b[44..48].copy_from_slice(&(n - 1).to_le_bytes());
    assert_eq!(
        refusal(reseal(b)),
        BinRefusal::MissingSection { tag: tag::SUBJ }
    );
}

/// Red if the provenance record is not read back as the writer wrote it.
#[test]
fn prov_reads_the_writer_record() {
    let f = CrateFile::parse(golden_bin()).expect("parse");
    let p = f.prov();
    assert_eq!(p.len(), 1);
    assert!(p[0].charon_version.is_some());
    assert_eq!(p[0].charon_commit, None);
}

/// Red if a function row loses its declaration index or the CRATE row gains one.
#[test]
fn fundecl_is_present_on_function_rows() {
    let f = CrateFile::parse(golden_bin()).expect("parse");
    let mut crate_rows = 0;
    for row in 0..f.n_rows() {
        if f.predicate(row) == "CRATE" {
            crate_rows += 1;
            assert_eq!(f.fundecl(row), None, "CRATE row {row}");
        } else {
            assert!(f.fundecl(row).is_some(), "row {row}");
        }
    }
    assert!(crate_rows >= 1, "golden must contain the CRATE row");
}
