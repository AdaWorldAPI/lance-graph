//! Negative controls for the fixture replay: each tamper must make the runner fail for its own reason.
//! Each test copies the committed fixtures, changes one thing, and (where stated) re-stamps the
//! manifest digest so that the SEMANTIC check — not the integrity check — is what must fail.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, process::Command};

const SRC: &str = "../../source-behavior-v1/fixtures65";

fn copy(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("replay65-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    for e in fs::read_dir(SRC).unwrap() {
        let e = e.unwrap();
        fs::copy(e.path(), d.join(e.file_name())).unwrap();
    }
    d
}

fn restamp(d: &PathBuf, name: &str) {
    let p = d.join("manifest65.json");
    let mut m: Value = serde_json::from_str(&fs::read_to_string(&p).unwrap()).unwrap();
    let h: String = Sha256::digest(fs::read(d.join(name)).unwrap()).iter().map(|b| format!("{b:02x}")).collect();
    m["sha256"][name] = Value::String(h);
    fs::write(&p, serde_json::to_string_pretty(&m).unwrap()).unwrap();
}

/// Rewrite the first line of a JSONL fixture through `f`.
fn edit_first(d: &PathBuf, name: &str, f: impl Fn(&mut Value)) {
    let p = d.join(name);
    let text = fs::read_to_string(&p).unwrap();
    let mut lines: Vec<String> = text.lines().map(String::from).collect();
    let mut v: Value = serde_json::from_str(&lines[0]).unwrap();
    f(&mut v);
    lines[0] = serde_json::to_string(&v).unwrap();
    fs::write(&p, lines.join("\n") + "\n").unwrap();
}

fn run(d: &PathBuf, extra: &[&str]) -> (i32, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_replay")).arg(d).args(extra).output().unwrap();
    (o.status.code().unwrap(), String::from_utf8_lossy(&o.stdout).into_owned())
}

#[test]
fn clean_fixtures_pass_full_tier() {
    let (c, out) = run(&PathBuf::from(SRC), &["--full"]);
    assert_eq!(c, 0, "{out}");
    assert!(out.contains("\"full\": 529") && out.contains("\"reduced\": 27"), "{out}");
}

#[test]
fn changed_source_byte_fails_semantically() {
    let d = copy("byte");
    edit_first(&d, "witness65_reduced.jsonl", |v| v["byte_source"] = Value::from(v["byte_source"].as_i64().unwrap() + 1));
    restamp(&d, "witness65_reduced.jsonl");
    let (c, out) = run(&d, &[]);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("source byte"), "{out}");
}

#[test]
fn changed_byte_without_restamp_fails_integrity() {
    let d = copy("integrity");
    edit_first(&d, "witness65_reduced.jsonl", |v| v["byte_source"] = Value::from(v["byte_source"].as_i64().unwrap() + 1));
    let (c, out) = run(&d, &[]);
    assert_eq!(c, 2, "{out}");
    assert!(out.contains("REFUSE integrity"), "{out}");
}

#[test]
fn negative_multiplicity_is_rejected() {
    let d = copy("neg");
    edit_first(&d, "witness65_reduced.jsonl", |v| {
        let row = v["row"].as_object_mut().unwrap();
        let k = row.keys().next().unwrap().clone();
        row.insert(k, Value::from(-1));
    });
    restamp(&d, "witness65_reduced.jsonl");
    let (c, out) = run(&d, &[]);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("NonPositiveMultiplicity"), "{out}");
}

#[test]
fn unsupported_domain_is_rejected() {
    let d = copy("domain");
    // 600 maxima: S = 600 * 2^19 > 2^28, outside the supported domain
    edit_first(&d, "witness65_reduced.jsonl", |v| {
        v["row"].as_object_mut().unwrap().insert("127".into(), Value::from(600));
    });
    restamp(&d, "witness65_reduced.jsonl");
    let (c, out) = run(&d, &[]);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("UnsupportedDomain"), "{out}");
}

#[test]
fn tampered_table_fails_its_digest() {
    let d = copy("table");
    let p = d.join("softmax_tables65.json");
    let mut t: Value = serde_json::from_str(&fs::read_to_string(&p).unwrap()).unwrap();
    let key = t["tables"].as_object().unwrap().keys().next().unwrap().clone();
    let b0 = t["tables"][&key]["B"][100].as_i64().unwrap();
    t["tables"][&key]["B"][100] = Value::from(b0 + 1);
    fs::write(&p, serde_json::to_string_pretty(&t).unwrap()).unwrap();
    restamp(&d, "softmax_tables65.json");
    let (c, out) = run(&d, &[]);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("table digest"), "{out}");
}

#[test]
fn exact_reciprocal_fails_every_witness_but_keeps_equivalence_rows() {
    let (c, out) = run(&PathBuf::from(SRC), &["--reciprocal", "exact", "--full"]);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("failures=556"), "{out}");
    assert!(out.contains("\"equivalence\": 7"), "{out}");
}

#[test]
fn missing_mandatory_maximum_is_rejected() {
    let d = copy("nomax");
    edit_first(&d, "witness65_reduced.jsonl", |v| {
        v["row"].as_object_mut().unwrap().remove("127");
    });
    restamp(&d, "witness65_reduced.jsonl");
    let (c, out) = run(&d, &[]);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("no mandatory maximum"), "{out}");
}

#[test]
fn incorrect_computed_sum_is_rejected() {
    let d = copy("sum");
    edit_first(&d, "witness65_reduced.jsonl", |v| v["S"] = Value::from(v["S"].as_i64().unwrap() + 1));
    restamp(&d, "witness65_reduced.jsonl");
    let (c, out) = run(&d, &[]);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("!= stored S"), "{out}");
}

/// The witness keeps its table and digest but claims another calibration: the identity check must fire,
/// not a byte comparison that might happen to agree.
#[test]
fn wrong_calibration_identity_is_rejected() {
    let d = copy("ident");
    edit_first(&d, "witness65_reduced.jsonl", |v| v["scale_bits"] = Value::from("3dcccccd"));
    restamp(&d, "witness65_reduced.jsonl");
    let (c, out) = run(&d, &[]);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("calibration identity"), "{out}");
}

/// The witness points at another calibration's table but keeps its own table digest.
#[test]
fn witness_pointing_at_another_table_is_rejected() {
    let d = copy("othertable");
    let tj: Value = serde_json::from_str(&fs::read_to_string(d.join("softmax_tables65.json")).unwrap()).unwrap();
    edit_first(&d, "witness65_reduced.jsonl", |v| {
        let own = v["table"].as_str().unwrap().to_string();
        let other = tj["tables"].as_object().unwrap().keys().find(|k| **k != own).unwrap().clone();
        v["table"] = Value::String(other);
    });
    restamp(&d, "witness65_reduced.jsonl");
    let (c, out) = run(&d, &[]);
    assert_eq!(c, 1, "{out}");
    assert!(out.contains("table_sha256"), "{out}");
}
