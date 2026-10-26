//! Replay the D-TFLP 6.5 softmax fixtures. Usage: replay FIXTURE_DIR [--reciprocal source|exact] [--full]
//!
//! Order: manifest digests (integrity) → each table's digest → each fixture. The reduced suite runs by
//! default; `--full` adds the complete 529-witness tier. Exit status 0 only if every check passes.
//! `--reciprocal exact` is the negative control: the witnesses must then fail and the
//! calibration-9 equivalence rows must still pass.
use replay::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path, process::ExitCode};

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn table(v: &Value) -> Table {
    let arr = |k: &str| v[k].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect::<Vec<_>>();
    let (adm, a, b) = (arr("admitted"), arr("A"), arr("B"));
    let mut t = Table { admitted: [false; 256], a: [0; 256], b: [0; 256] };
    for i in 0..256 {
        t.admitted[i] = adm[i] == 1;
        t.a[i] = a[i] as i32;
        t.b[i] = b[i] as i32;
    }
    t
}

fn row_pairs(v: &Value) -> Vec<(i32, i32)> {
    match v {
        Value::Object(m) => m.iter().map(|(k, n)| (k.parse().unwrap(), n.as_i64().unwrap() as i32)).collect(),
        Value::Array(a) => {
            let mut c: BTreeMap<i32, i32> = BTreeMap::new();
            for x in a {
                *c.entry(x.as_i64().unwrap() as i32).or_default() += 1;
            }
            c.into_iter().collect()
        }
        _ => panic!("row"),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let recip = if args.iter().any(|a| a == "exact") { Recip::Exact } else { Recip::Source };
    let full = args.iter().any(|a| a == "--full");
    let mut fails: Vec<String> = Vec::new();
    let t0 = std::time::Instant::now();
    // 1. integrity
    let man: Value = serde_json::from_str(&fs::read_to_string(dir.join("manifest65.json")).unwrap()).unwrap();
    for (name, want) in man["sha256"].as_object().unwrap() {
        let got = hex(&Sha256::digest(fs::read(dir.join(name)).unwrap()));
        if &got != want.as_str().unwrap() {
            fails.push(format!("integrity {name}"));
        }
    }
    if !fails.is_empty() {
        println!("REFUSE integrity: {fails:?}");
        return ExitCode::from(2);
    }
    let t_integrity = t0.elapsed();
    // 2. tables
    let tj: Value = serde_json::from_str(&fs::read_to_string(dir.join("softmax_tables65.json")).unwrap()).unwrap();
    let mut tables: BTreeMap<String, Table> = BTreeMap::new();
    for (key, t) in tj["tables"].as_object().unwrap() {
        let d = hex(&Sha256::digest(serde_json::to_string(t).unwrap()));
        if d != tj["sha256"][key].as_str().unwrap() {
            fails.push(format!("table digest {key}"));
        }
        tables.insert(key.clone(), table(t));
    }
    let t_tables = t0.elapsed() - t_integrity;
    // 3. fixtures
    let lines = |f: &str| fs::read_to_string(dir.join(f)).unwrap().lines().map(|l| serde_json::from_str::<Value>(l).unwrap()).collect::<Vec<_>>();
    let mut ok = BTreeMap::<&str, u32>::new();
    let mut check_w = |name: &'static str, ws: &[Value], fails: &mut Vec<String>| {
        for w in ws {
            let id = w["id"].as_str().unwrap();
            let key = w["table"].as_str().unwrap();
            if tj["sha256"][key] != w["table_sha256"] {
                fails.push(format!("{name} {id}: table_sha256"));
                continue;
            }
            let t = &tables[key];
            let row = row_pairs(&w["row"]);
            let d = w["d"].as_i64().unwrap() as i32;
            let r = (|| -> Result<(), String> {
                let s = row_sum(t, &row).map_err(|e| e.to_string())?;
                if s != w["S"].as_i64().unwrap() {
                    return Err(format!("sum {s} != stored S"));
                }
                if !row.iter().any(|&(c, _)| c == 127 + d) {
                    return Err("no element at d".into());
                }
                let bs = byte_of(t, &row, 127 + d, recip).map_err(|e| e.to_string())?;
                let bx = byte_of(t, &row, 127 + d, Recip::Exact).map_err(|e| e.to_string())?;
                if i64::from(bs) != w["byte_source"].as_i64().unwrap() || i64::from(bs) != w["kernel_byte"].as_i64().unwrap() {
                    return Err(format!("source byte {bs} != stored"));
                }
                if i64::from(bx) != w["byte_exact"].as_i64().unwrap() || bs == bx {
                    return Err(format!("exact byte {bx}"));
                }
                Ok(())
            })();
            match r {
                Ok(()) => *ok.entry(name).or_default() += 1,
                Err(e) => fails.push(format!("{name} {id}: {e}")),
            }
        }
    };
    check_w("reduced", &lines("witness65_reduced.jsonl"), &mut fails);
    if full {
        check_w("full", &lines("witness65_full.jsonl"), &mut fails);
    }
    for e in lines("equivalence65.jsonl") {
        let t = &tables[e["table"].as_str().unwrap()];
        let codes: Vec<i32> = e["row"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect();
        let row = row_pairs(&e["row"]);
        let want: Vec<i64> = e["kernel_bytes"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
        let got: Result<Vec<i64>, Refusal> = codes.iter().map(|&c| byte_of(t, &row, c, recip).map(i64::from)).collect();
        let got_x: Result<Vec<i64>, Refusal> = codes.iter().map(|&c| byte_of(t, &row, c, Recip::Exact).map(i64::from)).collect();
        if got.as_ref() == Ok(&want) && got_x.as_ref() == Ok(&want) {
            *ok.entry("equivalence").or_default() += 1;
        } else {
            fails.push(format!("equivalence {}", e["id"]));
        }
    }
    for u in lines("unreachable65.jsonl") {
        let t = &tables[u["table"].as_str().unwrap()];
        let (s, d) = (u["S"].as_i64().unwrap(), u["d"].as_i64().unwrap() as i32);
        let r = s - i64::from(t.b[255]) - if d != 0 { i64::from(t.b[(d + 255) as usize]) } else { 0 };
        let proved = match u["status"].as_str().unwrap() {
            "unreachable_negative" => r < 0,
            _ => r >= 0 && r <= 4_000_000 && !representable(t, r as usize),
        };
        if proved {
            *ok.entry("unreachable").or_default() += 1;
        } else {
            fails.push(format!("unreachable S={s} d={d}"));
        }
    }
    let t_total = t0.elapsed();
    eprintln!("timing_us integrity={} tables={} fixtures={} total={}", t_integrity.as_micros(), t_tables.as_micros(), (t_total - t_integrity - t_tables).as_micros(), t_total.as_micros());
    println!("recip={recip:?} passed={ok:?} failures={}", fails.len());
    for f in fails.iter().take(10) {
        println!("  {f}");
    }
    if fails.is_empty() { ExitCode::SUCCESS } else { ExitCode::from(1) }
}

/// Plain unbounded-coin DP over the admitted weights: is r a non-negative combination?
fn representable(t: &Table, r: usize) -> bool {
    let coins: Vec<usize> = (0..256).filter(|&i| t.admitted[i] && t.b[i] > 0).map(|i| t.b[i] as usize).collect();
    let mut rep = vec![false; r + 1];
    rep[0] = true;
    for x in 1..=r {
        rep[x] = coins.iter().any(|&c| c <= x && rep[x - c]);
    }
    rep[r]
}
