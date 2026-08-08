//! Parse a boot config with the REAL parser and print what a deployment would
//! see. Exists because "the YAML is well-formed" and "the deployment can boot
//! on it" are different claims, and only the second one matters.
//!
//! ```text
//! validate_soa_config <config.yaml>
//! ```

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 2 {
        eprintln!("usage: validate_soa_config <config.yaml>");
        std::process::exit(2);
    }
    let raw = std::fs::read_to_string(&a[1]).expect("read config");
    let cfg = match lance_graph::soa_config::parse(&raw) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("REJECTED by soa_config::parse: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "accepted: version {} ledger_prefix {}",
        cfg.version, cfg.ledger_prefix
    );
    for b in &cfg.bakes {
        println!(
            "  {:<12} table={:<20} classid=0x{:08X} hydrate={} digest={}",
            b.name,
            b.table,
            b.classid_u32().expect("classid parses"),
            b.hydrate,
            b.slab_digest.as_deref().unwrap_or("<unpinned>")
        );
    }
    println!(
        "on_existing: {:?}  bakes: {}",
        cfg.on_existing,
        cfg.bakes.len()
    );
}
