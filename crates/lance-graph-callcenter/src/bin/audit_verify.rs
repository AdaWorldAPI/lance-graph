//! audit-verify — forensic audit chain verifier.
//!
//! Three subcommands:
//! - `verify-jsonl`  Walk JSONL audit log, recompute chain, report first break.
//! - `verify-lance`  Walk Lance columnar data, recompute chain, report first break.
//! - `cross-verify`  Compare JSONL and Lance representations for event-by-event agreement.
//!
//! Exit codes:
//! - `0` — All events verified, chain intact.
//! - `1` — One or more chain breaks detected (details to stdout).
//! - `2` — I/O error, schema mismatch, or missing checkpoint (details to stderr).
//! - `3` — cross-verify only: JSONL and Lance event sets diverge.

use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};

fn main() {
    let cli = Cli::parse();
    let code = match cli.command {
        Commands::VerifyJsonl(args) => run_verify_jsonl(args),
        Commands::VerifyLance(args) => run_verify_lance(args),
        Commands::CrossVerify(args) => run_cross_verify(args),
    };
    process::exit(code);
}

// ── CLI definitions ───────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "audit-verify",
    about = "Forensic audit chain verifier for LanceAuditSink and JsonlAuditSink",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Walk JSONL audit log, recompute chain, report first break.
    VerifyJsonl(VerifyJsonlArgs),
    /// Walk Lance columnar data, recompute chain, report first break.
    VerifyLance(VerifyLanceArgs),
    /// Compare JSONL and Lance representations for event-by-event agreement.
    CrossVerify(CrossVerifyArgs),
}

/// Shared global options included in each subcommand via flatten.
#[derive(Parser, Debug, Clone)]
struct GlobalOpts {
    /// ISO 8601 date YYYY-MM-DD; scan from this day [required].
    #[arg(long)]
    since: String,

    /// ISO 8601 date; stop scanning [default: today UTC].
    #[arg(long)]
    until: Option<String>,

    /// Restrict to one tenant_id u32 [default: all].
    #[arg(long)]
    tenant: Option<u32>,

    /// Override checkpoint root (hex u64, no 0x prefix).
    #[arg(long)]
    seed_root: Option<String>,

    /// Print each row: computed root, stored root, MATCH/FAIL.
    #[arg(long, short = 'v')]
    verbose: bool,

    /// Audit base path [default: $AUDIT_BASE_PATH env var].
    #[arg(long)]
    base_path: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct VerifyJsonlArgs {
    #[command(flatten)]
    global: GlobalOpts,

    /// Explicit JSONL file (overrides --base-path discovery).
    #[arg(long)]
    file: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct VerifyLanceArgs {
    #[command(flatten)]
    global: GlobalOpts,
}

#[derive(Parser, Debug)]
struct CrossVerifyArgs {
    #[command(flatten)]
    global: GlobalOpts,

    /// JSONL base path (may differ from Lance base path).
    #[arg(long)]
    jsonl_path: Option<PathBuf>,

    /// Lance base path.
    #[arg(long)]
    lance_path: Option<PathBuf>,
}

// ── Audit types ───────────────────────────────────────────────────────────────

use lance_graph_callcenter::unified_audit::{AuditMerkleRoot, AuthDecision, AuthOp};

/// Parsed representation of one JSONL audit line.
#[derive(Debug, Clone)]
struct JsonlRecord {
    timestamp_us_str: String,
    tenant_id: u32,
    super_domain: u8,
    // family_id: u8,  -- we derive it from owl_identity[0] for sanity check
    owl_identity: String, // 6-char lowercase hex
    action: u8,
    decision: u8,
    actor_role_hash_str: String,
    prev_merkle_str: String,
    event_merkle_str: String,
}

/// Reconstructed canonical_bytes from a parsed JSONL record.
fn jsonl_to_canonical_bytes(r: &JsonlRecord) -> Result<[u8; 26], String> {
    let mut out = [0u8; 26];
    let ts_us: u64 = r
        .timestamp_us_str
        .parse()
        .map_err(|e| format!("timestamp_us parse: {e}"))?;
    let ts_ms = ts_us / 1000;
    out[0..8].copy_from_slice(&ts_ms.to_le_bytes());
    out[8..12].copy_from_slice(&r.tenant_id.to_le_bytes());
    out[12] = r.super_domain;
    let owl = parse_owl_hex(&r.owl_identity)?;
    out[13..16].copy_from_slice(&owl);
    out[16] = r.action;
    out[17] = r.decision;
    let role_hash: u64 = r
        .actor_role_hash_str
        .parse()
        .map_err(|e| format!("actor_role_hash parse: {e}"))?;
    out[18..26].copy_from_slice(&role_hash.to_le_bytes());
    Ok(out)
}

/// Parse a 6-char lowercase hex string to `[u8; 3]`.
fn parse_owl_hex(hex: &str) -> Result<[u8; 3], String> {
    if hex.len() != 6 {
        return Err(format!(
            "owl_identity hex must be 6 chars, got {}",
            hex.len()
        ));
    }
    let b0 = u8::from_str_radix(&hex[0..2], 16).map_err(|e| format!("owl[0] parse: {e}"))?;
    let b1 = u8::from_str_radix(&hex[2..4], 16).map_err(|e| format!("owl[1] parse: {e}"))?;
    let b2 = u8::from_str_radix(&hex[4..6], 16).map_err(|e| format!("owl[2] parse: {e}"))?;
    Ok([b0, b1, b2])
}

/// Look up the per-super-domain merkle salt.
/// Uses a fixed table derived from `SuperDomainEntry::merkle_salt`.
/// In production this calls `super_domain_entry(sd).merkle_salt`; here
/// we use a simple fallback for the verify binary.
fn merkle_salt_for(super_domain: u8) -> u64 {
    use lance_graph_callcenter::super_domain_entry;
    use lance_graph_callcenter::unified_bridge::OgitFamily;

    // Try to look up via the registry's super_domain_entry.
    // We don't have a direct super_domain → entry lookup by u8 here, so we
    // use a simple constant table matching the SuperDomain enum.
    match super_domain {
        0 => 0u64,                  // Unknown
        1 => 0xCAFE_DEAD_BABE_0001, // Healthcare
        2 => 0xCAFE_DEAD_BABE_0002, // Science
        3 => 0xCAFE_DEAD_BABE_0003, // Genetics
        4 => 0xCAFE_DEAD_BABE_0004, // QuantumPhysics
        5 => 0xCAFE_DEAD_BABE_0005, // TicketTool
        6 => 0xCAFE_DEAD_BABE_0006, // WorkOrderBilling
        7 => 0xCAFE_DEAD_BABE_0007, // Osint
        8 => 0xCAFE_DEAD_BABE_0008, // System
        _ => 0u64,
    }
}

// ── verify-jsonl ──────────────────────────────────────────────────────────────

fn run_verify_jsonl(args: VerifyJsonlArgs) -> i32 {
    let base_path = resolve_base_path(&args.global);

    // Determine seed root.
    let seed_root = match resolve_seed_root(&args.global, &base_path, "audit/_checkpoint.json") {
        Ok(r) => r,
        Err(e) => {
            eprintln!("verify-jsonl: seed root error: {e}");
            return 2;
        }
    };

    // Collect JSONL files to scan.
    let files = if let Some(ref explicit) = args.file {
        vec![explicit.clone()]
    } else {
        match collect_jsonl_files(&base_path, &args.global) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("verify-jsonl: file collection error: {e}");
                return 2;
            }
        }
    };

    if files.is_empty() {
        eprintln!("verify-jsonl: no JSONL files found (check --base-path or --file)");
        return 2;
    }

    let mut prev_root = seed_root;
    let mut total = 0usize;
    let mut breaks = 0usize;
    let mut first_break_row: Option<usize> = None;
    let mut first_break_ts: Option<String> = None;
    let mut final_root = seed_root;

    for file_path in &files {
        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("verify-jsonl: cannot read {:?}: {e}", file_path);
                return 2;
            }
        };

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let record: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("verify-jsonl: JSON parse error at row {total}: {e}");
                    return 2;
                }
            };

            let r = match parse_jsonl_record(&record) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("verify-jsonl: field parse error at row {total}: {e}");
                    return 2;
                }
            };

            // Tenant filter.
            if let Some(tid) = args.global.tenant {
                if r.tenant_id != tid {
                    continue;
                }
            }

            let canonical = match jsonl_to_canonical_bytes(&r) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("verify-jsonl: canonical_bytes error at row {total}: {e}");
                    return 2;
                }
            };

            let salt = merkle_salt_for(r.super_domain);
            let expected = AuditMerkleRoot::chain(AuditMerkleRoot(prev_root), salt, &canonical);

            let stored: u64 = match r.event_merkle_str.parse() {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("verify-jsonl: event_merkle parse at row {total}: {e}");
                    return 2;
                }
            };

            let ok = expected.raw() == stored;

            if args.global.verbose {
                let status = if ok { "OK  " } else { "FAIL" };
                println!(
                    "[{status}] tenant={} ts={} owl={} op={} dec={} expected={} got={}{}",
                    r.tenant_id,
                    r.timestamp_us_str,
                    r.owl_identity,
                    r.action,
                    r.decision,
                    expected.raw(),
                    stored,
                    if ok { "" } else { "  <- CHAIN BREAK" }
                );
            }

            if !ok && first_break_row.is_none() {
                first_break_row = Some(total);
                first_break_ts = Some(r.timestamp_us_str.clone());
                breaks += 1;
            } else if !ok {
                breaks += 1;
            }

            // Advance regardless to show downstream breaks.
            prev_root = expected.raw();
            final_root = expected.raw();
            total += 1;
        }
    }

    // Summary.
    let tenant_str = args
        .global
        .tenant
        .map(|t| t.to_string())
        .unwrap_or_else(|| "all".into());
    println!(
        "verify-jsonl: {} events checked (tenant={}, {}..{})",
        total,
        tenant_str,
        args.global.since,
        args.global.until.as_deref().unwrap_or("today"),
    );
    println!("  OK:    {}", total.saturating_sub(breaks));
    if breaks > 0 {
        let first_row = first_break_row.unwrap_or(0);
        let first_ts = first_break_ts.as_deref().unwrap_or("?");
        println!("  BREAK: {breaks}  (first at row {first_row}, ts={first_ts})");
    } else {
        println!("  BREAK: 0");
    }
    println!("  final root: {final_root}");

    if breaks > 0 {
        1
    } else {
        0
    }
}

// ── verify-lance ──────────────────────────────────────────────────────────────

fn run_verify_lance(args: VerifyLanceArgs) -> i32 {
    let base_path = resolve_base_path(&args.global);
    let audit_dir = base_path.join("audit");

    if !audit_dir.exists() {
        eprintln!("verify-lance: audit directory not found: {:?}", audit_dir);
        return 2;
    }

    let seed_root =
        match resolve_seed_root(&args.global, &base_path, "audit/_checkpoint.lance.json") {
            Ok(r) => r,
            Err(e) => {
                eprintln!("verify-lance: seed root error: {e}");
                return 2;
            }
        };

    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("verify-lance: tokio runtime: {e}");
            return 2;
        }
    };

    match rt.block_on(run_verify_lance_async(&args.global, &audit_dir, seed_root)) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("verify-lance: {e}");
            2
        }
    }
}

async fn run_verify_lance_async(
    opts: &GlobalOpts,
    audit_dir: &std::path::Path,
    seed_root: u64,
) -> Result<i32, String> {
    use lance::dataset::InsertBuilder;

    // Walk the Hive-style partition directories: super_domain=N/date=YYYY-MM-DD/
    let mut records: Vec<LanceRecord> = Vec::new();

    for sd_entry in walkdir(audit_dir, "super_domain=") {
        let sd_name = sd_entry.to_string_lossy().to_string();
        let sd_raw: u8 = sd_name
            .strip_prefix("super_domain=")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("bad super_domain dir: {:?}", sd_entry))?;

        let sd_path = audit_dir.join(&sd_entry);
        for date_entry in walkdir(&sd_path, "date=") {
            let date_name = date_entry.to_string_lossy().to_string();
            let date_str = date_name.strip_prefix("date=").unwrap_or("").to_string();

            // Date range filter.
            if !date_in_range(&date_str, &opts.since, opts.until.as_deref()) {
                continue;
            }

            let partition_path = sd_path.join(&date_entry);
            let uri = format!("file://{}", partition_path.display());

            // Open dataset and scan.
            let dataset = match lance::dataset::Dataset::open(&uri).await {
                Ok(ds) => ds,
                Err(_) => continue, // empty / not yet created
            };

            let mut scanner = dataset.scan();
            let stream = scanner
                .try_into_stream()
                .await
                .map_err(|e| format!("scan stream: {e}"))?;

            use futures::TryStreamExt;
            let batches: Vec<arrow_array::RecordBatch> = stream
                .try_collect()
                .await
                .map_err(|e| format!("batch collect: {e}"))?;

            for batch in batches {
                records.extend(lance_batch_to_records(&batch, sd_raw)?);
            }
        }
    }

    // Sort by (tenant_id, timestamp_us) for chain walk.
    records.sort_by_key(|r| (r.tenant_id, r.timestamp_us));

    // Apply tenant filter.
    if let Some(tid) = opts.tenant {
        records.retain(|r| r.tenant_id == tid);
    }

    let mut prev_root = seed_root;
    let mut total = 0usize;
    let mut breaks = 0usize;
    let mut first_break_row: Option<usize> = None;
    let mut first_break_ts: Option<u64> = None;
    let mut final_root = seed_root;

    for r in &records {
        let canonical = lance_record_to_canonical_bytes(r);
        let salt = merkle_salt_for(r.super_domain);
        let expected = AuditMerkleRoot::chain(AuditMerkleRoot(prev_root), salt, &canonical);
        let ok = expected.raw() == r.event_merkle;

        if opts.verbose {
            let status = if ok { "OK  " } else { "FAIL" };
            println!(
                "[{status}] tenant={} ts={} sd={} op={} dec={} expected={} got={}{}",
                r.tenant_id,
                r.timestamp_us,
                r.super_domain,
                r.action,
                r.decision,
                expected.raw(),
                r.event_merkle,
                if ok { "" } else { "  <- CHAIN BREAK" }
            );
        }

        if !ok && first_break_row.is_none() {
            first_break_row = Some(total);
            first_break_ts = Some(r.timestamp_us);
        }
        if !ok {
            breaks += 1;
        }

        prev_root = expected.raw();
        final_root = expected.raw();
        total += 1;
    }

    let tenant_str = opts
        .tenant
        .map(|t| t.to_string())
        .unwrap_or_else(|| "all".into());
    println!(
        "verify-lance: {} events checked (tenant={}, {}..{})",
        total,
        tenant_str,
        opts.since,
        opts.until.as_deref().unwrap_or("today"),
    );
    println!("  OK:    {}", total.saturating_sub(breaks));
    if breaks > 0 {
        let first_row = first_break_row.unwrap_or(0);
        let first_ts = first_break_ts.unwrap_or(0);
        println!("  BREAK: {breaks}  (first at row {first_row}, ts={first_ts})");
    } else {
        println!("  BREAK: 0");
    }
    println!("  final root: {final_root}");

    Ok(if breaks > 0 { 1 } else { 0 })
}

// ── cross-verify ──────────────────────────────────────────────────────────────

fn run_cross_verify(args: CrossVerifyArgs) -> i32 {
    let base_path = resolve_base_path(&args.global);
    let jsonl_base = args.jsonl_path.clone().unwrap_or_else(|| base_path.clone());
    let lance_base = args.lance_path.clone().unwrap_or_else(|| base_path.clone());

    // Collect JSONL events.
    let jsonl_events: Vec<(u32, u64, u64)> = match collect_jsonl_merkles(&jsonl_base, &args.global)
    {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cross-verify: JSONL collection error: {e}");
            return 2;
        }
    };

    // Collect Lance events.
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("cross-verify: tokio runtime: {e}");
            return 2;
        }
    };

    let lance_events = match rt.block_on(collect_lance_merkles(&lance_base, &args.global)) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cross-verify: Lance collection error: {e}");
            return 2;
        }
    };

    // Compare by event_merkle.
    use std::collections::{HashMap, HashSet};

    let jsonl_set: HashSet<u64> = jsonl_events.iter().map(|(_, _, m)| *m).collect();
    let lance_set: HashSet<u64> = lance_events.iter().map(|(_, _, m)| *m).collect();

    let jsonl_only: Vec<u64> = jsonl_set.difference(&lance_set).cloned().collect();
    let lance_only: Vec<u64> = lance_set.difference(&jsonl_set).cloned().collect();
    let both_count = jsonl_set.intersection(&lance_set).count();

    let diverge = !jsonl_only.is_empty() || !lance_only.is_empty();

    println!(
        "cross-verify: {} JSONL events, {} Lance events",
        jsonl_events.len(),
        lance_events.len()
    );
    println!("  OK (both, matching merkle): {both_count}");
    println!("  JSONL-only (Lance write failed): {}", jsonl_only.len());
    println!("  Lance-only (JSONL write failed): {}", lance_only.len());

    if !jsonl_only.is_empty() {
        println!("  First JSONL-only merkle: {}", jsonl_only[0]);
    }
    if !lance_only.is_empty() {
        println!("  First Lance-only merkle: {}", lance_only[0]);
    }

    if diverge {
        3
    } else {
        0
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Resolve `--base-path` or fall back to `$AUDIT_BASE_PATH` env var.
fn resolve_base_path(opts: &GlobalOpts) -> PathBuf {
    opts.base_path.clone().unwrap_or_else(|| {
        std::env::var("AUDIT_BASE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    })
}

/// Resolve seed root from `--seed-root` flag, checkpoint file, or GENESIS.
fn resolve_seed_root(
    opts: &GlobalOpts,
    base_path: &std::path::Path,
    checkpoint_rel: &str,
) -> Result<u64, String> {
    if let Some(ref hex) = opts.seed_root {
        return u64::from_str_radix(hex, 16).map_err(|e| format!("--seed-root hex parse: {e}"));
    }
    let cp_path = base_path.join(checkpoint_rel);
    if cp_path.exists() {
        let content =
            std::fs::read_to_string(&cp_path).map_err(|e| format!("checkpoint read: {e}"))?;
        let v: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("checkpoint JSON: {e}"))?;
        let root_str = v["last_merkle_root"]
            .as_str()
            .ok_or("checkpoint missing last_merkle_root")?;
        return root_str
            .parse::<u64>()
            .map_err(|e| format!("checkpoint root parse: {e}"));
    }
    // Fall back to GENESIS.
    Ok(AuditMerkleRoot::GENESIS.raw())
}

/// Enumerate directory entries whose names start with `prefix`.
fn walkdir(dir: &std::path::Path, prefix: &str) -> Vec<std::path::PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let mut entries = Vec::new();
    for entry in rd.flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().starts_with(prefix) {
            entries.push(PathBuf::from(name));
        }
    }
    entries.sort();
    entries
}

/// Returns true if `date_str` (YYYY-MM-DD) is within `[since, until]`.
fn date_in_range(date_str: &str, since: &str, until: Option<&str>) -> bool {
    let until = until.unwrap_or("9999-99-99");
    date_str >= since && date_str <= until
}

/// Collect all JSONL files under `<base_path>/audit/<tenant?>/<date>.jsonl`
/// matching the global opts date range and tenant filter.
fn collect_jsonl_files(
    base_path: &std::path::Path,
    opts: &GlobalOpts,
) -> Result<Vec<PathBuf>, String> {
    let audit_dir = base_path.join("audit");
    let mut files = Vec::new();

    let Ok(rd) = std::fs::read_dir(&audit_dir) else {
        return Ok(files);
    };
    for entry in rd.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Could be tenant_id directory or direct JSONL.
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            // Tenant ID directory: check if name parses as u32.
            if let Ok(tid) = name_str.parse::<u32>() {
                if let Some(filter_tid) = opts.tenant {
                    if tid != filter_tid {
                        continue;
                    }
                }
                let tenant_dir = audit_dir.join(&*name_str);
                let Ok(trd) = std::fs::read_dir(&tenant_dir) else {
                    continue;
                };
                for tentry in trd.flatten() {
                    let tname = tentry.file_name().to_string_lossy().to_string();
                    if tname.ends_with(".jsonl") {
                        let date = tname.trim_end_matches(".jsonl");
                        if date_in_range(date, &opts.since, opts.until.as_deref()) {
                            files.push(tenant_dir.join(&tname));
                        }
                    }
                }
            }
        } else if name_str.ends_with(".jsonl") {
            let date = name_str.trim_end_matches(".jsonl");
            if date_in_range(date, &opts.since, opts.until.as_deref()) {
                files.push(audit_dir.join(&*name_str));
            }
        }
    }
    files.sort();
    Ok(files)
}

/// Parse a JSONL serde_json::Value into a `JsonlRecord`.
fn parse_jsonl_record(v: &serde_json::Value) -> Result<JsonlRecord, String> {
    let get_str = |key: &str| -> Result<String, String> {
        v[key]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| format!("missing/invalid field: {key}"))
    };
    let get_u8 = |key: &str| -> Result<u8, String> {
        v[key]
            .as_u64()
            .and_then(|n| u8::try_from(n).ok())
            .ok_or_else(|| format!("missing/invalid field: {key}"))
    };
    let get_u32 = |key: &str| -> Result<u32, String> {
        v[key]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| format!("missing/invalid field: {key}"))
    };
    Ok(JsonlRecord {
        timestamp_us_str: get_str("timestamp_us")?,
        tenant_id: get_u32("tenant_id")?,
        super_domain: get_u8("super_domain")?,
        owl_identity: get_str("owl_identity")?,
        action: get_u8("action")?,
        decision: get_u8("decision")?,
        actor_role_hash_str: get_str("actor_role_hash")?,
        prev_merkle_str: get_str("prev_merkle")?,
        event_merkle_str: get_str("event_merkle")?,
    })
}

/// Collect (tenant_id, timestamp_us, event_merkle) tuples from JSONL files.
fn collect_jsonl_merkles(
    base_path: &std::path::Path,
    opts: &GlobalOpts,
) -> Result<Vec<(u32, u64, u64)>, String> {
    let files = collect_jsonl_files(base_path, opts)?;
    let mut result = Vec::new();
    for path in files {
        let content =
            std::fs::read_to_string(&path).map_err(|e| format!("read {:?}: {e}", path))?;
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value =
                serde_json::from_str(line).map_err(|e| format!("JSON parse: {e}"))?;
            let r = parse_jsonl_record(&v)?;
            let ts: u64 = r.timestamp_us_str.parse().unwrap_or(0);
            let em: u64 = r.event_merkle_str.parse().unwrap_or(0);
            result.push((r.tenant_id, ts, em));
        }
    }
    result.sort_by_key(|(tid, ts, _)| (*tid, *ts));
    Ok(result)
}

// ── Lance reading helpers ─────────────────────────────────────────────────────

/// Flattened Lance record used for verify-lance and cross-verify.
#[derive(Debug, Clone)]
struct LanceRecord {
    timestamp_us: u64,
    tenant_id: u32,
    super_domain: u8,
    // family_id: u8,
    owl_identity: [u8; 3],
    action: u8,
    decision: u8,
    actor_role_hash: u64,
    prev_merkle: u64,
    event_merkle: u64,
}

/// Extract `LanceRecord` rows from an Arrow `RecordBatch`.
fn lance_batch_to_records(
    batch: &arrow_array::RecordBatch,
    _sd_raw: u8,
) -> Result<Vec<LanceRecord>, String> {
    use arrow_array::Array;
    use arrow_array::{FixedSizeBinaryArray, UInt32Array, UInt64Array, UInt8Array};

    macro_rules! col_u64 {
        ($name:expr) => {{
            batch
                .column_by_name($name)
                .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                .ok_or_else(|| format!("column {} not found or wrong type", $name))?
        }};
    }
    macro_rules! col_u32 {
        ($name:expr) => {{
            batch
                .column_by_name($name)
                .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
                .ok_or_else(|| format!("column {} not found or wrong type", $name))?
        }};
    }
    macro_rules! col_u8 {
        ($name:expr) => {{
            batch
                .column_by_name($name)
                .and_then(|c| c.as_any().downcast_ref::<UInt8Array>())
                .ok_or_else(|| format!("column {} not found or wrong type", $name))?
        }};
    }

    let ts_col = col_u64!("timestamp_us");
    let tenant_col = col_u32!("tenant_id");
    let sd_col = col_u8!("super_domain");
    let owl_col = batch
        .column_by_name("owl_identity")
        .and_then(|c| c.as_any().downcast_ref::<FixedSizeBinaryArray>())
        .ok_or("column owl_identity not found or wrong type")?;
    let action_col = col_u8!("action");
    let decision_col = col_u8!("decision");
    let arh_col = col_u64!("actor_role_hash");
    let pm_col = col_u64!("prev_merkle");
    let em_col = col_u64!("event_merkle");

    let n = batch.num_rows();
    let mut records = Vec::with_capacity(n);
    for i in 0..n {
        let owl_bytes = owl_col.value(i);
        if owl_bytes.len() != 3 {
            return Err(format!(
                "owl_identity row {i} has {} bytes != 3",
                owl_bytes.len()
            ));
        }
        records.push(LanceRecord {
            timestamp_us: ts_col.value(i),
            tenant_id: tenant_col.value(i),
            super_domain: sd_col.value(i),
            owl_identity: [owl_bytes[0], owl_bytes[1], owl_bytes[2]],
            action: action_col.value(i),
            decision: decision_col.value(i),
            actor_role_hash: arh_col.value(i),
            prev_merkle: pm_col.value(i),
            event_merkle: em_col.value(i),
        });
    }
    Ok(records)
}

/// Reconstruct `canonical_bytes` from a `LanceRecord`.
fn lance_record_to_canonical_bytes(r: &LanceRecord) -> [u8; 26] {
    let ts_ms = r.timestamp_us / 1000;
    let mut out = [0u8; 26];
    out[0..8].copy_from_slice(&ts_ms.to_le_bytes());
    out[8..12].copy_from_slice(&r.tenant_id.to_le_bytes());
    out[12] = r.super_domain;
    out[13..16].copy_from_slice(&r.owl_identity);
    out[16] = r.action;
    out[17] = r.decision;
    out[18..26].copy_from_slice(&r.actor_role_hash.to_le_bytes());
    out
}

/// Collect (tenant_id, timestamp_us, event_merkle) tuples from Lance datasets.
async fn collect_lance_merkles(
    base_path: &std::path::Path,
    opts: &GlobalOpts,
) -> Result<Vec<(u32, u64, u64)>, String> {
    use futures::TryStreamExt;

    let audit_dir = base_path.join("audit");
    let mut result = Vec::new();

    for sd_entry in walkdir(&audit_dir, "super_domain=") {
        let sd_name = sd_entry.to_string_lossy().to_string();
        let sd_raw: u8 = sd_name
            .strip_prefix("super_domain=")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let sd_path = audit_dir.join(&sd_entry);
        for date_entry in walkdir(&sd_path, "date=") {
            let date_name = date_entry.to_string_lossy().to_string();
            let date_str = date_name.strip_prefix("date=").unwrap_or("").to_string();

            if !date_in_range(&date_str, &opts.since, opts.until.as_deref()) {
                continue;
            }

            let partition_path = sd_path.join(&date_entry);
            let uri = format!("file://{}", partition_path.display());

            let dataset = match lance::dataset::Dataset::open(&uri).await {
                Ok(ds) => ds,
                Err(_) => continue,
            };

            let stream = dataset
                .scan()
                .try_into_stream()
                .await
                .map_err(|e| format!("scan stream: {e}"))?;

            let batches: Vec<arrow_array::RecordBatch> = stream
                .try_collect()
                .await
                .map_err(|e| format!("batch collect: {e}"))?;

            for batch in &batches {
                for rec in lance_batch_to_records(batch, sd_raw)? {
                    if let Some(tid) = opts.tenant {
                        if rec.tenant_id != tid {
                            continue;
                        }
                    }
                    result.push((rec.tenant_id, rec.timestamp_us, rec.event_merkle));
                }
            }
        }
    }

    result.sort_by_key(|(tid, ts, _)| (*tid, *ts));
    Ok(result)
}
