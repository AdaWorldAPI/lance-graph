//! Runnable example for the D-V3-W6a "ONE two-metric range-count tool"
//! (`crate::classid_scan`): feed it a list of decoded `classid: u32` values
//! and it prints the [`AdoptionCounts`](lance_graph_contract::classid_scan::AdoptionCounts)
//! both routing.md §5 governance monitors read from — the SAME counting pass
//! feeds adoption% and the corpus-proof metric, never two separate scans.
//!
//! Per `.claude/v3/soa_layout/routing.md` §5 ("Monitor routing — adoption is
//! a range count"), the corpus-proof metric MUST scan ALL THREE legacy
//! classid shapes or it can falsely report a clean corpus while un-rebaked
//! render rows remain:
//!
//! - `0x0000_DDCC` — legacy core form (zero-prefix high)
//! - `0x1000_DDCC` — pre-flip V3-marker-high form
//! - `0xAAAA_DDCC` — legacy app/render-prefix-high form (e.g. MedCare's
//!   `0x0005_0901`)
//!
//! Scanning fewer than all three can falsely prove a corpus clean.
//!
//! ```sh
//! # built-in demo set (every ClassidForm variant classify_form can
//! # actually construct today, once each):
//! cargo run -p lance-graph-contract --example adoption_scan
//!
//! # scan real hex classids instead (with or without a 0x prefix):
//! cargo run -p lance-graph-contract --example adoption_scan -- 0x0700_0000 0x0000_0700
//! ```

#![allow(
    clippy::print_stdout,
    reason = "a scan CLI example writes to stdout by design"
)]

use std::process::ExitCode;

use lance_graph_contract::classid_scan::{classify_form, count_adoption};
use lance_graph_contract::ogar_codebook::{compose_classid_with, ClassidOrder};
use lance_graph_contract::{canonical_concept_id, AppPrefix, NodeGuid};

/// Parse one CLI arg as a `classid: u32` in hex, with or without a leading
/// `0x`/`0X` and with optional `_` digit-group separators (e.g.
/// `0x0700_0000`, matching how the same value is written as a Rust literal
/// elsewhere in this crate).
fn parse_hex_classid(arg: &str) -> Result<u32, String> {
    let trimmed = arg
        .strip_prefix("0x")
        .or_else(|| arg.strip_prefix("0X"))
        .unwrap_or(arg);
    let digits: String = trimmed.chars().filter(|c| *c != '_').collect();
    u32::from_str_radix(&digits, 16).map_err(|err| format!("{arg:?} is not a valid hex u32: {err}"))
}

/// The built-in demo set: one classid per `ClassidForm` variant that
/// `classify_form` can actually construct today — `CanonHigh`,
/// `LegacyZeroPrefixHigh`, `LegacyV3MarkerHigh`, `LegacyRenderPrefixHigh`.
///
/// `ClassidForm::Ambiguous` is deliberately NOT represented here:
/// `classify_form`'s own doc comment states it never returns `Ambiguous`
/// today (no `CODEBOOK` concept occupies domain `0x10`, the only domain
/// whose `CanonHigh` reading would collide with the `0x1000_DDCC` legacy
/// shape) — see `lance_graph_contract::classid_scan::ClassidForm::Ambiguous`.
///
/// Every id below is built ONLY through a sanctioned composer
/// (`compose_classid_with`) or a documented public constant
/// (`NodeGuid::CLASSID_*`) — never a hand-rolled composed-`u32` hex literal
/// — mirroring exactly how `classid_scan`'s own
/// `count_adoption_mixed_produces_correct_totals_and_pct` test builds its
/// mixed corpus (3 native, one of which is the degenerate default class,
/// plus one of each of the three legacy shapes).
fn demo_ids() -> Vec<(&'static str, u32)> {
    let concept = canonical_concept_id("patient").expect("patient is a CODEBOOK entry");
    vec![
        ("CLASSID_OSINT — native CanonHigh", NodeGuid::CLASSID_OSINT),
        (
            "CanonHigh + 0x1000 in custom — still native (marker lives in custom, not canon)",
            compose_classid_with(ClassidOrder::CanonHigh, concept, 0x1000),
        ),
        (
            "CLASSID_DEFAULT — degenerate default class, native",
            NodeGuid::CLASSID_DEFAULT,
        ),
        (
            "CLASSID_OSINT_LEGACY — 0x0000_DDCC (LegacyZeroPrefixHigh)",
            NodeGuid::CLASSID_OSINT_LEGACY,
        ),
        (
            "CanonLow + 0x1000 custom — 0x1000_DDCC (LegacyV3MarkerHigh)",
            compose_classid_with(ClassidOrder::CanonLow, concept, 0x1000),
        ),
        (
            "CanonLow + Healthcare app prefix — 0xAAAA_DDCC (LegacyRenderPrefixHigh)",
            compose_classid_with(
                ClassidOrder::CanonLow,
                concept,
                AppPrefix::Healthcare.prefix(),
            ),
        ),
    ]
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let ids: Vec<(String, u32)> = if args.is_empty() {
        demo_ids()
            .into_iter()
            .map(|(label, id)| (label.to_string(), id))
            .collect()
    } else {
        let mut parsed = Vec::with_capacity(args.len());
        for arg in &args {
            match parse_hex_classid(arg) {
                Ok(id) => parsed.push((arg.clone(), id)),
                Err(err) => {
                    eprintln!("error: {err}");
                    return ExitCode::from(2);
                }
            }
        }
        parsed
    };

    for (label, id) in &ids {
        println!("{id:#010x}  {:?}  ({label})", classify_form(*id));
    }

    let counts = count_adoption(ids.iter().map(|(_, id)| *id));
    println!("---");
    println!("total scanned: {}", counts.total);
    println!("canon_high:    {}", counts.canon_high);
    println!("old_form:      {}", counts.old_form);
    println!("ambiguous:     {}", counts.ambiguous);
    println!("adoption_pct:  {:.4}", counts.adoption_pct());
    println!("OLD-FORM ROWS: {}", counts.old_form);

    ExitCode::SUCCESS
}
