//! `probe_antecedent_binder` — W6: bind ALREADY-RESOLVED (pronoun_pos,
//! antecedent_pos) pairs into the `Locus::Antecedent` nibble of the
//! `CausalWitness` A9 lane, through the zero-copy [`WitnessLens`] producer/
//! consumer pair.
//!
//! This probe does NOT do coreference resolution — `deepnsm`'s
//! `spo_anaphora_nibble.rs` and `jc`'s `l9_loci_real_text.rs` already do that,
//! writing to a parallel local structure that never reaches the SoA lane. This
//! probe closes exactly that gap: given a GIVEN antecedent pair, write the
//! signed displacement into the row's `Locus::Antecedent` nibble and read it
//! back through [`WitnessLens::at`] — zero-copy, never a gather.
//!
//! **Escalate, never clamp.** A displacement of `0` (self-reference) or
//! outside the `±8` window is refused — the row's `Locus::Antecedent` nibble
//! stays `0` (unbound, the zero-fallback sentinel) and the binder records an
//! escalation instead of writing a saturated value. Saturating an
//! out-of-range offset to `±8`/`±7` would silently point the pronoun at the
//! WRONG event — the defect this probe exists to prevent.
//!
//! Usage: `cargo run -p lance-graph-planner --example probe_antecedent_binder`

use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow, ValueTenant};
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::witness_fabric::WitnessLens;

/// One fixture token: a stream position, a display label, and — for pronoun
/// tokens — the position of the antecedent it should bind to (`None` for
/// non-pronoun tokens).
struct Token {
    label: &'static str,
    antecedent_pos: Option<usize>,
}

/// A small fixture stream with pronouns at varying, mostly in-window
/// distances from their antecedents, plus one deliberately-out-of-window
/// case and one deliberately self-referential case (both must escalate, not
/// clamp).
fn fixture_stream() -> Vec<Token> {
    vec![
        Token {
            label: "Maria",
            antecedent_pos: None,
        }, // 0
        Token {
            label: "arrived",
            antecedent_pos: None,
        }, // 1
        Token {
            label: "late",
            antecedent_pos: None,
        }, // 2
        Token {
            label: "she",
            antecedent_pos: Some(0),
        }, // 3 -> offset -3 (in range)
        Token {
            label: "apologized",
            antecedent_pos: None,
        }, // 4
        Token {
            label: "quietly",
            antecedent_pos: None,
        }, // 5
        Token {
            label: "Tom",
            antecedent_pos: None,
        }, // 6
        Token {
            label: "smiled",
            antecedent_pos: None,
        }, // 7
        Token {
            label: "and",
            antecedent_pos: None,
        }, // 8
        Token {
            label: "he",
            antecedent_pos: Some(6),
        }, // 9 -> offset -3 (in range, same magnitude as #3 — distinct pair below covers distinctness)
        Token {
            label: "nodded",
            antecedent_pos: None,
        }, // 10
        Token {
            label: "it",
            antecedent_pos: Some(9),
        }, // 11 -> offset -2 (in range, distinct)
        Token {
            label: "himself",
            antecedent_pos: Some(11),
        }, // 12 -> offset -1 (in range, distinct — self? no, distinct target)
        // A pronoun whose antecedent is a full 21 tokens back — well outside
        // the ±8 window. Filler tokens 13..23 pad the distance.
        Token {
            label: "filler",
            antecedent_pos: None,
        }, // 13
        Token {
            label: "filler",
            antecedent_pos: None,
        }, // 14
        Token {
            label: "filler",
            antecedent_pos: None,
        }, // 15
        Token {
            label: "filler",
            antecedent_pos: None,
        }, // 16
        Token {
            label: "filler",
            antecedent_pos: None,
        }, // 17
        Token {
            label: "filler",
            antecedent_pos: None,
        }, // 18
        Token {
            label: "filler",
            antecedent_pos: None,
        }, // 19
        Token {
            label: "filler",
            antecedent_pos: None,
        }, // 20
        Token {
            label: "filler",
            antecedent_pos: None,
        }, // 21
        Token {
            label: "filler",
            antecedent_pos: None,
        }, // 22
        Token {
            label: "filler",
            antecedent_pos: None,
        }, // 23
        Token {
            label: "they", // pos 24, antecedent at pos 3 -> offset -21 (out of ±8 window)
            antecedent_pos: Some(3),
        }, // 24
        // A self-referential (d == 0) pronoun — must ALSO escalate (zero is
        // the unbound sentinel, never a legal "offset zero, meaning self").
        Token {
            label: "itself",
            antecedent_pos: Some(25),
        }, // 25 -> offset 0
    ]
}

/// The outcome of attempting to bind one pronoun's antecedent into the lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindOutcome {
    /// Wrote `offset` into `Locus::Antecedent` at `pos`.
    Bound { pos: usize, offset: i8 },
    /// Refused — `pos` stays unbound (nibble `0`). `reason` is human-readable.
    Escalated { pos: usize, reason: &'static str },
}

/// Compute the signed displacement and decide bind vs. escalate. Mirrors the
/// shipped resolvers' "escalate, never clamp" contract (`clamp_off` in
/// `jc/examples/l9_loci_real_text.rs`, despite its name, already refuses
/// rather than saturates): `d == 0` (self-reference) or `d` outside `-8..=7`
/// both escalate; only an in-range, nonzero `d` binds.
fn decide(self_pos: usize, antecedent_pos: usize) -> Result<i8, &'static str> {
    let d = antecedent_pos as i64 - self_pos as i64;
    if d == 0 {
        return Err("self-referential (d == 0), the unbound sentinel — refused");
    }
    if !(-8..=7).contains(&d) {
        return Err("antecedent outside the ±8 Markov window — refused, not clamped");
    }
    Ok(d as i8)
}

#[cfg(test)]
mod decide_tests {
    use super::decide;

    #[test]
    fn rejects_self_reference() {
        assert!(decide(5, 5).is_err(), "d == 0 must be refused, not bound");
    }

    #[test]
    fn rejects_positive_out_of_range() {
        // d = 8 - 0 = +8, one past the +7 boundary.
        assert!(decide(0, 8).is_err());
    }

    #[test]
    fn rejects_negative_out_of_range() {
        // d = 0 - 9 = -9, one past the -8 boundary.
        assert!(decide(9, 0).is_err());
    }

    #[test]
    fn accepts_lower_boundary() {
        // d = 0 - 8 = -8, the most-negative in-range displacement.
        assert_eq!(decide(8, 0), Ok(-8));
    }

    #[test]
    fn accepts_upper_boundary() {
        // d = 7 - 0 = +7, the most-positive in-range displacement.
        assert_eq!(decide(0, 7), Ok(7));
    }
}

fn main() {
    let stream = fixture_stream();
    let n = stream.len();

    // Build the row array: one NodeRow per token, ZERO-initialized value slab.
    let mut rows: Vec<NodeRow> = (0..n)
        .map(|i| NodeRow {
            key: NodeGuid::local(i as u32 + 1),
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        })
        .collect();

    // Canary the whole slab first so A5 (write isolation) has something
    // falsifiable to compare against — 0xEE everywhere except the 12-byte
    // CausalWitness register, which we leave at 0 to keep `decide`'s
    // zero-fallback semantics legible in the snapshot diff.
    let cw_start = ValueTenant::CausalWitness.value_offset();
    let cw_end = cw_start + ValueTenant::CausalWitness.byte_len();
    for row in &mut rows {
        for (i, b) in row.value.iter_mut().enumerate() {
            if i < cw_start || i >= cw_end {
                *b = 0xEE;
            }
        }
    }

    // Snapshot every row's FULL 480-byte value slab BEFORE any binder write
    // (for A5 write-isolation).
    let before: Vec<[u8; 480]> = rows.iter().map(|r| r.value).collect();
    // …and the key + edge blocks too. A5's claim is about the whole 512-byte
    // row; checking only the value slab would leave the claim unbacked for
    // the 32 bytes of key+edges (a doc-comment claim is not a behaviour).
    let before_keys: Vec<NodeGuid> = rows.iter().map(|r| r.key).collect();
    let before_edges: Vec<EdgeBlock> = rows.iter().map(|r| r.edges).collect();

    // ── Run the binder over every pronoun token ─────────────────────────
    let mut outcomes: Vec<BindOutcome> = Vec::new();
    for pos in 0..n {
        let Some(antecedent_pos) = stream[pos].antecedent_pos else {
            continue;
        };
        match decide(pos, antecedent_pos) {
            Ok(offset) => {
                let facet = CausalWitnessFacet::ZERO.with(Locus::Antecedent, offset);
                WitnessLens::write_register(&mut rows[pos], &facet);
                outcomes.push(BindOutcome::Bound { pos, offset });
            }
            Err(reason) => {
                // Explicitly leave the row's register untouched (ZERO,
                // i.e. Locus::Antecedent reads 0/unbound) — no write at all.
                outcomes.push(BindOutcome::Escalated { pos, reason });
            }
        }
    }

    println!("── Fixture stream ({n} tokens) ──");
    for (i, tok) in stream.iter().enumerate() {
        if let Some(a) = tok.antecedent_pos {
            println!("  [{i:>2}] {:<10} -> antecedent @ {a}", tok.label);
        }
    }

    println!("\n── Binder outcomes ──");
    for o in &outcomes {
        match o {
            BindOutcome::Bound { pos, offset } => {
                println!("  [{pos:>2}] BOUND   offset={offset}")
            }
            BindOutcome::Escalated { pos, reason } => {
                println!("  [{pos:>2}] ESCALATE {reason}")
            }
        }
    }

    let lens = WitnessLens::new(&rows);

    // ═══════════════════ A1 — round-trip ═══════════════════
    // Every BOUND outcome must read back identical through the lens.
    let bound: Vec<(usize, i8)> = outcomes
        .iter()
        .filter_map(|o| match o {
            BindOutcome::Bound { pos, offset } => Some((*pos, *offset)),
            BindOutcome::Escalated { .. } => None,
        })
        .collect();
    let a1_roundtrip_ok = bound.iter().all(|&(pos, offset)| {
        let facet = lens.at(pos).expect("pos in range");
        facet.at(Locus::Antecedent) == offset
    });
    // Anti-vacuity: at least 3 DISTINCT nonzero offsets must occur, or the
    // round-trip proves nothing beyond "reads back a single repeated value".
    let distinct_offsets: std::collections::BTreeSet<i8> = bound.iter().map(|&(_, o)| o).collect();
    let a1_distinct_ok = distinct_offsets.len() >= 3;
    let a1_pass = a1_roundtrip_ok && a1_pass_nonzero(&bound) && a1_distinct_ok;
    println!(
        "\nA1 round-trip: roundtrip_ok={a1_roundtrip_ok} distinct_offsets={distinct_offsets:?} (>=3: {a1_distinct_ok})"
    );

    // ═══════════════════ A2 — escalation fires ═══════════════════
    // The out-of-window pronoun (pos 24) must escalate AND its
    // Locus::Antecedent nibble must read 0 (unbound), not a saturated value.
    let escalated: Vec<usize> = outcomes
        .iter()
        .filter_map(|o| match o {
            BindOutcome::Escalated { pos, .. } => Some(*pos),
            BindOutcome::Bound { .. } => None,
        })
        .collect();
    let out_of_window_pos = 24usize;
    let a2_fires = escalated.contains(&out_of_window_pos);
    let a2_nibble_zero = lens
        .at(out_of_window_pos)
        .expect("pos in range")
        .at(Locus::Antecedent)
        == 0;
    let a2_pass = a2_fires && a2_nibble_zero;
    println!(
        "A2 escalation fires: pos={out_of_window_pos} escalated={a2_fires} nibble_zero(not saturated)={a2_nibble_zero}"
    );

    // ═══════════════════ A3 — escalation stays silent ═══════════════════
    // Every IN-RANGE, non-self-referential pronoun must produce ZERO
    // escalations. Build an all-in-range sub-fixture (positions 3,9,11,12)
    // and confirm none of them escalated.
    let all_in_range_positions = [3usize, 9, 11, 12];
    let a3_pass = all_in_range_positions
        .iter()
        .all(|p| !escalated.contains(p) && bound.iter().any(|&(pos, _)| pos == *p));
    println!("A3 escalation silent on in-range set {all_in_range_positions:?}: {a3_pass}");

    // ═══════════════════ A4 — zero is unbound ═══════════════════
    // The self-referential pronoun (pos 25, d == 0) must escalate with the
    // self-referential reason, and its nibble must read 0.
    let self_ref_pos = 25usize;
    let a4_escalated = escalated.contains(&self_ref_pos);
    let a4_reason_is_self = outcomes.iter().any(|o| {
        matches!(o, BindOutcome::Escalated { pos, reason } if *pos == self_ref_pos && reason.contains("self-referential"))
    });
    let a4_nibble_zero = lens
        .at(self_ref_pos)
        .expect("pos in range")
        .at(Locus::Antecedent)
        == 0;
    let a4_pass = a4_escalated && a4_reason_is_self && a4_nibble_zero;
    println!(
        "A4 zero is unbound: pos={self_ref_pos} escalated={a4_escalated} self_ref_reason={a4_reason_is_self} nibble_zero={a4_nibble_zero}"
    );

    // ═══════════════════ A5 — write isolation ═══════════════════
    // Writing the antecedent nibble at each bound position must change ONLY
    // that nibble — every other byte of the 512-byte row (key + edges +
    // rest of value, including the OTHER 11 bytes of the CausalWitness
    // register) must be unchanged from the canary snapshot.
    // `cw_start` is the CausalWitness TENANT start (the 4-byte facet_classid
    // prefix + the 12-byte register); the register itself begins 4 bytes in
    // (`WITNESS_FACET_CLASSID_BYTES`, private to `witness_fabric`), which
    // `WitnessLens` derives internally. We don't reach for that private
    // constant here — instead derive the same offset the lens uses via the
    // one-and-only zero-copy write path: write a KNOWN facet through
    // `WitnessLens::write_register` to a scratch row and diff against a
    // ZERO row to locate the touched byte, so this probe never hardcodes a
    // literal offset either.
    let probe_zero_row = NodeRow {
        key: NodeGuid::local(0),
        edges: EdgeBlock::default(),
        value: [0u8; 480],
    };
    let mut probe_written_row = probe_zero_row;
    WitnessLens::write_register(
        &mut probe_written_row,
        &CausalWitnessFacet::ZERO.with(Locus::Antecedent, -1),
    );
    let antecedent_byte_offset = (0..480)
        .find(|&i| probe_zero_row.value[i] != probe_written_row.value[i])
        .expect("write_register must touch at least one byte");
    let mut a5_pass = true;
    for &(pos, _) in &bound {
        let before_slab = &before[pos];
        let after_slab = &rows[pos].value;
        for i in 0..480 {
            let changed = before_slab[i] != after_slab[i];
            let should_change = i == antecedent_byte_offset;
            if changed != should_change {
                a5_pass = false;
                println!(
                    "  A5 FAIL: row {pos} value byte {i} changed={changed} expected_change={should_change}"
                );
            }
        }
        // The other 32 bytes of the 512-byte row: a lane write must not
        // touch the key (16 B) or the edge block (16 B) at all.
        if rows[pos].key != before_keys[pos] {
            a5_pass = false;
            println!("  A5 FAIL: row {pos} KEY changed — a lane write must never touch the key");
        }
        if rows[pos].edges != before_edges[pos] {
            a5_pass = false;
            println!(
                "  A5 FAIL: row {pos} EDGE BLOCK changed — a lane write must never touch edges"
            );
        }
    }
    println!(
        "A5 write isolation over {} bound rows: {a5_pass}",
        bound.len()
    );

    // ═══════════════════ Report ═══════════════════
    let gates: [(&str, bool, String); 5] = [
        (
            "A1",
            a1_pass,
            format!(
                "roundtrip_ok={a1_roundtrip_ok} distinct={}",
                distinct_offsets.len()
            ),
        ),
        (
            "A2",
            a2_pass,
            format!("fires={a2_fires} nibble_zero={a2_nibble_zero}"),
        ),
        (
            "A3",
            a3_pass,
            format!("silent_over={all_in_range_positions:?}"),
        ),
        (
            "A4",
            a4_pass,
            format!("escalated={a4_escalated} nibble_zero={a4_nibble_zero}"),
        ),
        ("A5", a5_pass, format!("bound_rows={}", bound.len())),
    ];

    println!("\n════════ GATES ════════");
    let mut all_pass = true;
    for (name, pass, detail) in &gates {
        println!(
            "  [{}] {name} — {detail}",
            if *pass { "PASS" } else { "FAIL" }
        );
        all_pass &= *pass;
    }
    if all_pass {
        println!("\nALL GATES GREEN");
    } else {
        panic!("PROBE-ANTECEDENT-BINDER: one or more gates FAILED — see report above");
    }
}

/// Every bound offset must be nonzero (the binder never writes the unbound
/// sentinel as a "bound" outcome — that would be a logic error, not merely
/// an escalation miss).
fn a1_pass_nonzero(bound: &[(usize, i8)]) -> bool {
    bound.iter().all(|&(_, offset)| offset != 0)
}
