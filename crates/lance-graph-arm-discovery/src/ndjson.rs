// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Newline-delimited JSON emit for [`CandidateTriple`].
//!
//! The on-the-wire shape is **byte-compatible** with `ruff_spo_triplet::ndjson`
//! and with what `lance_graph::graph::spo::odoo_ontology::parse_triples`
//! reads: one `{"s","p","o","f","c"}` object per line, field order
//! `s, p, o, f, c`, trailing newline. Emitting through this module is what
//! lets an ARM-discovered rule load into the SPO store through the *same*
//! loader as the static `ruff` extractor — no transform, no second format.
//!
//! This is a zero-dependency hand emitter (no `serde_json`) so the crate
//! stays std-only and offline-buildable. It produces the same logical shape;
//! the downstream loader parses it with `serde_json::from_str`.

use crate::translator::CandidateTriple;

/// Serialise candidate triples to ndjson (one object per line, trailing
/// newline). Order is preserved as given.
#[must_use]
pub fn to_ndjson(triples: &[CandidateTriple]) -> String {
    let mut out = String::new();
    for t in triples {
        out.push('{');
        push_str_field(&mut out, "s", &t.s, true);
        push_str_field(&mut out, "p", &t.p, false);
        push_str_field(&mut out, "o", &t.o, false);
        push_num_field(&mut out, "f", t.f, false);
        push_num_field(&mut out, "c", t.c, false);
        out.push('}');
        out.push('\n');
    }
    out
}

fn push_str_field(out: &mut String, key: &str, value: &str, first: bool) {
    if !first {
        out.push(',');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\":\"");
    escape_into(out, value);
    out.push('"');
}

fn push_num_field(out: &mut String, key: &str, value: f32, first: bool) {
    if !first {
        out.push(',');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\":");
    out.push_str(&fmt_f32(value));
}

/// Format an `f32` as a JSON-valid number, always with a fractional part so
/// the shape mirrors `serde_json`'s float rendering (`1` → `1.0`).
/// Non-finite values are coerced to `0.0` (defensive; truths are clamped to
/// `[0, 1]` upstream, so this never fires in practice).
fn fmt_f32(value: f32) -> String {
    if !value.is_finite() {
        return "0.0".to_string();
    }
    let s = format!("{value}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Minimal JSON string escaping for the characters that can appear in IRIs.
fn escape_into(out: &mut String, value: &str) {
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triple(s: &str, p: &str, o: &str, f: f32, c: f32) -> CandidateTriple {
        CandidateTriple {
            s: s.to_string(),
            p: p.to_string(),
            o: o.to_string(),
            f,
            c,
        }
    }

    #[test]
    fn line_shape_matches_spo_loader_contract() {
        let t = triple("arm:feat0=cat1", "implies", "arm:feat1=cat0", 0.9, 0.95);
        let text = to_ndjson(&[t]);
        // exact field order s, p, o, f, c — the shape parse_triples reads
        assert_eq!(
            text,
            "{\"s\":\"arm:feat0=cat1\",\"p\":\"implies\",\"o\":\"arm:feat1=cat0\",\"f\":0.9,\"c\":0.95}\n"
        );
    }

    #[test]
    fn integral_truth_renders_with_fraction() {
        // 1.0 must render as "1.0", not "1" — mirrors serde_json float output.
        let t = triple("a", "implies", "b", 1.0, 1.0);
        let text = to_ndjson(&[t]);
        assert!(text.contains("\"f\":1.0"), "got: {text}");
        assert!(text.contains("\"c\":1.0"), "got: {text}");
    }

    #[test]
    fn each_triple_is_one_line_with_trailing_newline() {
        let text = to_ndjson(&[
            triple("a", "implies", "b", 0.5, 0.5),
            triple("c", "implies", "d", 0.6, 0.6),
        ]);
        assert_eq!(text.lines().count(), 2);
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn quotes_and_backslashes_are_escaped() {
        let t = triple("a\"x", "implies", "b\\y", 0.5, 0.5);
        let text = to_ndjson(&[t]);
        assert!(text.contains("a\\\"x"), "quote escaped: {text}");
        assert!(text.contains("b\\\\y"), "backslash escaped: {text}");
    }
}
