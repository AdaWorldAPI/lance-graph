//! Corpus text → verses. The inbound leg's text handling lives here, in the
//! library, so `cargo test` actually exercises it.
//!
//! This was inline in `examples/bible_wave.rs`, where it carried a truncation
//! bug for its entire life **and could not be unit-tested**: `cargo test`
//! compiles examples but never runs their `main()`, and the corpus is not
//! committed. Extracting it is what lets the three `***` cases below be gated
//! by CI on synthetic fixtures instead of by one manual run.

/// The Project Gutenberg end-of-ebook fence, matched on its **full** text.
///
/// Deliberately not a bare `***`. See [`split_verses`].
pub const GUTENBERG_FOOTER: &str = "*** END OF THE PROJECT GUTENBERG";

/// The Old Testament's verse count in the KJV — the exact truncation point of
/// the historical bug.
///
/// **Documentation only. It is NOT a threshold.**
/// [`crossed_into_new_testament`] used to compare against it and that was
/// wrong in both directions (see that function's table); the gate now reads
/// the boundary from the parse instead. Kept because the number is the
/// external fact that made the bug legible — it is not authored here, and a
/// parse landing exactly on it is the bug's signature.
pub const KJV_OLD_TESTAMENT_VERSES: usize = 23_145;

/// Parse a `d+:d+` verse marker (e.g. `1:1`, `22:21`) into `(chapter, verse)`.
///
/// This is the ONE rule; [`is_verse_marker`] delegates to it, so the predicate
/// and the parse can never disagree about what a marker is. The explicit
/// ascii-digit guard is load-bearing: `u16::from_str` accepts a leading `+`,
/// which would make `+1:1` a marker here but not under the old predicate.
/// Out-of-`u16` components (`99999:1`) are not markers — the widest real book
/// is 150 chapters.
#[must_use]
pub fn parse_verse_marker(tok: &str) -> Option<(u16, u16)> {
    let (c, v) = tok.split_once(':')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(c) || !digits(v) {
        return None;
    }
    Some((c.parse().ok()?, v.parse().ok()?))
}

/// Is `tok` a `d+:d+` verse marker? See [`parse_verse_marker`].
#[must_use]
pub fn is_verse_marker(tok: &str) -> bool {
    parse_verse_marker(tok).is_some()
}

/// Split Gutenberg-formatted scripture into verses: a whitespace token shaped
/// `d+:d+` opens a new verse, and everything up to the next marker is its text.
///
/// # The three `***`, which are NOT interchangeable
///
/// | | |
/// |---|---|
/// | header | `*** START OF THE PROJECT GUTENBERG EBOOK 10 ***` — at **character 0** of the real file, so breaking on the *first* `***` returns nothing |
/// | separator | a **bare `***`** on its own line, between the testaments |
/// | footer | `*** END OF THE PROJECT GUTENBERG EBOOK 10 ***` |
///
/// The original `tok.contains("***") => break` stopped at the **separator**, so
/// the whole pipeline only ever saw the Old Testament — 39 books,
/// [`KJV_OLD_TESTAMENT_VERSES`] verses, ending at Malachi 4:6 — while its gate
/// still reported "whole book". Correct handling: truncate at
/// [`GUTENBERG_FOOTER`] matched in full, and skip a token that is **exactly**
/// `***` (not merely all-asterisks — `*` and `**` are ordinary body tokens and
/// deleting them would silently corrupt verse text).
#[must_use]
pub fn split_verses(text: &str) -> Vec<String> {
    split_verses_detailed(text).verses
}

/// The outcome of a [`split_verses_detailed`] walk: the verses, plus what the
/// walk **observed** while producing them.
///
/// `crossed_new_testament` exists so the gate in
/// [`crossed_into_new_testament`] can be answered from the parse itself rather
/// than by comparing a count against [`KJV_OLD_TESTAMENT_VERSES`]. A count
/// comparison is wrong in two directions on legitimate input: a New-Testament-
/// only corpus has *fewer* verses than the Old Testament and would read as
/// "did not cross", and an uppercase heading would not match a case-sensitive
/// search at all.
#[derive(Debug, Clone, Default)]
pub struct CorpusSplit {
    /// The verses, in order.
    pub verses: Vec<String>,
    /// The `(chapter, verse)` marker that OPENED each verse, positionally
    /// aligned with `verses` — `markers[i]` opened `verses[i]`.
    ///
    /// Collected in the SAME walk that produces `verses`, which is the point:
    /// a caller that needs the markers must not re-scan the raw text with its
    /// own `split_whitespace().filter(is_verse_marker)` loop. Such a loop does
    /// not apply the footer trim and would admit any `d+:d+` token from the
    /// Gutenberg header or license footer as if it were scripture. On the
    /// shipped KJV file that second walk happens to be clean (measured: zero
    /// markers outside the body), but nothing in the file's structure
    /// guarantees it — the alignment here does.
    pub markers: Vec<(u16, u16)>,
    /// Did the walk emit at least one verse that *began* after a New Testament
    /// heading? Detected case-insensitively, from the token stream.
    ///
    /// "Began after", not "flushed after": a verse that started before the
    /// heading and flushed after it is Old Testament text and must not set
    /// this flag (see `a_parse_that_stops_at_the_heading_fails_the_gate`).
    pub crossed_new_testament: bool,
}

/// Is `tok`, stripped of surrounding punctuation and lowercased, equal to
/// `want`? Used to spot the `New Testament` heading without allocating a
/// lowercase copy of the whole corpus.
fn tok_eq_ci(tok: &str, want: &str) -> bool {
    let t = tok.trim_matches(|c: char| !c.is_alphanumeric());
    t.len() == want.len() && t.to_ascii_lowercase() == want
}

/// [`split_verses`], plus the boundary metadata the New-Testament gate needs.
///
/// See [`split_verses`] for the `***` handling, which is the historically
/// load-bearing part.
#[must_use]
pub fn split_verses_detailed(text: &str) -> CorpusSplit {
    let body = match text.find(GUTENBERG_FOOTER) {
        Some(i) => &text[..i],
        None => text,
    };

    let mut verses: Vec<String> = Vec::new();
    let mut markers: Vec<(u16, u16)> = Vec::new();
    let mut cur = String::new();
    let mut in_body = false;
    // Two-token state machine over "new" "testament", case-insensitive.
    let mut saw_new = false;
    let mut nt_heading_seen = false;
    let mut crossed = false;
    // Did the verse currently accumulating in `cur` *start* after the heading?
    // Reading `nt_heading_seen` at flush time is WRONG: a verse that began
    // before the heading and flushed after it would be credited to the New
    // Testament. `"1:1 old verse The New Testament"` is the minimal case — one
    // verse, entirely pre-heading, flushed at end-of-input with the heading
    // already seen. Reading the flag at flush called that `crossed`, so a parse
    // that truncates AT the heading — exactly the historical bug G1b exists to
    // catch — passed the gate.
    let mut cur_started_after_heading = false;
    for tok in body.split_whitespace() {
        if !nt_heading_seen {
            if saw_new && tok_eq_ci(tok, "testament") {
                nt_heading_seen = true;
            }
            saw_new = tok_eq_ci(tok, "new");
        }
        if let Some(m) = parse_verse_marker(tok) {
            in_body = true;
            markers.push(m);
            if !cur.is_empty() {
                verses.push(std::mem::take(&mut cur));
                if cur_started_after_heading {
                    crossed = true;
                }
            }
            // The verse THIS marker opens is post-heading iff the heading has
            // already gone by. Set after the flush above, which belongs to the
            // previous verse.
            cur_started_after_heading = nt_heading_seen;
        } else if in_body {
            if tok == "***" {
                continue;
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(tok);
        }
    }
    if !cur.is_empty() {
        verses.push(cur);
        if cur_started_after_heading {
            crossed = true;
        }
    }
    // A marker whose verse never accumulated any body text (two markers in a
    // row, or a marker as the final token) would leave `markers` longer than
    // `verses` and silently de-align every later index. Truncate rather than
    // ship a mis-aligned pair; the alignment is the field's whole contract.
    markers.truncate(verses.len());
    CorpusSplit {
        verses,
        markers,
        crossed_new_testament: crossed,
    }
}

/// Did the parse of `text` recorded in `split` actually cross into the New
/// Testament?
///
/// `None` when `text` announces no New Testament (nothing to check). Otherwise
/// `Some(crossed)`, read from the **parse** via
/// [`CorpusSplit::crossed_new_testament`].
///
/// # Why this does not compare against [`KJV_OLD_TESTAMENT_VERSES`]
///
/// It used to, and the doc used to claim the check "asserts nothing about a
/// specific corpus total". That claim was false — the comparison **was** the
/// corpus total, and it broke on legitimate input in **both** directions:
///
/// | input | old behaviour | why it was wrong |
/// |---|---|---|
/// | a New-Testament-**only** corpus | `Some(false)` → KILL | an NT-only corpus has *fewer* verses than the OT, so it can never exceed the threshold — a valid parse read as a truncation |
/// | an uppercase `THE NEW TESTAMENT` heading | `None` → gate silently off | the announcement search was case-sensitive |
///
/// Reading the boundary from the token walk fixes both and keeps the property
/// that actually mattered: it still fails on the historical truncating parser,
/// which stopped at the lone `***` **before** the heading and therefore emitted
/// no verse after it. [`KJV_OLD_TESTAMENT_VERSES`] survives only as the
/// documented count of the historical bug, not as a threshold.
#[must_use]
pub fn crossed_into_new_testament(text: &str, split: &CorpusSplit) -> Option<bool> {
    if announces_new_testament(text) {
        Some(split.crossed_new_testament)
    } else {
        None
    }
}

/// Does `text` announce a New Testament anywhere, case-insensitively?
///
/// Deliberately case-insensitive: real Gutenberg texts render the heading as
/// `The New Testament`, `THE NEW TESTAMENT`, and other casings, and a
/// case-sensitive miss disables the gate silently rather than loudly.
#[must_use]
pub fn announces_new_testament(text: &str) -> bool {
    let mut saw_new = false;
    for tok in text.split_whitespace() {
        if saw_new && tok_eq_ci(tok, "testament") {
            return true;
        }
        saw_new = tok_eq_ci(tok, "new");
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_stars_do_not_truncate_the_corpus() {
        // The real file's FIRST `***` is at character 0. A parser that breaks
        // on it unconditionally returns an empty corpus.
        let src = "*** START OF THE PROJECT GUTENBERG EBOOK 10 *** \
                   1:1 in the beginning 1:2 and the earth";
        let v = split_verses(src);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], "in the beginning");
    }

    #[test]
    fn bare_separator_does_not_truncate_and_does_not_enter_text() {
        // THE historical bug: a lone `***` between the testaments ended the
        // parse. Both halves must survive, and the fence must not become text.
        let src = "1:1 old testament verse *** 1:1 new testament verse";
        let v = split_verses(src);
        assert_eq!(v.len(), 2, "separator must not truncate");
        assert_eq!(v[0], "old testament verse");
        assert_eq!(v[1], "new testament verse");
        assert!(!v.iter().any(|t| t.contains('*')));
    }

    #[test]
    fn footer_truncates_and_trailing_junk_is_dropped() {
        let src = "1:1 kept 1:2 also kept \
                   *** END OF THE PROJECT GUTENBERG EBOOK 10 *** 1:3 dropped";
        let v = split_verses(src);
        assert_eq!(v.len(), 2);
        assert_eq!(v[1], "also kept");
    }

    #[test]
    fn only_exactly_three_stars_is_skipped() {
        // `tok.bytes().all(|c| c == b'*')` would delete `*` and `**` too,
        // silently corrupting verse text. Only the exact fence is a separator.
        let src = "1:1 a * b ** c **** d";
        let v = split_verses(src);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], "a * b ** c **** d");
    }

    #[test]
    fn markers_are_positionally_aligned_with_verses() {
        let s = split_verses_detailed("3:16 alpha 4:1 beta 4:2 gamma");
        assert_eq!(s.verses, vec!["alpha", "beta", "gamma"]);
        assert_eq!(s.markers, vec![(3, 16), (4, 1), (4, 2)]);
    }

    #[test]
    fn markers_do_not_admit_footer_or_header_tokens() {
        // The falsifier for the second-walk exposure: a `d+:d+` token AFTER
        // the footer. A caller re-scanning the raw text with its own
        // `split_whitespace().filter(is_verse_marker)` would collect 9:9 as a
        // third marker and de-align every downstream index from there on. The
        // splitter's own walk trims at the footer, so it cannot.
        let src = "1:1 a 1:2 b \
                   *** END OF THE PROJECT GUTENBERG EBOOK 10 *** \
                   see section 9:9 of the license";
        let s = split_verses_detailed(src);
        assert_eq!(s.verses.len(), 2, "footer text must not become a verse");
        assert_eq!(s.markers, vec![(1, 1), (1, 2)]);

        // Anti-vacuity: the naive second walk really does over-collect, so the
        // assertion above is discriminating and not tautological.
        let naive = src
            .split_whitespace()
            .filter(|t| is_verse_marker(t))
            .count();
        assert_eq!(naive, 3, "the naive walk over-collects — that IS the bug");
    }

    #[test]
    fn a_trailing_marker_with_no_text_does_not_misalign() {
        // A marker as the final token opens a verse that never accumulates,
        // so it is dropped from `verses`. `markers` must drop it too.
        let s = split_verses_detailed("1:1 alpha 1:2");
        assert_eq!(s.verses.len(), 1);
        assert_eq!(s.markers, vec![(1, 1)]);
    }

    #[test]
    fn markers_cross_the_testament_separator() {
        // The bare `***` fence between the testaments is not a marker and not
        // a terminator: the walk carries straight through it. 4:6 -> 1:1 is
        // the real file's boundary (Malachi 4:6 -> Matthew 1:1), and 4:6 is
        // exactly where the historical `contains("***") => break` stopped.
        let s = split_verses_detailed("4:6 malachi last *** The New Testament 1:1 matthew first");
        assert_eq!(s.markers, vec![(4, 6), (1, 1)]);
        assert!(s.crossed_new_testament);
    }

    #[test]
    fn a_marker_is_exactly_what_the_parse_accepts() {
        assert_eq!(parse_verse_marker("22:21"), Some((22, 21)));
        // `u16::from_str` would take the `+`; the digit guard refuses it.
        assert_eq!(parse_verse_marker("+1:1"), None);
        assert!(!is_verse_marker("+1:1"));
        // Out of u16 range — not a chapter number any book has.
        assert_eq!(parse_verse_marker("99999:1"), None);
        assert!(!is_verse_marker("99999:1"));
    }

    #[test]
    fn marker_detection_rejects_non_numeric_colons() {
        assert!(is_verse_marker("1:1"));
        assert!(is_verse_marker("22:21"));
        assert!(!is_verse_marker("a:1"));
        assert!(!is_verse_marker("1:"));
        assert!(!is_verse_marker(":1"));
        assert!(!is_verse_marker("word"));
    }

    #[test]
    fn crossed_into_new_testament_is_the_falsifier_and_can_fail() {
        // A corpus that announces the NT and has verses on BOTH sides of the
        // heading: the gate must pass.
        let whole = "1:1 old verse The New Testament 1:1 new verse";
        let s = split_verses_detailed(whole);
        assert_eq!(s.verses.len(), 2);
        assert_eq!(crossed_into_new_testament(whole, &s), Some(true));

        // THE CAN-FIRE HALF: the historical truncating parser stopped at the
        // lone `***` BEFORE the heading, so it emitted no verse after it.
        // Simulated by a split whose walk never saw the heading.
        let truncated = CorpusSplit {
            verses: vec!["old verse".to_string()],
            markers: vec![(1, 1)],
            crossed_new_testament: false,
        };
        assert_eq!(
            crossed_into_new_testament(whole, &truncated),
            Some(false),
            "a parse that never reached the NT heading must FAIL the gate"
        );

        // No New Testament announced: nothing to assert.
        let none = split_verses_detailed("1:1 genesis only");
        assert_eq!(crossed_into_new_testament("1:1 genesis only", &none), None);
    }

    #[test]
    fn new_testament_only_corpus_passes_the_gate() {
        // REGRESSION: the old count-based gate returned Some(false) here — an
        // NT-only corpus has FEWER verses than the OT, so it could never clear
        // `> KJV_OLD_TESTAMENT_VERSES`. A valid parse read as a truncation.
        let nt_only = "The New Testament 1:1 the book of the generation 1:2 abraham begat isaac";
        let s = split_verses_detailed(nt_only);
        assert_eq!(s.verses.len(), 2);
        assert!(
            s.verses.len() < KJV_OLD_TESTAMENT_VERSES,
            "fixture must sit below the old threshold, or it does not regress the bug"
        );
        assert_eq!(crossed_into_new_testament(nt_only, &s), Some(true));
    }

    #[test]
    fn a_parse_that_stops_at_the_heading_fails_the_gate() {
        // THE CAN-FIRE HALF, from a REAL parse rather than a hand-built
        // `CorpusSplit`. This is the shape the historical truncating parser
        // produced: text runs into the heading and stops, so no verse ever
        // STARTS after it.
        //
        // The first version of this fix read `nt_heading_seen` at flush time,
        // which credited the pre-heading verse to the New Testament because it
        // happened to flush (at end-of-input) after the heading had gone by —
        // `Some(true)`, gate silently passed on a truncated parse. Tracking
        // where the verse STARTED is what makes it `Some(false)`.
        let truncated_at_heading = "1:1 old verse The New Testament";
        let s = split_verses_detailed(truncated_at_heading);
        assert_eq!(s.verses.len(), 1, "one verse, entirely pre-heading");
        assert!(
            announces_new_testament(truncated_at_heading),
            "the heading IS present — the gate must be armed, not skipped"
        );
        assert_eq!(
            crossed_into_new_testament(truncated_at_heading, &s),
            Some(false),
            "a parse that emitted no verse AFTER the heading must FAIL the gate"
        );

        // ...and the ONE extra verse after the heading flips it. Same text plus
        // a post-heading marker: the difference is the whole discrimination.
        let one_more = "1:1 old verse The New Testament 1:1 new verse";
        let s2 = split_verses_detailed(one_more);
        assert_eq!(crossed_into_new_testament(one_more, &s2), Some(true));
    }

    #[test]
    fn uppercase_heading_still_arms_the_gate() {
        // REGRESSION: the old announcement search was case-sensitive on
        // "The New Testament", so an uppercase heading returned None and
        // silently DISABLED the gate rather than failing loudly.
        let upper = "1:1 old verse THE NEW TESTAMENT 1:1 new verse";
        let s = split_verses_detailed(upper);
        assert!(announces_new_testament(upper));
        assert_eq!(crossed_into_new_testament(upper, &s), Some(true));

        // ...and the gate must still be able to FAIL on that same casing.
        let truncated = CorpusSplit {
            verses: vec!["old verse".to_string()],
            markers: vec![(1, 1)],
            crossed_new_testament: false,
        };
        assert_eq!(crossed_into_new_testament(upper, &truncated), Some(false));
    }

    #[test]
    fn announcement_detection_does_not_fire_on_unrelated_text() {
        // The can-stay-silent twin: "new" and "testament" must be ADJACENT.
        assert!(!announces_new_testament(
            "a new covenant and an old testament"
        ));
        assert!(!announces_new_testament("nothing relevant here"));
        assert!(announces_new_testament("the new testament"));
        assert!(announces_new_testament("The New Testament,"));
    }
}
