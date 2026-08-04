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

/// The Old Testament's verse count in the KJV — the truncation point of the
/// historical bug, and the floor [`crossed_into_new_testament`] checks against.
pub const KJV_OLD_TESTAMENT_VERSES: usize = 23_145;

/// Is `tok` a `d+:d+` verse marker (e.g. `1:1`, `22:21`)?
#[must_use]
pub fn is_verse_marker(tok: &str) -> bool {
    tok.split_once(':').is_some_and(|(a, b)| {
        !a.is_empty()
            && !b.is_empty()
            && a.bytes().all(|c| c.is_ascii_digit())
            && b.bytes().all(|c| c.is_ascii_digit())
    })
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
    let body = match text.find(GUTENBERG_FOOTER) {
        Some(i) => &text[..i],
        None => text,
    };

    let mut verses: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_body = false;
    for tok in body.split_whitespace() {
        if is_verse_marker(tok) {
            in_body = true;
            if !cur.is_empty() {
                verses.push(std::mem::take(&mut cur));
            }
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
    }
    verses
}

/// Did a parse of `text` yielding `verse_count` verses actually cross into the
/// New Testament?
///
/// `None` when `text` announces no New Testament (nothing to check). Otherwise
/// `Some(crossed)`. This is the general form of the falsifier — it asserts
/// nothing about a specific corpus total, so it works on any input and still
/// fails on the truncating parser, whose count is exactly
/// [`KJV_OLD_TESTAMENT_VERSES`].
#[must_use]
pub fn crossed_into_new_testament(text: &str, verse_count: usize) -> Option<bool> {
    if text.contains("The New Testament") {
        Some(verse_count > KJV_OLD_TESTAMENT_VERSES)
    } else {
        None
    }
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
        let with_nt = "The New Testament of the King James Bible";
        // The truncating parser's exact count — must read as NOT crossed.
        assert_eq!(
            crossed_into_new_testament(with_nt, KJV_OLD_TESTAMENT_VERSES),
            Some(false),
            "the OT-only count must fail the gate"
        );
        assert_eq!(
            crossed_into_new_testament(with_nt, 31_102),
            Some(true),
            "the whole-book count must pass the gate"
        );
        // No New Testament announced: nothing to assert.
        assert_eq!(crossed_into_new_testament("Genesis only", 10), None);
    }
}
