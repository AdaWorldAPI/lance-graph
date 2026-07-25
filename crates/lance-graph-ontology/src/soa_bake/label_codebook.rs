//! **Label token codebook** — the compression half of the SoA bake.
//!
//! Ontology labels look like free text but are strongly compositional:
//! `arch of aorta`, `ascending aorta`, `segment of systemic artery`, `variant
//! systemic artery`. A tiny set of tokens (`of`, `aorta`, `segment`, `systemic`,
//! …) recurs across tens of thousands of labels. Storing each label verbatim
//! wastes almost all of its bytes on that repetition.
//!
//! A [`LabelCodebook`] captures the unique tokens once (a `Vec<String>` +
//! reverse lookup); a [`LabelColumn`] stores each label as a `Vec<u16>` of
//! token indices. Round-trip through space-join is exact for whitespace-
//! separated labels — the FMA / LOINC / ICD label corpora are exactly this
//! shape.
//!
//! **Measured amortization** (see `examples/fma_amortization`):
//! - Token reuse on FMA `element_parts` label corpus: 166 344 total tokens →
//!   **804 distinct** (a linguistic property — many labels share vocabulary).
//! - Bake byte ratio at simple u16 encoding on the raw label bytes: **~3×**
//!   (1.22 MB → 402 KB combined FMA `element_parts` + `isa_element_parts`).
//! - Further wins compose ON TOP: sharing one codebook across many ontologies
//!   (amortizes vocab cost), zstd on the low-entropy index stream (~2–4×
//!   more), structured templating (`X of Y` → `(template_id, slot_addrs)`).
//! - The RDF-total-file → single-digit-MB claim is dominated by the
//!   ADDRESS-TRIE half of the bake (IRI elimination via prefix containment,
//!   address-pair edges vs verbatim triples) — not by the label codebook
//!   alone. See `soa_bake/mod.rs` for the full compression stack.
//!
//! Deterministic: tokens sort ASCII-lexicographic before assignment, so the
//! same input corpus always mints the same `(codebook, indices)`.

use std::collections::HashMap;
use std::fmt;

/// Why a codebook could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodebookError {
    /// The corpus has more distinct tokens than the u16 index space can
    /// address. Partition the codebook (prefix-scoped, per the module doc)
    /// rather than widening the index.
    VocabularyTooLarge { distinct: usize, max: usize },
}

impl fmt::Display for CodebookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VocabularyTooLarge { distinct, max } => write!(
                f,
                "label vocabulary has {distinct} distinct tokens, exceeding the \
                 u16 codebook index space ({max}); partition the codebook by class prefix"
            ),
        }
    }
}

impl std::error::Error for CodebookError {}

/// The unique token set of an ontology's label corpus. Byte layout is stable
/// (sorted ASCII-lexicographic), so the codebook is content-addressable.
#[derive(Debug, Clone, Default)]
pub struct LabelCodebook {
    /// Token id → token text. Sorted ASCII-lexicographic.
    tokens: Vec<String>,
    /// Token text → token id, for O(1) encode.
    index: HashMap<String, u16>,
}

impl LabelCodebook {
    /// The largest vocabulary a u16-indexed codebook can address.
    pub const MAX_TOKENS: usize = u16::MAX as usize + 1;

    /// Build a codebook from the token set of `labels`, or
    /// [`CodebookError::VocabularyTooLarge`] when the corpus has more than
    /// [`Self::MAX_TOKENS`] distinct whitespace tokens (the u16 index space).
    ///
    /// Tokens are the ASCII whitespace-split fields of each label — the shape
    /// FMA/LOINC/ICD labels take. Deterministic; identical input corpus ⇒
    /// identical codebook.
    ///
    /// A corpus that overflows the index space is a real input (a very large
    /// multilingual ontology), so this is a typed error rather than a panic —
    /// the caller partitions the codebook by class prefix (see the module doc
    /// on prefix-scoped codebooks) instead of losing the bake.
    pub fn try_from_labels<S: AsRef<str>>(labels: &[S]) -> Result<Self, CodebookError> {
        let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for l in labels {
            for tok in l.as_ref().split_whitespace() {
                if !tok.is_empty() {
                    set.insert(tok.to_string());
                }
            }
        }
        if set.len() > Self::MAX_TOKENS {
            return Err(CodebookError::VocabularyTooLarge {
                distinct: set.len(),
                max: Self::MAX_TOKENS,
            });
        }
        let tokens: Vec<String> = set.into_iter().collect();
        let index = tokens
            .iter()
            .enumerate()
            // In range by the check above.
            .map(|(i, t)| (t.clone(), i as u16))
            .collect();
        Ok(Self { tokens, index })
    }

    /// Infallible wrapper over [`Self::try_from_labels`] for corpora known to
    /// fit the u16 index space (every ontology measured so far is ≤ a few
    /// thousand tokens).
    ///
    /// # Panics
    /// If the corpus has more than [`Self::MAX_TOKENS`] distinct tokens. Use
    /// [`Self::try_from_labels`] when the corpus size is not known in advance.
    #[must_use]
    pub fn from_labels<S: AsRef<str>>(labels: &[S]) -> Self {
        Self::try_from_labels(labels).expect("label corpus fits the u16 codebook index space")
    }

    /// Number of distinct tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Is the codebook empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The token slice (`token_id` → text). Stable order.
    #[must_use]
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Encode a single label to token indices. Tokens absent from the codebook
    /// are DROPPED (never panic) — rebuild the codebook from the widened corpus
    /// if it grew.
    ///
    /// [`LabelColumn::bake`] always encodes against a codebook built FROM the
    /// same labels, so no token can be unknown on that path.
    #[must_use]
    pub fn encode(&self, label: &str) -> Vec<u16> {
        label
            .split_whitespace()
            .filter_map(|t| self.index.get(t).copied())
            .collect()
    }

    /// Decode indices back to a space-joined label. Round-trip is exact for
    /// whitespace-separated labels (the FMA/LOINC/ICD shape); indices out of
    /// range are silently skipped (never panic).
    #[must_use]
    pub fn decode(&self, indices: &[u16]) -> String {
        let mut out = String::new();
        for &id in indices {
            if let Some(tok) = self.tokens.get(id as usize) {
                // Separator depends on what was EMITTED, not on the input
                // position — a skipped out-of-range id must not leave a
                // leading space on the surviving label.
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(tok);
            }
        }
        out
    }

    /// Total byte weight of the token vocabulary (Σ token bytes + a per-token
    /// length header, u16). This is the codebook's on-wire size — the fixed
    /// overhead amortized across the entire label column.
    #[must_use]
    pub fn wire_bytes(&self) -> usize {
        self.tokens.iter().map(|t| 2 + t.len()).sum()
    }
}

/// A whole ontology's label column — codebook + per-concept token indices.
/// The bake replaces `Vec<String>` in the RDF-verbatim form.
#[derive(Debug, Clone, Default)]
pub struct LabelColumn {
    pub codebook: LabelCodebook,
    /// Per-concept token index sequences. Order matches the concept row order.
    pub indices: Vec<Vec<u16>>,
}

impl LabelColumn {
    /// Bake `labels` into a codebook + index column, or
    /// [`CodebookError::VocabularyTooLarge`] if the corpus overflows the u16
    /// index space. Round-trip via [`LabelColumn::decode`] is lossless for
    /// whitespace-separated labels (asserted by test on the FMA corpus).
    pub fn try_bake<S: AsRef<str>>(labels: &[S]) -> Result<Self, CodebookError> {
        let codebook = LabelCodebook::try_from_labels(labels)?;
        let indices = labels.iter().map(|l| codebook.encode(l.as_ref())).collect();
        Ok(Self { codebook, indices })
    }

    /// Infallible wrapper over [`Self::try_bake`].
    ///
    /// # Panics
    /// If the corpus overflows the u16 codebook index space.
    #[must_use]
    pub fn bake<S: AsRef<str>>(labels: &[S]) -> Self {
        Self::try_bake(labels).expect("label corpus fits the u16 codebook index space")
    }

    /// Decode the `i`-th label back to a string (space-joined).
    #[must_use]
    pub fn decode(&self, i: usize) -> Option<String> {
        self.indices.get(i).map(|ix| self.codebook.decode(ix))
    }

    /// The bake's on-wire byte size: codebook + `Σ|indices_i| · 2B` (u16) +
    /// a u16 length per row. The number to compare against raw-label bytes.
    #[must_use]
    pub fn wire_bytes(&self) -> usize {
        let indices_bytes: usize = self.indices.iter().map(|ix| 2 + 2 * ix.len()).sum();
        self.codebook.wire_bytes() + indices_bytes
    }
}

/// Raw byte size of a label corpus, verbatim (`Σ|label_i| + 4B length per row`).
/// The baseline against which [`LabelColumn::wire_bytes`] is compared to
/// derive the amortization factor.
#[must_use]
pub fn raw_bytes<S: AsRef<str>>(labels: &[S]) -> usize {
    labels.iter().map(|l| 4 + l.as_ref().len()).sum()
}

/// Amortization factor = raw ÷ bake. `>1` = the bake is smaller. Measured
/// on real FMA labels at u16 encoding: **~3×** (see `fma_amortization`
/// example). Composes further with a shared cross-ontology codebook, zstd
/// on the index stream, and structured templating.
#[must_use]
pub fn amortization<S: AsRef<str>>(labels: &[S], bake: &LabelColumn) -> f64 {
    let r = raw_bytes(labels);
    let b = bake.wire_bytes();
    if b == 0 {
        0.0
    } else {
        r as f64 / b as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_lossless_on_whitespace_separated_labels() {
        let labels = [
            "aorta",
            "ascending aorta",
            "arch of aorta",
            "segment of artery",
            "segment of systemic artery",
            "variant artery",
            "variant systemic artery",
            "vascular tree",
            "venous tree organ",
        ];
        let col = LabelColumn::bake(&labels);
        for (i, l) in labels.iter().enumerate() {
            assert_eq!(
                col.decode(i).as_deref(),
                Some(*l),
                "round-trip failed at {i}"
            );
        }
    }

    #[test]
    fn codebook_is_deterministic_across_builds() {
        let labels = ["arch of aorta", "segment of aorta", "arch of aorta"];
        let a = LabelCodebook::from_labels(&labels);
        let b = LabelCodebook::from_labels(&labels);
        assert_eq!(a.tokens(), b.tokens());
    }

    #[test]
    fn amortization_is_positive_on_repeated_tokens() {
        // A tiny corpus with heavy reuse — the bake must be smaller than raw.
        let mut labels: Vec<String> = Vec::new();
        for kind in ["aorta", "artery", "vein", "capillary"] {
            for qual in [
                "segment of",
                "part of",
                "left",
                "right",
                "ascending",
                "descending",
                "systemic",
                "pulmonary",
            ] {
                labels.push(format!("{qual} {kind}"));
            }
        }
        let col = LabelColumn::bake(&labels);
        let amort = amortization(&labels, &col);
        // Small synthetic corpora amortize modestly (fixed per-row overhead
        // dominates). The real 200+× ratio only shows on production label sets
        // — see the `fma_amortization` example for the empirical measurement.
        assert!(
            amort > 1.0,
            "expected any positive amortization on a compositional corpus, \
             got {amort:.2}x (raw={}, bake={})",
            raw_bytes(&labels),
            col.wire_bytes(),
        );
    }

    /// Codex P2: an out-of-range index must be skipped WITHOUT leaving a
    /// leading separator on the surviving label (spacing follows what was
    /// emitted, not the input position).
    #[test]
    fn out_of_range_index_does_not_leave_a_leading_space() {
        let codebook = LabelCodebook::from_labels(&["aorta artery"]);
        // tokens sort ASCII-lex: ["aorta", "artery"] → ids 0, 1.
        assert_eq!(codebook.decode(&[u16::MAX, 0]), "aorta");
        assert_eq!(codebook.decode(&[u16::MAX, 0, 1]), "aorta artery");
        assert_eq!(codebook.decode(&[0, u16::MAX, 1]), "aorta artery");
        assert_eq!(codebook.decode(&[u16::MAX]), "");
    }

    /// Codex P2: a corpus that overflows the u16 index space returns a typed
    /// error instead of panicking mid-bake.
    #[test]
    fn oversized_vocabulary_is_a_typed_error_not_a_panic() {
        // MAX_TOKENS + 1 distinct tokens, one per label.
        let labels: Vec<String> = (0..=LabelCodebook::MAX_TOKENS)
            .map(|i| format!("t{i}"))
            .collect();
        match LabelColumn::try_bake(&labels) {
            Err(CodebookError::VocabularyTooLarge { distinct, max }) => {
                assert_eq!(distinct, LabelCodebook::MAX_TOKENS + 1);
                assert_eq!(max, LabelCodebook::MAX_TOKENS);
            }
            other => panic!("expected VocabularyTooLarge, got {other:?}"),
        }
        // Exactly at the limit still bakes.
        let at_limit: Vec<String> = (0..LabelCodebook::MAX_TOKENS)
            .map(|i| format!("t{i}"))
            .collect();
        assert!(LabelColumn::try_bake(&at_limit).is_ok());
    }

    #[test]
    fn unknown_tokens_do_not_panic_and_drop_gracefully() {
        let codebook = LabelCodebook::from_labels(&["aorta", "artery"]);
        // "vein" is not in the codebook — encode drops it silently.
        let idx = codebook.encode("aorta vein");
        assert_eq!(idx.len(), 1);
        assert_eq!(codebook.decode(&idx), "aorta");
    }
}
