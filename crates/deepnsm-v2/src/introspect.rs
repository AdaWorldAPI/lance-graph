//! `introspect` — the self-reference falsifier (D-SRS-4): the graph answers
//! questions about its OWN reasoning, and the answers are checked against an
//! INDEPENDENT recomputation done by separate code.
//!
//! Two arms, each an implementation-faithfulness gate (is the self-read
//! CORRECT), not a cognition-strength conjecture:
//!
//! - **Provenance** (`G-SRS4-1`): "which premises concluded triple X?" — the
//!   graph's stored premise pointers ([`crate::reason::DerivationArena`]) must
//!   independently RE-COMPOSE to the conclusion (`(A,p,B)+(B,p,C) ⇒ (A,p,C)`,
//!   shared pivot, same predicate). Strictly stronger than D-SRS-1
//!   resolvability, which never checked that the cited premises actually compose.
//! - **Confidence-delta** (`G-SRS4-2`): "did your confidence in belief Y change
//!   between v1 and v2?" — a NARS confidence read THROUGH the graph's own
//!   version-range window primitive must equal a direct recount over the raw
//!   `(version, triple)` stream.

use crate::reason::DerivationArena;
use crate::spo::Spo;
use crate::TemporalStream;

/// Provenance self-check (`G-SRS4-1`): does every derived triple's stored
/// premise pair independently re-compose to it?
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProvenanceReport {
    /// Derived triples examined (rung ≥ 1).
    pub derived: usize,
    /// Derived triples whose stored premises `[i, j]` satisfy: same predicate,
    /// `entries[i].object == entries[j].subject` (shared pivot),
    /// `X == (entries[i].subject, p, entries[j].object)`.
    pub composes: usize,
}

impl ProvenanceReport {
    /// PASS ⇔ every derived triple's premises re-compose to it.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.composes == self.derived
    }
}

/// Verify the provenance the arena reports about its own reasoning: for each
/// derived entry, re-apply the composition RULE to the cited premises and check
/// it yields exactly that entry's triple. Independent of the resolvability /
/// acyclicity checks (`reason::DerivationArena::gate`).
///
/// A derived entry must cite EXACTLY two premises (the transitive rule composes
/// a pair); anything else fails to compose.
#[must_use]
pub fn provenance_check(arena: &DerivationArena) -> ProvenanceReport {
    let entries = arena.entries();
    let mut derived = 0usize;
    let mut composes = 0usize;
    for d in entries.iter().filter(|d| d.rung >= 1) {
        derived += 1;
        let [i, j] = d.premises[..] else {
            continue; // not a two-premise composition ⇒ does not compose
        };
        let (Some(pi), Some(pj)) = (entries.get(i), entries.get(j)) else {
            continue;
        };
        let (a, b) = (pi.triple, pj.triple);
        // (A, p, B) + (B, p, C) ⇒ (A, p, C): same predicate, shared pivot B,
        // endpoints A and C, and the conclusion is exactly this entry's triple.
        if a.predicate == b.predicate
            && a.predicate == d.triple.predicate
            && a.object == b.subject
            && d.triple.subject == a.subject
            && d.triple.object == b.object
        {
            composes += 1;
        }
    }
    ProvenanceReport { derived, composes }
}

/// A confidence-delta self-answer (`G-SRS4-2`): the graph's NARS confidence in a
/// belief as of two versions, read through its own version-range window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceAnswer {
    /// Occurrences of the belief at `version ≤ v1`.
    pub n1: usize,
    /// Occurrences of the belief at `version ≤ v2`.
    pub n2: usize,
    /// NARS confidence `n1/(n1+k)` as of v1.
    pub c1: f32,
    /// NARS confidence `n2/(n2+k)` as of v2.
    pub c2: f32,
    /// `c2 − c1` (≥ 0 when the belief recurs between the two versions).
    pub delta: f32,
}

/// NARS confidence `n / (n + k)` (evidence horizon `k`). `k = 0, n = 0` ⇒ 0.
#[must_use]
pub fn nars_confidence(n: usize, k: usize) -> f32 {
    let d = n + k;
    if d == 0 {
        0.0
    } else {
        n as f32 / d as f32
    }
}

/// The graph's SELF-ANSWER to "did your confidence in belief `y` change between
/// `v1` and `v2`?" — computed THROUGH the version-range window primitive
/// ([`TemporalStream::window_at`], the `TemporalPov::at` contract). This is the
/// introspective read: it counts `y` in each contemporary window, not by
/// touching any raw list directly.
#[must_use]
pub fn confidence_delta_self(
    stream: &TemporalStream,
    y: Spo,
    v1: u64,
    v2: u64,
    k: usize,
) -> ConfidenceAnswer {
    let count_at = |v: u64| stream.window_at(v).into_iter().filter(|&t| t == y).count();
    let n1 = count_at(v1);
    let n2 = count_at(v2);
    let c1 = nars_confidence(n1, k);
    let c2 = nars_confidence(n2, k);
    ConfidenceAnswer {
        n1,
        n2,
        c1,
        c2,
        delta: c2 - c1,
    }
}

/// The INDEPENDENT ground-truth recount over a raw `(version, triple)` slice —
/// SEPARATE code that never touches the window API. `G-SRS4-2` PASSes iff this
/// equals [`confidence_delta_self`] within tolerance.
#[must_use]
pub fn confidence_delta_recount(
    raw: &[(u64, Spo)],
    y: Spo,
    v1: u64,
    v2: u64,
    k: usize,
) -> ConfidenceAnswer {
    let n1 = raw.iter().filter(|&&(v, t)| v <= v1 && t == y).count();
    let n2 = raw.iter().filter(|&&(v, t)| v <= v2 && t == y).count();
    let c1 = nars_confidence(n1, k);
    let c2 = nars_confidence(n2, k);
    ConfidenceAnswer {
        n1,
        n2,
        c1,
        c2,
        delta: c2 - c1,
    }
}

/// Pick belief Y for `G-SRS4-2`: the single most-frequent triple in `raw`
/// (deterministic — ties broken by smallest `pack()`), with its first and last
/// occurrence versions. `None` if `raw` is empty.
#[must_use]
pub fn most_frequent_belief(raw: &[(u64, Spo)]) -> Option<(Spo, u64, u64)> {
    use std::collections::HashMap;
    let mut counts: HashMap<Spo, (usize, u64, u64)> = HashMap::new();
    for &(v, t) in raw {
        let e = counts.entry(t).or_insert((0, v, v));
        e.0 += 1;
        e.1 = e.1.min(v);
        e.2 = e.2.max(v);
    }
    counts
        .into_iter()
        .max_by(|(ta, a), (tb, b)| {
            // most occurrences; tie → smallest pack() (so choose the SMALLER
            // pack, i.e. reverse the tie comparison since max_by keeps the max).
            a.0.cmp(&b.0).then_with(|| tb.pack().cmp(&ta.pack()))
        })
        .map(|(t, (_, first, last))| (t, first, last))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sound transitive closure has 100%-composing provenance.
    #[test]
    fn provenance_holds_on_sound_closure() {
        let p = 7;
        let base = [Spo::new(1, p, 2), Spo::new(2, p, 3), Spo::new(3, p, 4)];
        let arena = DerivationArena::derive_transitive(&base);
        let r = provenance_check(&arena);
        assert!(r.derived >= 1);
        assert!(r.passed(), "every derived triple must re-compose: {r:?}");
    }

    /// Different-predicate base derives nothing ⇒ provenance passes vacuously.
    #[test]
    fn provenance_vacuous_when_no_derivations() {
        let base = [Spo::new(1, 7, 2), Spo::new(2, 9, 3)];
        let arena = DerivationArena::derive_transitive(&base);
        let r = provenance_check(&arena);
        assert_eq!(r.derived, 0);
        assert!(r.passed());
    }

    /// The windowed self-read equals the independent recount, and confidence
    /// rises as a belief recurs.
    #[test]
    fn confidence_self_matches_recount_and_rises() {
        let y = Spo::new(10, 20, 30);
        let other = Spo::new(1, 2, 3);
        // Y occurs at versions 0, 2, 5; noise elsewhere.
        let raw = vec![
            (0u64, y),
            (1, other),
            (2, y),
            (3, other),
            (4, other),
            (5, y),
        ];
        let mut stream = TemporalStream::new();
        for &(v, t) in &raw {
            stream.push(v, t);
        }
        let (v1, v2) = (2u64, 5u64);
        let self_ans = confidence_delta_self(&stream, y, v1, v2, 1);
        let truth = confidence_delta_recount(&raw, y, v1, v2, 1);
        assert_eq!(self_ans, truth, "self-read must equal independent recount");
        assert_eq!(self_ans.n1, 2); // versions 0, 2
        assert_eq!(self_ans.n2, 3); // versions 0, 2, 5
        assert!(self_ans.delta > 0.0, "confidence must rise as Y recurs");
        // NARS: c1 = 2/3, c2 = 3/4.
        assert!((self_ans.c1 - 2.0 / 3.0).abs() < 1e-6);
        assert!((self_ans.c2 - 3.0 / 4.0).abs() < 1e-6);
    }

    /// `most_frequent_belief` is deterministic: most occurrences, ties by
    /// smallest `pack()`, with correct first/last versions.
    #[test]
    fn most_frequent_belief_is_deterministic() {
        let a = Spo::new(1, 1, 1); // pack smaller
        let b = Spo::new(9, 9, 9); // pack larger, same count → a wins the tie
        let raw = vec![(0u64, a), (1, b), (2, a), (3, b)];
        let (y, first, last) = most_frequent_belief(&raw).unwrap();
        assert_eq!(y, a, "tie broken by smallest pack()");
        assert_eq!((first, last), (0, 2));
        // a clear winner by count.
        let raw2 = vec![(0u64, b), (5, a), (7, a), (9, a)];
        let (y2, f2, l2) = most_frequent_belief(&raw2).unwrap();
        assert_eq!(y2, a);
        assert_eq!((f2, l2), (5, 9));
        assert!(most_frequent_belief(&[]).is_none());
    }

    /// nars_confidence edge cases.
    #[test]
    fn nars_confidence_bounds() {
        assert_eq!(nars_confidence(0, 1), 0.0);
        assert_eq!(nars_confidence(0, 0), 0.0);
        assert!((nars_confidence(1, 1) - 0.5).abs() < 1e-6);
        assert!(nars_confidence(1_000_000, 1) > 0.999);
    }
}
