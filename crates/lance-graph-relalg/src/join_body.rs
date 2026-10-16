//! A join body as a hypergraph, and its α-acyclicity test (GYO reduction).
//!
//! Probe P-3 of the worst-case-optimal-join map
//! (`.claude/board/entries/2026-10-16-coresearch-wcoj-leapfrog.md`): the
//! router that sends an α-acyclic body to the semijoin chain (Yannakakis,
//! exact) and a cyclic one to a leapfrog-style intersection needs this test
//! at compile time, before anything runs.
//!
//! Coordinates only: a variable is a `u16`, an atom is the set of variables
//! it binds. Which relation an atom reads does not matter for acyclicity, so
//! it is not part of the description.

/// The variables one atom of the body binds.
pub type Edge = Vec<u16>;

/// GYO (Graham; Yu–Özsoyoğlu) reduction. Repeat until nothing changes:
/// drop every variable that occurs in exactly one edge, and drop every edge
/// that is empty or contained in another edge. The body is α-acyclic iff
/// this leaves no edge.
#[must_use]
pub fn is_alpha_acyclic(body: &[Edge]) -> bool {
    let mut edges: Vec<Edge> = body
        .iter()
        .map(|e| {
            let mut e = e.clone();
            e.sort_unstable();
            e.dedup();
            e
        })
        .collect();
    loop {
        let before: usize = edges.iter().map(Vec::len).sum::<usize>() + edges.len();
        // Variables in exactly one edge.
        let mut count = std::collections::BTreeMap::<u16, usize>::new();
        for e in &edges {
            for &v in e {
                *count.entry(v).or_default() += 1;
            }
        }
        for e in &mut edges {
            e.retain(|v| count[v] > 1);
        }
        // Edges that are empty or contained in another edge. Of two equal
        // edges, keep the first.
        let mut keep = vec![true; edges.len()];
        for i in 0..edges.len() {
            if edges[i].is_empty() {
                keep[i] = false;
                continue;
            }
            for j in 0..edges.len() {
                if i == j || !keep[j] {
                    continue;
                }
                let contained = edges[i].iter().all(|v| edges[j].binary_search(v).is_ok());
                if contained && (edges[i].len() < edges[j].len() || j < i) {
                    keep[i] = false;
                    break;
                }
            }
        }
        let mut k = keep.into_iter();
        edges.retain(|_| k.next().unwrap_or(false));
        let after: usize = edges.iter().map(Vec::len).sum::<usize>() + edges.len();
        if edges.is_empty() {
            return true;
        }
        if after == before {
            return false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u16 = 0;
    const B: u16 = 1;
    const C: u16 = 2;
    const D: u16 = 3;
    const E: u16 = 4;
    const F: u16 = 5;

    fn body(edges: &[&[u16]]) -> Vec<Edge> {
        edges.iter().map(|e| e.to_vec()).collect()
    }

    /// Acyclic bodies the router must send to the semijoin chain.
    #[test]
    fn acyclic_bodies_are_recognised() {
        for (name, b) in [
            ("one atom", body(&[&[A, B]])),
            ("chain", body(&[&[A, B], &[B, C], &[C, D]])),
            ("star", body(&[&[A, B], &[A, C], &[A, D]])),
            ("tree", body(&[&[A, B], &[B, C], &[B, D], &[D, E]])),
            // The 3-ary atom covers the triangle: α-acyclic, not cyclic.
            (
                "covered triangle",
                body(&[&[A, B, C], &[A, B], &[B, C], &[A, C]]),
            ),
            ("repeated atom", body(&[&[A, B], &[A, B], &[B, C]])),
            ("cross product", body(&[&[A], &[B]])),
        ] {
            assert!(is_alpha_acyclic(&b), "{name} misread as cyclic");
        }
    }

    /// Cyclic bodies the router must send to the leapfrog arm. Disable run:
    /// make the test return `true` and every case here goes red.
    #[test]
    fn cyclic_bodies_are_recognised() {
        for (name, b) in [
            ("triangle", body(&[&[A, B], &[B, C], &[A, C]])),
            ("4-cycle", body(&[&[A, B], &[B, C], &[C, D], &[D, A]])),
            (
                "barbell",
                body(&[
                    &[A, B],
                    &[B, C],
                    &[A, C],
                    &[C, D],
                    &[D, E],
                    &[E, F],
                    &[D, F],
                ]),
            ),
            (
                "triangle with a tail",
                body(&[&[A, B], &[B, C], &[A, C], &[C, D]]),
            ),
        ] {
            assert!(!is_alpha_acyclic(&b), "{name} misread as acyclic");
        }
    }
}
