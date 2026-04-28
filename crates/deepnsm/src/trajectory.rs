//! META-AGENT: add `pub mod trajectory;` to lib.rs.

#[derive(Debug, Clone)]
pub struct Trajectory {
    pub fingerprint: Vec<f32>,
    pub radius: u32,
}

impl Trajectory {
    pub fn role_bundle(&self, start: usize, stop: usize) -> Vec<f32> {
        let stop = stop.min(self.fingerprint.len());
        if start >= stop {
            return Vec::new();
        }
        self.fingerprint[start..stop].to_vec()
    }

    pub fn role_candidates(
        &self,
        start: usize,
        stop: usize,
        codebook: &[Vec<f32>],
    ) -> Vec<Candidate> {
        let bundle = self.role_bundle(start, stop);
        let mut scored: Vec<Candidate> = codebook
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let score = cosine(&bundle, entry);
                Candidate {
                    codebook_index: i,
                    score,
                }
            })
            .filter(|c| c.score > 0.5)
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(5);
        scored
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let dot: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    let na: f32 = a[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub codebook_index: usize,
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn role_bundle_returns_slice() {
        let t = Trajectory {
            fingerprint: vec![1.0; 100],
            radius: 5,
        };
        assert_eq!(t.role_bundle(10, 30).len(), 20);
    }
    #[test]
    fn role_bundle_empty_when_inverted() {
        let t = Trajectory {
            fingerprint: vec![1.0; 100],
            radius: 5,
        };
        assert_eq!(t.role_bundle(50, 30).len(), 0);
    }
    #[test]
    fn role_candidates_filters_by_threshold() {
        let t = Trajectory {
            fingerprint: vec![1.0; 100],
            radius: 5,
        };
        let codebook: Vec<Vec<f32>> = vec![vec![1.0; 100], vec![-1.0; 100]];
        let cands = t.role_candidates(0, 100, &codebook);
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].codebook_index, 0);
    }
}
