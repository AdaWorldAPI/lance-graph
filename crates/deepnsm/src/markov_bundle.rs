//! META-AGENT: add `pub mod markov_bundle;` to lib.rs.

use crate::trajectory::Trajectory;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Kernel {
    Uniform,
    #[default]
    MexicanHat,
    Gaussian,
}

impl Kernel {
    pub fn weight(&self, delta: i32, radius: u32) -> f32 {
        let d = delta.abs() as f32 / radius.max(1) as f32;
        match self {
            Self::Uniform => 1.0,
            Self::MexicanHat => (1.0 - d * d) * (-(d * d) / 2.0).exp(),
            Self::Gaussian => (-(d * d) / 2.0).exp(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammaticalRole {
    Subject,
    Predicate,
    Object,
    Modifier,
    Context,
    Temporal,
    Kausal,
    Modal,
    Lokal,
    Instrument,
}

impl GrammaticalRole {
    /// Slice of the 16384-dim VSA carrier that owns this role.
    pub fn slice(&self) -> (usize, usize) {
        match self {
            Self::Subject => (0, 3277),
            Self::Predicate => (3277, 6554),
            Self::Object => (6554, 9830),
            Self::Modifier => (9830, 13107),
            Self::Context => (13107, 16384),
            // TEKAMOLO sub-slices inside Context band.
            Self::Temporal => (13107, 13762),
            Self::Kausal => (13762, 14418),
            Self::Modal => (14418, 15074),
            Self::Lokal => (15074, 15729),
            Self::Instrument => (15729, 16384),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenWithRole {
    pub content_fp: Vec<f32>,
    pub role: GrammaticalRole,
}

#[derive(Debug, Clone)]
pub struct WindowedSentence {
    pub tokens: Vec<TokenWithRole>,
}

pub struct MarkovBundler {
    pub radius: u32,
    pub kernel: Kernel,
    pub dims: usize,
    buffer: std::collections::VecDeque<WindowedSentence>,
}

impl MarkovBundler {
    pub fn new(radius: u32, kernel: Kernel) -> Self {
        Self {
            radius,
            kernel,
            dims: 16_384,
            buffer: std::collections::VecDeque::with_capacity((2 * radius + 1) as usize),
        }
    }

    pub fn push(&mut self, sentence: WindowedSentence) -> Option<Trajectory> {
        let cap = (2 * self.radius + 1) as usize;
        if self.buffer.len() == cap {
            self.buffer.pop_front();
        }
        self.buffer.push_back(sentence);
        if self.buffer.len() < cap {
            return None;
        }
        Some(self.bundle_current())
    }

    fn bundle_current(&self) -> Trajectory {
        let mut acc = vec![0.0f32; self.dims];
        let focal = self.radius as i32;
        for (i, sent) in self.buffer.iter().enumerate() {
            let delta = (i as i32) - focal;
            let weight = self.kernel.weight(delta, self.radius);
            for tok in &sent.tokens {
                let (start, stop) = tok.role.slice();
                let len = (stop - start).min(tok.content_fp.len());
                for k in 0..len {
                    acc[start + k] += weight * tok.content_fp[k];
                }
            }
        }
        // permute by position offset (rotate_right)
        if !acc.is_empty() {
            let k = (self.radius as usize) % acc.len();
            acc.rotate_right(k);
        }
        Trajectory {
            fingerprint: acc,
            radius: self.radius,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tok(role: GrammaticalRole, len: usize) -> TokenWithRole {
        TokenWithRole {
            content_fp: vec![1.0; len],
            role,
        }
    }
    #[test]
    fn first_pushes_return_none_until_window_full() {
        let mut b = MarkovBundler::new(5, Kernel::MexicanHat);
        for _ in 0..10 {
            assert!(b
                .push(WindowedSentence {
                    tokens: vec![tok(GrammaticalRole::Subject, 4)]
                })
                .is_none());
        }
        assert!(b
            .push(WindowedSentence {
                tokens: vec![tok(GrammaticalRole::Subject, 4)]
            })
            .is_some());
    }
    #[test]
    fn kernel_uniform_constant() {
        assert_eq!(Kernel::Uniform.weight(0, 5), 1.0);
        assert_eq!(Kernel::Uniform.weight(3, 5), 1.0);
    }
    #[test]
    fn kernel_mexican_symmetric() {
        assert!(
            (Kernel::MexicanHat.weight(-2, 5) - Kernel::MexicanHat.weight(2, 5)).abs() < 1e-6
        );
    }
    #[test]
    fn role_slices_disjoint() {
        let s = GrammaticalRole::Subject.slice();
        let p = GrammaticalRole::Predicate.slice();
        assert_eq!(s.1, p.0);
    }
}
