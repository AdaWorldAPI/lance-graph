//! CAM-PQ Scan: physical operator for CAM-PQ distance computation.
//!
//! Inserts into the SCAN phase of the resonance pipeline when the cost model
//! determines that CAM-PQ compression is beneficial (> 100K candidates).
//!
//! # Decision Boundary
//!
//! ```text
//! < 100K candidates  →  ScanOp::Full (brute force Hamming on raw fingerprints)
//! 100K - 10M         →  CamPqScanOp::FullAdc (6-byte ADC, no cascade)
//! > 10M              →  CamPqScanOp::Cascade (stroke 1→2→3 progressive)
//! > 100M             →  IVF probe → CamPqScanOp::Cascade per partition
//! ```

use super::PhysicalOperator;

/// CAM-PQ scan strategy (distinct from the Hamming ScanStrategy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CamPqStrategy {
    /// Full ADC: compute all 6 subspace distances for every candidate.
    /// Good for 100K–10M candidates.
    FullAdc,
    /// Stroke cascade: HEEL → BRANCH → full.
    /// 99% rejection before full ADC. Good for > 10M candidates.
    Cascade,
    /// IVF + Cascade: coarse partition probe then cascade within each.
    /// Good for > 100M candidates.
    IvfCascade,
}

impl CamPqStrategy {
    /// Whether this strategy ranks **every** candidate by its full 6-byte ADC
    /// distance (`FullAdc`), or discards candidates on heuristic sub-distance
    /// thresholds first (`Cascade` / `IvfCascade`).
    ///
    /// The distinction is a *result-semantics* change, not a performance knob:
    /// the cascade's stroke cuts are absolute thresholds on 1 and 2 of the 6
    /// subspaces, so they are **not** lower bounds on the full distance and the
    /// prune is **not admissible** — a candidate with one bad byte and five
    /// excellent ones can be dropped and provably could have been the true
    /// nearest. Exposed so a consumer can tell which ranking it received;
    /// [`CamPqScanOp::select_strategy`] switches between them purely on corpus
    /// size, which would otherwise change the answer invisibly.
    #[inline]
    #[must_use]
    pub const fn is_exact_over_adc(self) -> bool {
        matches!(self, CamPqStrategy::FullAdc)
    }
}

/// The **enumerable complement** of a scan's returned results: every candidate
/// that was excluded, kept in lanes by *cause*.
///
/// A prune nobody can enumerate is a blind spot; a prune you can enumerate is a
/// budget (`E-PERIPHERAL-DISSENT-GUARDS-THE-STRATIFICATION-1`). The three lanes
/// are deliberately **not** merged into one `rejected` bucket: `at_topk` is a
/// declared budget the caller asked for, while `at_heel` / `at_branch` are
/// heuristic guesses that may be wrong. Conflating a budgeted truncation with a
/// heuristic drop is exactly the defect
/// `E-SPLIT-THE-CARRIER-NOT-THE-CALL-SITES-1` names — one field that cannot say
/// *why* forces every reader to re-derive the cause, or (worse) not to.
///
/// Collected only under [`RejectPolicy::Collect`]; the default path never
/// allocates these.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CascadeRejects {
    /// Rejected at stroke 1, with the HEEL sub-distance that rejected it.
    /// **Heuristic** — one subspace of six.
    pub at_heel: Vec<(usize, f32)>,
    /// Rejected at stroke 2, with the HEEL+BRANCH partial that rejected it.
    /// **Heuristic** — two subspaces of six.
    pub at_branch: Vec<(usize, f32)>,
    /// Survived every stroke, computed its full 6-byte ADC distance, and lost
    /// only to `truncate(top_k)`. **Budgeted, not heuristic** — this exclusion
    /// is exactly what the caller asked for and is ranked on complete
    /// information.
    pub at_topk: Vec<(usize, f32)>,
}

impl CascadeRejects {
    /// Count of candidates dropped on a **heuristic** threshold (strokes 1+2).
    /// Excludes `at_topk`, which is a budget rather than a guess.
    #[must_use]
    pub fn heuristic_len(&self) -> usize {
        self.at_heel.len() + self.at_branch.len()
    }

    /// Count of every excluded candidate across all three lanes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heuristic_len() + self.at_topk.len()
    }

    /// Whether nothing at all was excluded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A deterministic **spread** sample of the heuristic rejects — up to `k`
    /// candidate indices, strided across the whole rejected range rather than
    /// taken from its cheap edge.
    ///
    /// The stride is the load-bearing part. The cheap edge here is "just barely
    /// over the threshold": sampling the `k` nearest-misses would draw exactly
    /// the candidates most likely to be genuinely bad-but-close, and least
    /// likely to reveal a systematic threshold error — re-creating the
    /// blindness one level down. Because the prune cuts on 1 (or 2) of 6
    /// subspaces while the answer is the sum of all 6, the informative rejects
    /// are the ones rejected *hard* at stroke 1: that is where one bad byte can
    /// hide five good ones. Striding reaches them.
    ///
    /// The two heuristic lanes are strided **separately**, each within its own
    /// sort key — a HEEL sub-distance and a HEEL+BRANCH partial are not
    /// comparable numbers, and merging them would silently rank one lane
    /// against the other's scale.
    ///
    /// Endpoint-inclusive: for `take > 1` the sample contains both the nearest
    /// miss and the **extremal** reject. (The `i * (n / take)` form used
    /// elsewhere in the workspace stops short of the far edge; here the far
    /// edge is the point.)
    ///
    /// Deterministic by construction (no RNG), so a dissent is reproducible and
    /// auditable rather than a lucky draw.
    #[must_use]
    pub fn heuristic_sample(&self, k: usize) -> Vec<usize> {
        if k == 0 || self.heuristic_len() == 0 {
            return Vec::new();
        }
        // Split the budget proportionally between the lanes.
        let total = self.heuristic_len();
        let heel_k = (k * self.at_heel.len()).div_ceil(total).min(k);
        let branch_k = k - heel_k;
        let mut out = Self::stride_lane(&self.at_heel, heel_k);
        out.extend(Self::stride_lane(&self.at_branch, branch_k));
        out
    }

    /// Sort one lane ascending by its own rejection score, then take `take`
    /// endpoint-inclusive strided picks.
    fn stride_lane(lane: &[(usize, f32)], take: usize) -> Vec<usize> {
        let n = lane.len();
        if take == 0 || n == 0 {
            return Vec::new();
        }
        let mut sorted: Vec<(usize, f32)> = lane.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let take = take.min(n);
        if take == 1 {
            // A single pick takes the FAR edge, not the near one: with a budget
            // of one probe the hardest reject is the informative one.
            return vec![sorted[n - 1].0];
        }
        (0..take)
            .map(|i| sorted[i * (n - 1) / (take - 1)].0)
            .collect()
    }
}

/// A **signal** that the cascade's hand-tuned thresholds are mis-set for this
/// data: at least one heuristically-rejected candidate would have ranked inside
/// the returned top-k had its full 6-byte ADC distance been computed.
///
/// Same contract as `WaveGrounding::Escalate` / `StyleStrategy::peripheral_dissent`
/// / `OutlierSuggestion`: the periphery may force a deeper look, it never
/// decides. The operator does **not** re-insert the candidate, does not re-rank,
/// and does not re-run itself — that would make the periphery the verdict, which
/// is the same failure one level up. The caller (e.g. `CamSearch`) decides
/// whether to re-run with [`CamPqStrategy::FullAdc`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdDissent {
    /// How many rejects were probed with a full 6-byte ADC.
    pub sampled: usize,
    /// How many of those would have placed inside the returned top-k.
    pub would_have_ranked: usize,
    /// The best (lowest, 0-based) rank any probed reject would have taken in
    /// the returned result list.
    pub worst_miss_rank: usize,
    /// If the deepest miss was rejected at stroke 1: its HEEL sub-distance. Any
    /// `heel_threshold` **strictly greater** than this admits it. `None` if the
    /// deepest miss was rejected at stroke 2 instead.
    pub suggested_heel_threshold: Option<f32>,
    /// If the deepest miss was rejected at stroke 2: its HEEL+BRANCH partial.
    /// Any `branch_threshold` strictly greater than this admits it. Kept in its
    /// own field because a stroke-1 and a stroke-2 score are different
    /// quantities and one number could not say which it was.
    pub suggested_branch_threshold: Option<f32>,
}

/// Whether a scan collects its excluded set.
///
/// Retaining rejects is O(n) memory on a path whose entire purpose is to avoid
/// O(n) work, so this is **opt-in** and the disabled arm is the untouched
/// original code path — not a re-implementation guarded by a flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RejectPolicy {
    /// No complement, no dissent probe, no allocation. Bit-identical to
    /// [`CamPqScanOp::execute`].
    #[default]
    Disabled,
    /// Collect the full [`CascadeRejects`] complement, and probe up to
    /// `dissent_sample` strided heuristic rejects with a full 6-byte ADC.
    /// `dissent_sample: 0` collects the complement without probing.
    Collect {
        /// Number of rejects to probe for [`ThresholdDissent`].
        dissent_sample: usize,
    },
}

/// The observable result of a scan: what was returned, **which strategy
/// produced it**, and (opt-in) what was excluded.
///
/// `strategy` exists because [`CamPqScanOp::select_strategy`] switches on
/// corpus size alone: cross 10M rows and the same query silently stops being
/// exact-over-ADC and starts being threshold-pruned. A consumer that measured
/// recall at 1M rows learns nothing about 11M rows unless it can see the
/// switch.
#[derive(Debug, Clone)]
pub struct ScanOutcome {
    /// The strategy that actually ran.
    pub strategy: CamPqStrategy,
    /// Top-k results, sorted by distance ascending.
    pub results: Vec<(usize, f32)>,
    /// The excluded set, when [`RejectPolicy::Collect`] was requested.
    pub rejects: Option<CascadeRejects>,
    /// Dissent signal, when probing was requested AND a probed reject would
    /// have ranked. `None` means "the periphery agrees" or "not probed".
    pub dissent: Option<ThresholdDissent>,
}

impl ScanOutcome {
    /// Whether `results` is an exact ranking over the ADC distance (no
    /// heuristic candidate was discarded). See
    /// [`CamPqStrategy::is_exact_over_adc`].
    #[inline]
    #[must_use]
    pub fn is_exact_over_adc(&self) -> bool {
        self.strategy.is_exact_over_adc()
    }
}

/// CAM-PQ physical scan operator.
#[derive(Debug)]
pub struct CamPqScanOp {
    /// Strategy selected by cost model.
    pub strategy: CamPqStrategy,
    /// HEEL distance threshold for stroke 1 (only used in Cascade/IvfCascade).
    pub heel_threshold: f32,
    /// HEEL+BRANCH distance threshold for stroke 2.
    pub branch_threshold: f32,
    /// Top-K results.
    pub top_k: usize,
    /// Number of IVF partitions to probe (only used in IvfCascade).
    pub num_probes: usize,
    /// Estimated output cardinality.
    pub estimated_cardinality: f64,
    /// Child operator (BroadcastOp or IVF probe output).
    pub child: Box<dyn PhysicalOperator>,
}

impl CamPqScanOp {
    /// Execute CAM-PQ scan on packed 6-byte fingerprints.
    ///
    /// `cam_data[i]` = 6-byte CAM fingerprint for candidate i.
    /// `distance_tables[subspace][centroid]` = precomputed distance.
    pub fn execute(
        &self,
        distance_tables: &[[f32; 256]; 6],
        cam_data: &[[u8; 6]],
    ) -> Vec<(usize, f32)> {
        match self.strategy {
            CamPqStrategy::FullAdc => self.full_adc(distance_tables, cam_data),
            CamPqStrategy::Cascade => self.cascade(distance_tables, cam_data),
            CamPqStrategy::IvfCascade => self.cascade(distance_tables, cam_data),
        }
    }

    /// Full ADC: 6 table lookups per candidate.
    fn full_adc(&self, dt: &[[f32; 256]; 6], cam_data: &[[u8; 6]]) -> Vec<(usize, f32)> {
        let mut results: Vec<(usize, f32)> = cam_data
            .iter()
            .enumerate()
            .map(|(idx, cam)| {
                let dist = dt[0][cam[0] as usize]
                    + dt[1][cam[1] as usize]
                    + dt[2][cam[2] as usize]
                    + dt[3][cam[3] as usize]
                    + dt[4][cam[4] as usize]
                    + dt[5][cam[5] as usize];
                (idx, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.top_k);
        results
    }

    /// Stroke cascade: progressive rejection.
    fn cascade(&self, dt: &[[f32; 256]; 6], cam_data: &[[u8; 6]]) -> Vec<(usize, f32)> {
        // Stroke 1: HEEL only
        let mut survivors: Vec<usize> = Vec::with_capacity(cam_data.len() / 10);
        for (idx, cam) in cam_data.iter().enumerate() {
            if dt[0][cam[0] as usize] < self.heel_threshold {
                survivors.push(idx);
            }
        }

        // Stroke 2: HEEL + BRANCH
        let mut refined: Vec<usize> = Vec::with_capacity(survivors.len() / 10);
        for &idx in &survivors {
            let cam = &cam_data[idx];
            let partial = dt[0][cam[0] as usize] + dt[1][cam[1] as usize];
            if partial < self.branch_threshold {
                refined.push(idx);
            }
        }

        // Stroke 3: full 6-byte ADC on finalists
        let mut results: Vec<(usize, f32)> = refined
            .iter()
            .map(|&idx| {
                let cam = &cam_data[idx];
                let dist = dt[0][cam[0] as usize]
                    + dt[1][cam[1] as usize]
                    + dt[2][cam[2] as usize]
                    + dt[3][cam[3] as usize]
                    + dt[4][cam[4] as usize]
                    + dt[5][cam[5] as usize];
                (idx, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.top_k);
        results
    }

    /// Execute, and report **what was excluded and why** alongside the results.
    ///
    /// With [`RejectPolicy::Disabled`] this delegates to [`Self::execute`] —
    /// literally the same code, so the disabled path costs nothing and is
    /// bit-identical by construction rather than by promise.
    ///
    /// With [`RejectPolicy::Collect`] the cascade additionally retains its
    /// complement in three cause-separated lanes and may emit a
    /// [`ThresholdDissent`] signal. **Instrumentation never changes the
    /// verdict**: `results` is identical either way (test-pinned on `to_bits()`).
    pub fn execute_observed(
        &self,
        distance_tables: &[[f32; 256]; 6],
        cam_data: &[[u8; 6]],
        policy: RejectPolicy,
    ) -> ScanOutcome {
        let dissent_sample = match policy {
            RejectPolicy::Disabled => {
                return ScanOutcome {
                    strategy: self.strategy,
                    results: self.execute(distance_tables, cam_data),
                    rejects: None,
                    dissent: None,
                };
            }
            RejectPolicy::Collect { dissent_sample } => dissent_sample,
        };

        let (results, rejects) = match self.strategy {
            CamPqStrategy::FullAdc => self.full_adc_instrumented(distance_tables, cam_data),
            CamPqStrategy::Cascade | CamPqStrategy::IvfCascade => {
                self.cascade_instrumented(distance_tables, cam_data)
            }
        };
        let dissent = self.threshold_dissent(
            distance_tables,
            cam_data,
            &results,
            &rejects,
            dissent_sample,
        );
        ScanOutcome {
            strategy: self.strategy,
            results,
            rejects: Some(rejects),
            dissent,
        }
    }

    /// Full ADC with its complement retained. Every exclusion here lands in
    /// `at_topk`: `FullAdc` ranks on complete information, so it has no
    /// heuristic lane at all. (That asymmetry is the point of separating the
    /// lanes — it is visible in the data rather than needing to be known.)
    fn full_adc_instrumented(
        &self,
        dt: &[[f32; 256]; 6],
        cam_data: &[[u8; 6]],
    ) -> (Vec<(usize, f32)>, CascadeRejects) {
        let mut results = self.adc_all(dt, cam_data.iter().enumerate().map(|(i, _)| i), cam_data);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let at_topk = if results.len() > self.top_k {
            results.split_off(self.top_k)
        } else {
            Vec::new()
        };
        (
            results,
            CascadeRejects {
                at_topk,
                ..Default::default()
            },
        )
    }

    /// Stroke cascade with its complement retained, cause by cause.
    fn cascade_instrumented(
        &self,
        dt: &[[f32; 256]; 6],
        cam_data: &[[u8; 6]],
    ) -> (Vec<(usize, f32)>, CascadeRejects) {
        let mut rejects = CascadeRejects::default();

        // Stroke 1: HEEL only.
        let mut survivors: Vec<usize> = Vec::with_capacity(cam_data.len() / 10);
        for (idx, cam) in cam_data.iter().enumerate() {
            let heel = dt[0][cam[0] as usize];
            if heel < self.heel_threshold {
                survivors.push(idx);
            } else {
                rejects.at_heel.push((idx, heel));
            }
        }

        // Stroke 2: HEEL + BRANCH.
        let mut refined: Vec<usize> = Vec::with_capacity(survivors.len() / 10);
        for &idx in &survivors {
            let cam = &cam_data[idx];
            let partial = dt[0][cam[0] as usize] + dt[1][cam[1] as usize];
            if partial < self.branch_threshold {
                refined.push(idx);
            } else {
                rejects.at_branch.push((idx, partial));
            }
        }

        // Stroke 3: full 6-byte ADC on finalists.
        let mut results = self.adc_all(dt, refined.iter().copied(), cam_data);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if results.len() > self.top_k {
            rejects.at_topk = results.split_off(self.top_k);
        }
        (results, rejects)
    }

    /// Full 6-byte ADC for a set of candidate indices, in iteration order.
    fn adc_all(
        &self,
        dt: &[[f32; 256]; 6],
        idxs: impl Iterator<Item = usize>,
        cam_data: &[[u8; 6]],
    ) -> Vec<(usize, f32)> {
        idxs.map(|idx| (idx, Self::adc(dt, &cam_data[idx])))
            .collect()
    }

    /// The full 6-subspace ADC distance for one candidate.
    #[inline]
    fn adc(dt: &[[f32; 256]; 6], cam: &[u8; 6]) -> f32 {
        dt[0][cam[0] as usize]
            + dt[1][cam[1] as usize]
            + dt[2][cam[2] as usize]
            + dt[3][cam[3] as usize]
            + dt[4][cam[4] as usize]
            + dt[5][cam[5] as usize]
    }

    /// Probe a strided sample of the heuristic rejects with the **full** 6-byte
    /// ADC and report whether any of them would have ranked.
    ///
    /// O(k) — the sample is bounded, so this is negligible beside the scan it
    /// audits. Returns `None` when the periphery agrees; that silence is a real
    /// reading, not an absence of instrumentation.
    fn threshold_dissent(
        &self,
        dt: &[[f32; 256]; 6],
        cam_data: &[[u8; 6]],
        results: &[(usize, f32)],
        rejects: &CascadeRejects,
        k: usize,
    ) -> Option<ThresholdDissent> {
        let sample = rejects.heuristic_sample(k);
        if sample.is_empty() {
            return None;
        }
        let mut would_have_ranked = 0usize;
        let mut worst_miss_rank = usize::MAX;
        let mut deepest: Option<usize> = None;
        for idx in &sample {
            let full = Self::adc(dt, &cam_data[*idx]);
            // Rank it would have taken among the returned results.
            let rank = results.iter().filter(|(_, d)| *d <= full).count();
            // It "would have ranked" if it lands inside the returned top_k —
            // either because the list is short of the budget, or because it
            // beats a returned entry.
            if rank < results.len() || results.len() < self.top_k {
                would_have_ranked += 1;
                if rank < worst_miss_rank {
                    worst_miss_rank = rank;
                    deepest = Some(*idx);
                }
            }
        }
        let deepest = deepest?;
        // Which lane rejected the deepest miss decides which threshold to
        // suggest — the two scores are different quantities.
        let suggested_heel_threshold = rejects
            .at_heel
            .iter()
            .find(|(i, _)| *i == deepest)
            .map(|(_, s)| *s);
        let suggested_branch_threshold = rejects
            .at_branch
            .iter()
            .find(|(i, _)| *i == deepest)
            .map(|(_, s)| *s);
        Some(ThresholdDissent {
            sampled: sample.len(),
            would_have_ranked,
            worst_miss_rank,
            suggested_heel_threshold,
            suggested_branch_threshold,
        })
    }

    /// Cost model: select strategy based on candidate count.
    ///
    /// **This changes result semantics, not just cost.** Below 10M candidates
    /// the returned ranking is exact over the ADC distance; at or above it, an
    /// inadmissible threshold prune runs first. Callers that care must carry the
    /// returned strategy forward — see [`ScanOutcome::strategy`] and
    /// [`CamPqStrategy::is_exact_over_adc`].
    pub fn select_strategy(num_candidates: u64) -> CamPqStrategy {
        if num_candidates >= 100_000_000 {
            CamPqStrategy::IvfCascade
        } else if num_candidates >= 10_000_000 {
            CamPqStrategy::Cascade
        } else {
            CamPqStrategy::FullAdc
        }
    }

    /// Estimated cost in microseconds.
    pub fn estimated_cost_us(num_candidates: u64, strategy: CamPqStrategy) -> f64 {
        match strategy {
            CamPqStrategy::FullAdc => {
                // 6 lookups + 5 adds ≈ 2ns per candidate
                num_candidates as f64 * 0.002
            }
            CamPqStrategy::Cascade => {
                // Stroke 1: 1 lookup ≈ 0.5ns, 90% rejection
                // Stroke 2: 2 lookups ≈ 1ns on 10% survivors
                // Stroke 3: 6 lookups ≈ 2ns on 1% survivors
                let s1 = num_candidates as f64 * 0.0005;
                let s2 = num_candidates as f64 * 0.1 * 0.001;
                let s3 = num_candidates as f64 * 0.01 * 0.002;
                s1 + s2 + s3
            }
            CamPqStrategy::IvfCascade => {
                // IVF probe: ~50µs
                // Then cascade on ~1% of total
                50.0 + Self::estimated_cost_us(num_candidates / 100, CamPqStrategy::Cascade)
            }
        }
    }
}

impl PhysicalOperator for CamPqScanOp {
    fn name(&self) -> &str {
        "CamPqScan"
    }

    fn cardinality(&self) -> f64 {
        self.estimated_cardinality
    }

    fn is_pipeline_breaker(&self) -> bool {
        // Cascade is streaming (no materialization needed)
        false
    }

    fn children(&self) -> Vec<&dyn PhysicalOperator> {
        vec![&*self.child]
    }
}

#[cfg(test)]
mod tests {
    use super::super::broadcast::BroadcastOp;
    use super::*;

    fn make_distance_tables() -> [[f32; 256]; 6] {
        let mut dt = [[0.0f32; 256]; 6];
        for (s, subspace) in dt.iter_mut().enumerate() {
            for (c, val) in subspace.iter_mut().enumerate() {
                // Distance increases with centroid index
                *val = c as f32 * (s as f32 + 1.0) * 0.1;
            }
        }
        dt
    }

    fn make_cam_data(n: usize) -> Vec<[u8; 6]> {
        (0..n)
            .map(|i| {
                let v = (i % 256) as u8;
                [
                    v,
                    v.wrapping_add(1),
                    v.wrapping_add(2),
                    v.wrapping_add(3),
                    v.wrapping_add(4),
                    v.wrapping_add(5),
                ]
            })
            .collect()
    }

    fn dummy_child() -> Box<dyn PhysicalOperator> {
        Box::new(BroadcastOp {
            fingerprint: vec![0u64; 4],
            partitions: 1,
            cardinality: 1.0,
        })
    }

    #[test]
    fn test_full_adc() {
        let dt = make_distance_tables();
        let cams = make_cam_data(1000);
        let op = CamPqScanOp {
            strategy: CamPqStrategy::FullAdc,
            heel_threshold: 50.0,
            branch_threshold: 25.0,
            top_k: 10,
            num_probes: 0,
            estimated_cardinality: 10.0,
            child: dummy_child(),
        };

        let results = op.execute(&dt, &cams);
        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }

        // Closest should be cam[0] = [0,1,2,3,4,5] with small distances
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_cascade() {
        let dt = make_distance_tables();
        let cams = make_cam_data(10000);
        let op = CamPqScanOp {
            strategy: CamPqStrategy::Cascade,
            heel_threshold: 5.0, // Only pass centroids with heel index < ~50
            branch_threshold: 10.0,
            top_k: 10,
            num_probes: 0,
            estimated_cardinality: 10.0,
            child: dummy_child(),
        };

        // NOTE: `assert!(results.len() <= 10)` used to stand here. It restates
        // `results.truncate(top_k)` with `top_k: 10` and no input can fail it
        // (`E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1` instance #1). These
        // assertions constrain instead: the budget is FILLED (candidates
        // survive at all), and every survivor obeys the stroke-2 invariant
        // that produced it.
        let results = op.execute(&dt, &cams);
        assert_eq!(results.len(), 10, "the top_k budget must actually fill");

        // Each survivor passed stroke 2, so its HEEL+BRANCH partial is strictly
        // below `branch_threshold` — a property of the filter, not of truncate.
        for (idx, _) in &results {
            let cam = &cams[*idx];
            let partial = dt[0][cam[0] as usize] + dt[1][cam[1] as usize];
            assert!(
                partial < op.branch_threshold,
                "candidate {idx} has partial {partial} >= branch_threshold {}",
                op.branch_threshold
            );
        }

        // Anti-vacuity: the filter must actually filter (kept ≪ total).
        let (_, rejects) = op.cascade_instrumented(&dt, &cams);
        assert!(
            results.len() * 3 < cams.len(),
            "cascade kept {} of {} — the prune is inert",
            results.len(),
            cams.len()
        );
        assert!(rejects.heuristic_len() > cams.len() / 2);
    }

    // ── anti-blindness fixtures ──────────────────────────────────────────
    //
    // `make_cam_data` is monotone in the index (every byte tracks `i % 256`),
    // so the cascade's HEEL cut is perfectly correlated with the full ADC
    // distance and the prune loses *nothing*. That fixture cannot falsify a
    // recall claim. These two can.

    /// Deterministic pseudo-random CAM codes (SplitMix64-derived, no RNG dep):
    /// the six bytes are independent, so a bad HEEL byte carries no information
    /// about the other five — the geometry the cascade actually gets wrong.
    fn make_scattered_cam_data(n: usize) -> Vec<[u8; 6]> {
        let mut out = Vec::with_capacity(n);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..n {
            let mut cam = [0u8; 6];
            for b in cam.iter_mut() {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                *b = ((z ^ (z >> 31)) >> 24) as u8;
            }
            out.push(cam);
        }
        out
    }

    fn scan_op(strategy: CamPqStrategy, heel: f32, branch: f32, top_k: usize) -> CamPqScanOp {
        CamPqScanOp {
            strategy,
            heel_threshold: heel,
            branch_threshold: branch,
            top_k,
            num_probes: 0,
            estimated_cardinality: top_k as f64,
            child: dummy_child(),
        }
    }

    /// THE FALSIFIER the old `test_cascade_rejection_rate` was not. The doc
    /// comment on `CamPqStrategy::Cascade` claims "99% rejection before full
    /// ADC" and nothing measured the *cost* side of that claim. This runs both
    /// strategies over the same data and reports recall@k as a number.
    #[test]
    fn cascade_recall_against_full_adc() {
        let dt = make_distance_tables();
        let cams = make_scattered_cam_data(5_000);
        let top_k = 10;

        let exact = scan_op(CamPqStrategy::FullAdc, 0.0, 0.0, top_k).execute(&dt, &cams);
        assert_eq!(exact.len(), top_k, "exact baseline must fill the budget");

        // Shipped production thresholds (api.rs CamSearch::top_k).
        let casc = scan_op(CamPqStrategy::Cascade, 50.0, 25.0, top_k).execute(&dt, &cams);

        let truth: std::collections::HashSet<usize> = exact.iter().map(|(i, _)| *i).collect();
        let hits = casc.iter().filter(|(i, _)| truth.contains(i)).count();
        let recall = hits as f64 / top_k as f64;

        // Measured rejection rate, so the "99%" prose has a number beside it.
        let (_, rejects) =
            scan_op(CamPqStrategy::Cascade, 50.0, 25.0, top_k).cascade_instrumented(&dt, &cams);
        let rejection_rate = rejects.heuristic_len() as f64 / cams.len() as f64;

        println!(
            "cascade recall@{top_k} = {recall:.2} ({hits}/{top_k}); \
             heuristic rejection rate = {rejection_rate:.4} \
             (at_heel {}, at_branch {}, at_topk {})",
            rejects.at_heel.len(),
            rejects.at_branch.len(),
            rejects.at_topk.len(),
        );

        // Both bounds are measured, and both can fail:
        //  - a regression that prunes harder drops recall below the floor;
        //  - a "fix" that stops pruning pushes the rate below the ceiling and
        //    silently turns the cascade into a slower FullAdc.
        assert!(
            recall >= 0.40,
            "cascade recall@{top_k} regressed to {recall:.2}"
        );
        assert!(
            recall < 1.0,
            "recall is 1.0 — the fixture no longer exercises a lossy prune, \
             so this test has stopped falsifying anything"
        );
        assert!(
            rejection_rate > 0.50,
            "heuristic rejection rate {rejection_rate:.4} — the prune stopped pruning"
        );
    }

    /// The complement must PARTITION the candidate set: kept ∪ the three reject
    /// lanes == every index, pairwise disjoint. A complement that loses rows is
    /// the blind spot it was built to remove.
    #[test]
    fn rejects_partition_the_candidate_set() {
        let dt = make_distance_tables();
        let cams = make_scattered_cam_data(2_000);
        // Thresholds chosen so ALL THREE lanes populate — the partition is
        // only informative if each cause actually occurs. (The shipped
        // `heel_threshold: 50.0` is inert against these tables: the largest
        // possible HEEL sub-distance is 25.5, so stroke 1 rejects nothing and
        // `at_heel` would be trivially empty. That is itself worth knowing.)
        let op = scan_op(CamPqStrategy::Cascade, 12.0, 20.0, 10);
        let out = op.execute_observed(&dt, &cams, RejectPolicy::Collect { dissent_sample: 8 });
        let r = out.rejects.expect("collect requested");

        let mut seen: Vec<usize> = out.results.iter().map(|(i, _)| *i).collect();
        seen.extend(r.at_heel.iter().map(|(i, _)| *i));
        seen.extend(r.at_branch.iter().map(|(i, _)| *i));
        seen.extend(r.at_topk.iter().map(|(i, _)| *i));
        assert_eq!(
            seen.len(),
            cams.len(),
            "kept + rejected must account for every candidate exactly once"
        );
        let uniq: std::collections::HashSet<usize> = seen.iter().copied().collect();
        assert_eq!(uniq.len(), cams.len(), "lanes must be pairwise disjoint");

        // Anti-vacuity: all three lanes must be non-trivially populated on this
        // fixture, or the partition proves nothing about the split.
        assert!(!r.at_heel.is_empty(), "at_heel empty — prune inert");
        assert!(!r.at_branch.is_empty(), "at_branch empty — stroke 2 inert");
        assert!(!r.at_topk.is_empty(), "at_topk empty — budget never bound");
    }

    /// The three causes must stay in separate lanes. A HEEL reject is a guess;
    /// a top-k reject is a budget ranked on complete information. `FullAdc`
    /// makes the distinction observable: it has NO heuristic rejects at all.
    #[test]
    fn full_adc_has_no_heuristic_rejects_only_a_budget() {
        let dt = make_distance_tables();
        let cams = make_scattered_cam_data(500);
        let out = scan_op(CamPqStrategy::FullAdc, 50.0, 25.0, 10).execute_observed(
            &dt,
            &cams,
            RejectPolicy::Collect { dissent_sample: 4 },
        );
        let r = out.rejects.as_ref().expect("collect requested");
        assert_eq!(r.heuristic_len(), 0, "FullAdc must not guess");
        assert_eq!(r.at_topk.len(), cams.len() - 10);
        assert!(out.is_exact_over_adc());
        assert!(
            out.dissent.is_none(),
            "no heuristic rejects ⇒ nothing to dissent about"
        );
    }

    /// The stride must reach a HARD reject, not only near-misses. Asserted by
    /// contrast with the cheap-edge alternative on the same lane — if striding
    /// bought nothing, this fails.
    #[test]
    fn stride_sample_reaches_hard_rejects_not_near_misses() {
        let dt = make_distance_tables();
        let cams = make_scattered_cam_data(3_000);
        let op = scan_op(CamPqStrategy::Cascade, 12.0, 20.0, 10);
        let (_, r) = op.cascade_instrumented(&dt, &cams);

        let mut by_score = r.at_heel.clone();
        by_score.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let min = by_score.first().unwrap().1;
        let max = by_score.last().unwrap().1;
        assert!(
            max > min,
            "fixture must span a real range of heel distances"
        );

        let k = 8;
        let sample = r.heuristic_sample(k);
        let score_of = |idx: usize| {
            r.at_heel
                .iter()
                .chain(r.at_branch.iter())
                .find(|(i, _)| *i == idx)
                .unwrap()
                .1
        };
        let sampled_max = sample.iter().map(|i| score_of(*i)).fold(f32::MIN, f32::max);

        // The far edge is reached — endpoint-inclusive striding, not `n/k`.
        assert!(
            sampled_max >= max,
            "strided sample max {sampled_max} did not reach the extremal reject {max}"
        );

        // And the contrast that makes it non-vacuous: the cheap edge (the k
        // nearest misses) stays in the near periphery.
        let cheap_edge_max = by_score
            .iter()
            .take(k)
            .map(|(_, s)| *s)
            .fold(f32::MIN, f32::max);
        assert!(
            cheap_edge_max < min + (max - min) * 0.5,
            "cheap-edge max {cheap_edge_max} was not confined to the near half \
             [{min}, {max}] — the fixture no longer contrasts the two samplers"
        );
    }

    /// A watchdog that cannot bark is the defect one level up. Construct the
    /// exact failure geometry the prune is blind to — one bad HEEL byte hiding
    /// five perfect ones — and assert the channel reports it.
    #[test]
    fn threshold_dissent_can_actually_fire() {
        let dt = make_distance_tables();
        // Baseline: mediocre-but-admissible candidates (heel byte 10 ⇒ heel
        // distance 1.0 < 5.0), with expensive tail bytes.
        let mut cams: Vec<[u8; 6]> = (0..200).map(|_| [10, 10, 200, 200, 200, 200]).collect();
        // The planted candidate: heel byte 250 ⇒ heel distance 25.0, rejected at
        // stroke 1 — yet its other five subspaces are perfect, so its FULL ADC
        // distance beats every survivor.
        cams.push([250, 0, 0, 0, 0, 0]);
        let planted = cams.len() - 1;

        let op = scan_op(CamPqStrategy::Cascade, 5.0, 10.0, 10);
        let out = op.execute_observed(&dt, &cams, RejectPolicy::Collect { dissent_sample: 4 });

        let r = out.rejects.as_ref().unwrap();
        assert!(
            r.at_heel.iter().any(|(i, _)| *i == planted),
            "planted candidate must be rejected at stroke 1"
        );
        assert!(
            !out.results.iter().any(|(i, _)| *i == planted),
            "planted candidate must be absent from the results"
        );
        assert!(
            CamPqScanOp::adc(&dt, &cams[planted]) < out.results[0].1,
            "planted candidate must actually beat the returned best"
        );

        let d = out.dissent.expect("dissent must fire on a provable miss");
        assert!(d.would_have_ranked >= 1);
        assert_eq!(d.worst_miss_rank, 0, "it beats every returned result");
        assert_eq!(
            d.suggested_heel_threshold,
            Some(25.0),
            "the threshold that would have admitted it"
        );
        assert_eq!(
            d.suggested_branch_threshold, None,
            "lanes must not conflate"
        );

        // The signal never DECIDES: results are byte-identical to the
        // uninstrumented path.
        let plain = op.execute(&dt, &cams);
        assert_eq!(plain.len(), out.results.len());
        for (a, b) in plain.iter().zip(out.results.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1.to_bits(), b.1.to_bits());
        }
    }

    /// The converse: a channel that always fires is as useless as one that
    /// never does. With thresholds loose enough to admit everything the
    /// periphery is empty and the guard stays silent.
    #[test]
    fn no_periphery_no_dissent() {
        let dt = make_distance_tables();
        let cams = make_scattered_cam_data(400);
        let out = scan_op(CamPqStrategy::Cascade, 1.0e9, 1.0e9, 10).execute_observed(
            &dt,
            &cams,
            RejectPolicy::Collect { dissent_sample: 16 },
        );
        let r = out.rejects.as_ref().unwrap();
        assert_eq!(r.heuristic_len(), 0, "nothing should be pruned");
        assert!(out.dissent.is_none(), "no periphery ⇒ no dissent");
    }

    /// Opt-in means opt-in: disabled is bit-identical and collects nothing.
    #[test]
    fn disabled_policy_is_bit_identical_and_allocation_free() {
        let dt = make_distance_tables();
        let cams = make_scattered_cam_data(1_500);
        for strategy in [
            CamPqStrategy::FullAdc,
            CamPqStrategy::Cascade,
            CamPqStrategy::IvfCascade,
        ] {
            let op = scan_op(strategy, 50.0, 25.0, 10);
            let plain = op.execute(&dt, &cams);
            let off = op.execute_observed(&dt, &cams, RejectPolicy::Disabled);
            assert!(off.rejects.is_none() && off.dissent.is_none());
            assert_eq!(off.strategy, strategy);
            assert_eq!(plain.len(), off.results.len());
            for (a, b) in plain.iter().zip(off.results.iter()) {
                assert_eq!(a.0, b.0);
                assert_eq!(a.1.to_bits(), b.1.to_bits());
            }
            // ...and instrumentation ON must not move the verdict either.
            let on = op.execute_observed(&dt, &cams, RejectPolicy::Collect { dissent_sample: 8 });
            assert_eq!(plain.len(), on.results.len());
            for (a, b) in plain.iter().zip(on.results.iter()) {
                assert_eq!(a.0, b.0, "instrumentation changed the ranking");
                assert_eq!(a.1.to_bits(), b.1.to_bits());
            }
        }
    }

    /// The corpus-size switch changes result semantics; it must be observable.
    #[test]
    fn strategy_switch_is_observable_and_changes_semantics() {
        assert!(CamPqScanOp::select_strategy(9_999_999).is_exact_over_adc());
        assert!(!CamPqScanOp::select_strategy(10_000_000).is_exact_over_adc());
        assert!(!CamPqScanOp::select_strategy(100_000_000).is_exact_over_adc());

        // And the two really do disagree on the same data — which is why the
        // caller needs to be told which one ran.
        let dt = make_distance_tables();
        let cams = make_scattered_cam_data(2_000);
        let exact = scan_op(CamPqStrategy::FullAdc, 50.0, 25.0, 10).execute_observed(
            &dt,
            &cams,
            RejectPolicy::Disabled,
        );
        let pruned = scan_op(CamPqStrategy::Cascade, 50.0, 25.0, 10).execute_observed(
            &dt,
            &cams,
            RejectPolicy::Disabled,
        );
        assert!(exact.is_exact_over_adc());
        assert!(!pruned.is_exact_over_adc());
        assert_ne!(
            exact.results, pruned.results,
            "if the two agree here the fixture stopped testing the switch"
        );
    }

    #[test]
    fn test_strategy_selection() {
        assert_eq!(
            CamPqScanOp::select_strategy(1_000_000),
            CamPqStrategy::FullAdc
        );
        assert_eq!(
            CamPqScanOp::select_strategy(10_000_000),
            CamPqStrategy::Cascade
        );
        assert_eq!(
            CamPqScanOp::select_strategy(500_000_000),
            CamPqStrategy::IvfCascade
        );
    }

    #[test]
    fn test_cost_model() {
        // Cascade should be cheaper than FullAdc for large datasets
        let n = 100_000_000;
        let full_cost = CamPqScanOp::estimated_cost_us(n, CamPqStrategy::FullAdc);
        let cascade_cost = CamPqScanOp::estimated_cost_us(n, CamPqStrategy::Cascade);
        assert!(
            cascade_cost < full_cost,
            "cascade {}µs should be < full_adc {}µs for {}M candidates",
            cascade_cost,
            full_cost,
            n / 1_000_000
        );
    }

    #[test]
    fn test_physical_operator_trait() {
        let op = CamPqScanOp {
            strategy: CamPqStrategy::Cascade,
            heel_threshold: 50.0,
            branch_threshold: 25.0,
            top_k: 10,
            num_probes: 3,
            estimated_cardinality: 100.0,
            child: dummy_child(),
        };

        assert_eq!(op.name(), "CamPqScan");
        assert_eq!(op.cardinality(), 100.0);
        assert!(!op.is_pipeline_breaker());
        assert_eq!(op.children().len(), 1);
    }
}
