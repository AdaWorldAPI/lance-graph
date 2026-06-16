//! Weighted grid: buses (nodes) + transmission lines (edges), and the
//! susceptance-weighted graph Laplacian they induce.

/// A transmission line between two buses.
///
/// `susceptance` is the per-unit electrical weight `b_e` (the DC-power-flow
/// `1/x` reactance term). `limit` is the flow magnitude above which the line
/// trips during a cascade.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub susceptance: f64,
    pub limit: f64,
}

impl Edge {
    pub fn new(from: usize, to: usize, susceptance: f64, limit: f64) -> Self {
        Self {
            from,
            to,
            susceptance,
            limit,
        }
    }
}

/// A power network: `n` buses and a list of lines.
#[derive(Debug, Clone)]
pub struct Grid {
    pub n: usize,
    pub edges: Vec<Edge>,
}

impl Grid {
    pub fn new(n: usize, edges: Vec<Edge>) -> Self {
        for e in &edges {
            assert!(e.from < n && e.to < n, "edge endpoint out of range");
            assert!(e.from != e.to, "self-loops not allowed");
        }
        Self { n, edges }
    }

    /// Susceptance-weighted Laplacian of the sub-network whose edges are
    /// `alive` (row-major `n×n`). `L[i][i] = Σ b_e` over incident alive edges;
    /// `L[i][j] = −Σ b_e` over alive edges between `i` and `j`.
    pub fn laplacian_of(&self, alive: &[bool]) -> Vec<f64> {
        assert_eq!(alive.len(), self.edges.len());
        let n = self.n;
        let mut l = vec![0.0; n * n];
        for (idx, e) in self.edges.iter().enumerate() {
            if !alive[idx] {
                continue;
            }
            let b = e.susceptance;
            l[e.from * n + e.from] += b;
            l[e.to * n + e.to] += b;
            l[e.from * n + e.to] -= b;
            l[e.to * n + e.from] -= b;
        }
        l
    }

    /// Laplacian of the full network (all lines in service).
    pub fn laplacian(&self) -> Vec<f64> {
        self.laplacian_of(&vec![true; self.edges.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laplacian_rows_sum_to_zero() {
        let g = Grid::new(
            3,
            vec![Edge::new(0, 1, 1.0, 10.0), Edge::new(1, 2, 2.0, 10.0)],
        );
        let n = 3;
        let l = g.laplacian();
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| l[i * n + j]).sum();
            assert!(row_sum.abs() < 1e-12, "Laplacian row {i} must sum to 0");
        }
        // degree of bus 1 = 1.0 + 2.0
        assert!((l[n + 1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn tripping_edge_removes_its_weight() {
        let g = Grid::new(2, vec![Edge::new(0, 1, 4.0, 10.0)]);
        let dead = g.laplacian_of(&[false]);
        assert!(dead.iter().all(|&x| x == 0.0));
    }
}
