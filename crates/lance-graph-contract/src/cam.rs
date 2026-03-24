//! CAM-PQ distance contract.
//!
//! Defines the trait for CAM-PQ operations. ndarray provides the codec;
//! lance-graph provides the UDF + storage; consumers call this trait.

/// CAM fingerprint size in bytes.
pub const CAM_SIZE: usize = 6;

/// Number of subspaces.
pub const NUM_SUBSPACES: usize = 6;

/// Number of centroids per subspace.
pub const NUM_CENTROIDS: usize = 256;

/// Named CAM bytes (stroke positions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CamByte {
    Heel   = 0,
    Branch = 1,
    TwigA  = 2,
    TwigB  = 3,
    Leaf   = 4,
    Gamma  = 5,
}

/// CAM-PQ scan strategy (selected by cost model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CamStrategy {
    /// Full ADC: 6 table lookups per candidate. Good for < 10M.
    FullAdc,
    /// Stroke cascade: HEEL → BRANCH → full. 99% rejection. Good for > 10M.
    Cascade,
    /// IVF + Cascade: coarse partition probe then cascade. Good for > 100M.
    IvfCascade,
}

/// Precomputed distance tables (6 subspaces × 256 centroids = 6KB).
///
/// Fits in L1 cache. Computed once per query, reused for all candidates.
pub trait DistanceTableProvider: Send + Sync {
    /// Precompute distance tables from query vector + codebook.
    fn precompute(&self, query: &[f32], codebook: &[Vec<Vec<f32>>]) -> [[f32; 256]; 6];

    /// ADC distance to one 6-byte CAM fingerprint using precomputed tables.
    fn distance(&self, tables: &[[f32; 256]; 6], cam: &[u8; 6]) -> f32;

    /// Batch distance for N candidates. Returns distances.
    fn distance_batch(&self, tables: &[[f32; 256]; 6], cams: &[[u8; 6]]) -> Vec<f32>;
}

/// CAM-PQ codec contract.
///
/// ndarray implements this with its AVX-512 kernels.
/// lance-graph-planner uses it via the CamPqScanOp physical operator.
pub trait CamCodecContract: Send + Sync {
    /// Encode a batch of vectors to CAM fingerprints.
    fn encode_batch(&self, vectors: &[Vec<f32>], codebook: &[Vec<Vec<f32>>]) -> Vec<[u8; 6]>;

    /// Decode a CAM fingerprint back to approximate vector.
    fn decode(&self, cam: &[u8; 6], codebook: &[Vec<Vec<f32>>]) -> Vec<f32>;

    /// Train codebook from sample vectors (k-means).
    fn train_codebook(&self, vectors: &[Vec<f32>], dim: usize) -> Vec<Vec<Vec<f32>>>;

    /// Select scan strategy based on candidate count.
    fn select_strategy(&self, num_candidates: u64) -> CamStrategy;
}

/// IVF (Inverted File) contract for billion-scale search.
pub trait IvfContract: Send + Sync {
    /// Train coarse centroids from sample vectors.
    fn train(&self, vectors: &[Vec<f32>], num_partitions: usize);

    /// Assign a vector to its nearest partition.
    fn assign(&self, vector: &[f32]) -> u32;

    /// Find top-P closest partitions for a query.
    fn probe(&self, query: &[f32], num_probes: usize) -> Vec<(u32, f32)>;
}
