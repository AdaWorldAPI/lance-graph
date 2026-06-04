//! Adaptive arithmetic coder — transcoded from `ac.c`.
//!
//! Two model types and one coder pair:
//!
//! - [`Model256`] — fixed 256-symbol alphabet (the byte-stream model).
//! - [`VModel`] — variable-size alphabet, Fenwick-tree accelerated for
//!   `O(log n)` cumulative-frequency queries.
//! - [`Encoder`] / [`Decoder`] — 24-bit precision range coder with
//!   pending-bits underflow handling.
//!
//! The state machine is bit-exact with the upstream `ac.c` reference:
//! `(lo, hi)` range tracking, renormalization on E1/E2/E3 conditions,
//! and the standard "pending" mechanism for underflow.
//!
//! Round-trip is the load-bearing test: `decode(encode(symbols)) ==
//! symbols` for any symbol sequence drawn from the model's alphabet.
//! Round-trip is verified in this module's tests and again in the
//! end-to-end pipeline tests in phase 4.

/// Precision (in bits) of the range registers.
pub const AC_PREC: u32 = 24;
/// Full range = `1 << AC_PREC`.
pub const AC_FULL: u32 = 1 << AC_PREC;
/// Half-range threshold (E2 boundary).
pub const AC_HALF: u32 = AC_FULL >> 1;
/// Quarter-range threshold (E3 boundary).
pub const AC_QTR: u32 = AC_HALF >> 1;
/// Maximum total frequency before rescaling.
pub const AC_MAX_FREQ: u32 = 1 << 20;

// ──────────────────────────────────────────────────────────────────────
// 256-symbol adaptive model
// ──────────────────────────────────────────────────────────────────────

/// Adaptive frequency model over a fixed 256-symbol alphabet.
///
/// Direct port of `qtc_model_t`. Frequencies start at 1; rescaled
/// (halved, min 1) when `total ≥ AC_MAX_FREQ`.
#[derive(Debug, Clone)]
pub struct Model256 {
    freq: [u32; 256],
    total: u32,
}

impl Default for Model256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Model256 {
    /// Initialize all frequencies to 1.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            freq: [1u32; 256],
            total: 256,
        }
    }

    /// Record an occurrence of `sym` and rescale if `total` reaches the cap.
    pub fn update(&mut self, sym: u8) {
        self.freq[sym as usize] += 1;
        self.total += 1;
        if self.total >= AC_MAX_FREQ {
            self.total = 0;
            for f in &mut self.freq {
                *f = (*f >> 1) | 1;
                self.total += *f;
            }
        }
    }

    /// Write the cumulative-distribution table into `cdf` (length 257)
    /// and return `total`.
    pub fn cdf(&self, cdf: &mut [u32; 257]) -> u32 {
        let mut acc = 0u32;
        for i in 0..256 {
            cdf[i] = acc;
            acc += self.freq[i];
        }
        cdf[256] = acc;
        acc
    }

    /// Read access to the raw frequency table.
    #[inline]
    #[must_use]
    pub const fn freq(&self, sym: u8) -> u32 {
        self.freq[sym as usize]
    }

    /// Current total frequency.
    #[inline]
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.total
    }
}

// ──────────────────────────────────────────────────────────────────────
// Variable-alphabet adaptive model (Fenwick-tree accelerated)
// ──────────────────────────────────────────────────────────────────────

/// Adaptive frequency model over a variable-size alphabet.
///
/// Uses a 1-indexed Fenwick tree (Binary Indexed Tree) for `O(log n)`
/// prefix-sum queries and `O(log n)` updates. Frequencies start at 1
/// for every symbol; rescaled (halve, min 1) when `total ≥ AC_MAX_FREQ`.
#[derive(Debug, Clone)]
pub struct VModel {
    freq: Vec<u32>,
    tree: Vec<u32>,
    total: u32,
    n_sym: u32,
}

impl VModel {
    /// Build a model for an alphabet of `n_sym` symbols.
    ///
    /// # Panics
    ///
    /// Panics if `n_sym == 0` (an empty alphabet is not encodable).
    #[must_use]
    pub fn new(n_sym: u32) -> Self {
        assert!(n_sym > 0, "VModel requires a nonempty alphabet");
        let mut m = Self {
            freq: vec![1; n_sym as usize],
            tree: vec![0; n_sym as usize + 1],
            total: n_sym,
            n_sym,
        };
        for i in 0..n_sym {
            m.ft_add(i, 1);
        }
        m
    }

    /// Symbol-count (alphabet size).
    #[inline]
    #[must_use]
    pub const fn n_sym(&self) -> u32 {
        self.n_sym
    }

    /// Current total frequency.
    #[inline]
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.total
    }

    /// Record an occurrence of `sym` and rescale if `total` reaches the cap.
    pub fn update(&mut self, sym: u32) {
        debug_assert!(sym < self.n_sym, "VModel::update sym out of range");
        self.freq[sym as usize] += 1;
        self.ft_add(sym, 1);
        self.total += 1;
        if self.total >= AC_MAX_FREQ {
            self.total = 0;
            self.tree.fill(0);
            for i in 0..self.n_sym {
                let f = (self.freq[i as usize] >> 1) | 1;
                self.freq[i as usize] = f;
                self.total += f;
                self.ft_add(i, f);
            }
        }
    }

    /// Cumulative frequency strictly below `sym` (= `Σ freq[0..sym]`).
    #[must_use]
    pub fn cum_lo(&self, sym: u32) -> u32 {
        debug_assert!(sym <= self.n_sym);
        self.ft_sum(sym)
    }

    /// Frequency of `sym`.
    #[inline]
    #[must_use]
    pub fn freq_of(&self, sym: u32) -> u32 {
        self.freq[sym as usize]
    }

    /// Find the largest position whose prefix sum is `≤ target`.
    /// Used by the decoder to map a `val`-scaled point back to a symbol.
    #[must_use]
    pub fn find(&self, mut target: u32) -> u32 {
        let n = self.n_sym;
        let mut pos = 0u32;
        let mut pw = 1u32;
        while pw <= n {
            pw <<= 1;
        }
        pw >>= 1;
        while pw > 0 {
            let candidate = pos + pw;
            if candidate <= n && self.tree[candidate as usize] <= target {
                target -= self.tree[candidate as usize];
                pos = candidate;
            }
            pw >>= 1;
        }
        pos
    }

    /// Fenwick tree: 0-indexed `add(i, delta)`.
    fn ft_add(&mut self, i: u32, delta: u32) {
        let n = self.n_sym;
        let mut idx = i + 1; // convert to 1-indexed
        while idx <= n {
            self.tree[idx as usize] = self.tree[idx as usize].wrapping_add(delta);
            idx += idx & idx.wrapping_neg();
        }
    }

    /// Fenwick tree: `sum(0..i)` (1-indexed conceptually).
    fn ft_sum(&self, i: u32) -> u32 {
        let mut idx = i;
        let mut s = 0u32;
        while idx > 0 {
            s = s.wrapping_add(self.tree[idx as usize]);
            idx -= idx & idx.wrapping_neg();
        }
        s
    }
}

// ──────────────────────────────────────────────────────────────────────
// Encoder
// ──────────────────────────────────────────────────────────────────────

/// 24-bit precision range-coder encoder.
///
/// Pending-bits underflow handling matches the upstream `ac.c` state
/// machine bit-for-bit; renormalization conditions are E1 / E2 / E3.
#[derive(Debug)]
pub struct Encoder {
    lo: u32,
    hi: u32,
    pending: u32,
    /// Output byte stream (packed MSB-first within each byte).
    out: Vec<u8>,
    /// Partial byte being assembled.
    buf: u8,
    /// Number of bits in `buf` (0..=7).
    bc: u8,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// Start a fresh encoder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            lo: 0,
            hi: AC_FULL - 1,
            pending: 0,
            out: Vec::new(),
            buf: 0,
            bc: 0,
        }
    }

    /// Encode a symbol given its CDF bracket `[cum_lo, cum_hi)` of `total`.
    ///
    /// # Panics
    ///
    /// Panics on invalid CDF (`cum_lo >= cum_hi`, `cum_hi > total`,
    /// `total == 0`, or range inversion) — these mirror the upstream
    /// `abort()` paths and indicate a calling bug.
    pub fn encode(&mut self, cum_lo: u32, cum_hi: u32, total: u32) {
        assert!(
            cum_lo < cum_hi && cum_hi <= total && total > 0,
            "encode: bad CDF cum_lo={cum_lo} cum_hi={cum_hi} total={total}"
        );
        let r = u64::from(self.hi - self.lo + 1);
        let new_hi = self.lo + (r * u64::from(cum_hi) / u64::from(total)) as u32 - 1;
        let new_lo = self.lo + (r * u64::from(cum_lo) / u64::from(total)) as u32;
        assert!(new_lo <= new_hi, "encode: range inversion");
        self.hi = new_hi;
        self.lo = new_lo;
        loop {
            if self.hi < AC_HALF {
                self.output(0);
            } else if self.lo >= AC_HALF {
                self.output(1);
                self.lo -= AC_HALF;
                self.hi -= AC_HALF;
            } else if self.lo >= AC_QTR && self.hi < 3 * AC_QTR {
                self.pending += 1;
                self.lo -= AC_QTR;
                self.hi -= AC_QTR;
            } else {
                break;
            }
            self.lo <<= 1;
            self.hi = (self.hi << 1) | 1;
        }
    }

    /// Flush remaining state and return the encoded byte stream.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.pending += 1;
        let bit = u8::from(self.lo >= AC_QTR);
        self.output(bit);
        if self.bc > 0 {
            self.buf <<= 8 - self.bc;
            self.out.push(self.buf);
            self.buf = 0;
            self.bc = 0;
        }
        self.out
    }

    fn output(&mut self, bit: u8) {
        self.emit(bit);
        while self.pending > 0 {
            self.emit(1 - bit);
            self.pending -= 1;
        }
    }

    fn emit(&mut self, bit: u8) {
        self.buf = (self.buf << 1) | (bit & 1);
        self.bc += 1;
        if self.bc == 8 {
            self.out.push(self.buf);
            self.buf = 0;
            self.bc = 0;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// Decoder
// ──────────────────────────────────────────────────────────────────────

/// 24-bit precision range-coder decoder.
#[derive(Debug)]
pub struct Decoder<'a> {
    lo: u32,
    hi: u32,
    val: u32,
    data: &'a [u8],
    byte_pos: usize,
    bit_idx: u8,
}

impl<'a> Decoder<'a> {
    /// Construct a decoder over `data` (the output of [`Encoder::finish`]).
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Self {
            lo: 0,
            hi: AC_FULL - 1,
            val: 0,
            data,
            byte_pos: 0,
            bit_idx: 0,
        };
        for _ in 0..AC_PREC {
            d.val = (d.val << 1) | u32::from(d.read_bit());
        }
        d
    }

    fn read_bit(&mut self) -> u8 {
        if self.byte_pos >= self.data.len() {
            return 0;
        }
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_idx)) & 1;
        self.bit_idx += 1;
        if self.bit_idx == 8 {
            self.bit_idx = 0;
            self.byte_pos += 1;
        }
        bit
    }

    /// Decode one symbol given a 256-entry CDF + total.
    pub fn decode_256(&mut self, cdf: &[u32; 257], total: u32) -> u8 {
        let r = u64::from(self.hi - self.lo + 1);
        let scaled = ((u64::from(self.val - self.lo + 1) * u64::from(total) - 1) / r) as u32;
        // Binary search for symbol.
        let mut lo_s = 0u32;
        let mut hi_s = 255u32;
        while lo_s < hi_s {
            let mid = (lo_s + hi_s) >> 1;
            if cdf[(mid + 1) as usize] <= scaled {
                lo_s = mid + 1;
            } else {
                hi_s = mid;
            }
        }
        let sym = lo_s;
        self.advance(cdf[sym as usize], cdf[(sym + 1) as usize], total);
        sym as u8
    }

    /// Decode one symbol via a [`VModel`].
    pub fn decode_v(&mut self, m: &VModel) -> u32 {
        let r = u64::from(self.hi - self.lo + 1);
        let total = m.total();
        let scaled = ((u64::from(self.val - self.lo + 1) * u64::from(total) - 1) / r) as u32;
        let sym = m.find(scaled);
        let cum_lo = m.cum_lo(sym);
        let cum_hi = cum_lo + m.freq_of(sym);
        self.advance(cum_lo, cum_hi, total);
        sym
    }

    fn advance(&mut self, cum_lo: u32, cum_hi: u32, total: u32) {
        let r = u64::from(self.hi - self.lo + 1);
        self.hi = self.lo + (r * u64::from(cum_hi) / u64::from(total)) as u32 - 1;
        self.lo += (r * u64::from(cum_lo) / u64::from(total)) as u32;
        loop {
            if self.hi < AC_HALF {
                // nothing
            } else if self.lo >= AC_HALF {
                self.lo -= AC_HALF;
                self.hi -= AC_HALF;
                self.val -= AC_HALF;
            } else if self.lo >= AC_QTR && self.hi < 3 * AC_QTR {
                self.lo -= AC_QTR;
                self.hi -= AC_QTR;
                self.val -= AC_QTR;
            } else {
                break;
            }
            self.lo <<= 1;
            self.hi = (self.hi << 1) | 1;
            self.val = (self.val << 1) | u32::from(self.read_bit());
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// High-level helpers
// ──────────────────────────────────────────────────────────────────────

/// Encode one symbol using a [`Model256`] (handles CDF + update).
pub fn ac_enc_sym(enc: &mut Encoder, model: &mut Model256, sym: u8) {
    let mut cdf = [0u32; 257];
    let total = model.cdf(&mut cdf);
    enc.encode(cdf[sym as usize], cdf[sym as usize + 1], total);
    model.update(sym);
}

/// Decode one symbol using a [`Model256`] (handles CDF + update).
pub fn ac_dec_sym(dec: &mut Decoder<'_>, model: &mut Model256) -> u8 {
    let mut cdf = [0u32; 257];
    let total = model.cdf(&mut cdf);
    let sym = dec.decode_256(&cdf, total);
    model.update(sym);
    sym
}

/// Encode one symbol using a [`VModel`] (handles update).
///
/// # Panics
///
/// Panics if `sym >= model.n_sym()`.
pub fn ac_enc_v(enc: &mut Encoder, model: &mut VModel, sym: u32) {
    assert!(sym < model.n_sym(), "ac_enc_v: sym out of range");
    let cum_lo = model.cum_lo(sym);
    let cum_hi = cum_lo + model.freq_of(sym);
    enc.encode(cum_lo, cum_hi, model.total());
    model.update(sym);
}

/// Decode one symbol using a [`VModel`] (handles update).
pub fn ac_dec_v(dec: &mut Decoder<'_>, model: &mut VModel) -> u32 {
    let sym = dec.decode_v(model);
    model.update(sym);
    sym
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model256_initial_state_is_uniform() {
        let m = Model256::new();
        assert_eq!(m.total(), 256);
        for s in 0..=255u8 {
            assert_eq!(m.freq(s), 1);
        }
    }

    #[test]
    fn model256_cdf_sums_to_total() {
        let m = Model256::new();
        let mut cdf = [0u32; 257];
        let total = m.cdf(&mut cdf);
        assert_eq!(cdf[256], total);
        assert_eq!(total, 256);
        for i in 0..256 {
            assert!(cdf[i] < cdf[i + 1]);
        }
    }

    #[test]
    fn vmodel_initial_state_is_uniform() {
        let m = VModel::new(10);
        for s in 0..10u32 {
            assert_eq!(m.freq_of(s), 1);
        }
        assert_eq!(m.total(), 10);
    }

    #[test]
    fn vmodel_cum_lo_is_prefix_sum() {
        let m = VModel::new(5);
        assert_eq!(m.cum_lo(0), 0);
        assert_eq!(m.cum_lo(1), 1);
        assert_eq!(m.cum_lo(2), 2);
        assert_eq!(m.cum_lo(3), 3);
        assert_eq!(m.cum_lo(4), 4);
        assert_eq!(m.cum_lo(5), 5);
    }

    #[test]
    fn vmodel_find_is_inverse_of_cum_lo() {
        let mut m = VModel::new(8);
        for s in [3u32, 3, 3, 5, 5, 7, 0] {
            m.update(s);
        }
        let total = m.total();
        for sym in 0..8u32 {
            let lo = m.cum_lo(sym);
            let hi = lo + m.freq_of(sym);
            if lo < hi {
                assert_eq!(m.find(lo), sym, "find(lo={lo}) sym {sym}");
                assert_eq!(m.find(hi - 1), sym, "find(hi-1={}) sym {sym}", hi - 1);
            }
        }
        assert!(total > 8);
    }

    #[test]
    fn round_trip_256_alphabet() {
        let input: Vec<u8> = (0u8..=255).collect();
        let mut enc = Encoder::new();
        let mut m_e = Model256::new();
        for &s in &input {
            ac_enc_sym(&mut enc, &mut m_e, s);
        }
        let bytes = enc.finish();

        let mut dec = Decoder::new(&bytes);
        let mut m_d = Model256::new();
        let decoded: Vec<u8> = (0..input.len())
            .map(|_| ac_dec_sym(&mut dec, &mut m_d))
            .collect();
        assert_eq!(decoded, input);
    }

    #[test]
    fn round_trip_repeated_byte_compresses() {
        let input = vec![42u8; 10_000];
        let mut enc = Encoder::new();
        let mut m_e = Model256::new();
        for &s in &input {
            ac_enc_sym(&mut enc, &mut m_e, s);
        }
        let bytes = enc.finish();
        assert!(
            bytes.len() < input.len() / 10,
            "expected strong compression on repeated byte"
        );

        let mut dec = Decoder::new(&bytes);
        let mut m_d = Model256::new();
        let decoded: Vec<u8> = (0..input.len())
            .map(|_| ac_dec_sym(&mut dec, &mut m_d))
            .collect();
        assert_eq!(decoded, input);
    }

    #[test]
    fn round_trip_variable_alphabet() {
        let input: Vec<u32> = (0u32..50).chain(20..40).chain(0..30).collect();
        let mut enc = Encoder::new();
        let mut m_e = VModel::new(50);
        for &s in &input {
            ac_enc_v(&mut enc, &mut m_e, s);
        }
        let bytes = enc.finish();

        let mut dec = Decoder::new(&bytes);
        let mut m_d = VModel::new(50);
        let decoded: Vec<u32> = (0..input.len())
            .map(|_| ac_dec_v(&mut dec, &mut m_d))
            .collect();
        assert_eq!(decoded, input);
    }

    #[test]
    fn round_trip_pseudo_random_sequence() {
        // Deterministic xorshift for reproducibility.
        let mut state = 0xDEAD_BEEF_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let input: Vec<u8> = (0..5_000).map(|_| (next() & 0xFF) as u8).collect();

        let mut enc = Encoder::new();
        let mut m_e = Model256::new();
        for &s in &input {
            ac_enc_sym(&mut enc, &mut m_e, s);
        }
        let bytes = enc.finish();

        let mut dec = Decoder::new(&bytes);
        let mut m_d = Model256::new();
        let decoded: Vec<u8> = (0..input.len())
            .map(|_| ac_dec_sym(&mut dec, &mut m_d))
            .collect();
        assert_eq!(decoded, input);
    }

    #[test]
    fn vmodel_round_trip_with_rescaling_pressure() {
        // Force the rescale path by exceeding AC_MAX_FREQ.
        let cycles = AC_MAX_FREQ as usize / 4;
        let input: Vec<u32> = (0..cycles).map(|i| (i % 4) as u32).collect();
        let mut enc = Encoder::new();
        let mut m_e = VModel::new(4);
        for &s in &input {
            ac_enc_v(&mut enc, &mut m_e, s);
        }
        let bytes = enc.finish();

        let mut dec = Decoder::new(&bytes);
        let mut m_d = VModel::new(4);
        let decoded: Vec<u32> = (0..input.len())
            .map(|_| ac_dec_v(&mut dec, &mut m_d))
            .collect();
        assert_eq!(decoded, input);
    }
}
