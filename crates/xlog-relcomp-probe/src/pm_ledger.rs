//! D-XLOG-PM-1 (lab): a keyed signed counter held as Codec9 / DR512 `P × M`
//! lanes, summed with `add_pm`, with an exact wide escape and demotion.
//!
//! `add_pm` and its scalar reference are a lab copy of ndarray
//! `examples/add_pm/pm.rs` (ndarray `acaa120`, D-ADD-PM), where the kernel
//! is proven against the `i32` oracle on all 512 × 512 codeword pairs. It is
//! an example there, not a library item, so a crate cannot import it; where
//! it should live as a library is a separate decision.
//!
//! A key's value is in exactly one of two places:
//!
//! - a lane (`|value| <= 255`, or never touched), in 64-lane [`Pm`] blocks;
//! - the wide escape (`i64`), once a sum carried past ±255. Its lane is then
//!   the never-touched code and its bit is set in the block's escape mask.
//!
//! A wide value that returns within ±255 is demoted back to a lane. What the
//! ledger counts, separately, because they are different events:
//!
//! - `carries`: an addition whose exact sum left ±255 (`+255 +255 -255`
//!   carries once, though it ends at 255);
//! - `demotions`: a wide value moved back to a lane;
//! - [`PmLedger::escalated`]: keys wide right now.
use ndarray::simd::U8x64;

/// 64 `P × M` lanes: `P=0, M=0` never touched, `P=0, M>0` is `-M`, `P=1`
/// is `+M`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pm {
    /// Magnitude per lane.
    pub mag: U8x64,
    /// Polarity per lane.
    pub pos: u64,
}

impl Pm {
    /// Every lane never touched.
    pub fn untouched() -> Self {
        Pm {
            mag: U8x64::splat(0),
            pos: 0,
        }
    }
}

/// Lanes that hold a value: `P=1`, or `M != 0`.
#[inline(always)]
pub fn touched(x: Pm) -> u64 {
    x.pos | !x.mag.cmpeq_mask(U8x64::splat(0))
}

/// `a ⊕ b` lane-wise, and the lanes whose exact sum is outside ±255 (their
/// result lane is saturated and must not be read).
#[inline(always)]
pub fn add_pm(a: Pm, b: Pm) -> (Pm, u64) {
    let zero = U8x64::splat(0);
    let (ta, tb) = (touched(a), touched(b));
    let same = !(a.pos ^ b.pos);
    let sum = a.mag.saturating_add(b.mag);
    let room = U8x64::splat(255).saturating_sub(b.mag);
    let carry = same & ta & tb & a.mag.cmpgt_mask(room);
    let hi = a.mag.simd_max(b.mag);
    let lo = a.mag.simd_min(b.mag);
    let diff = hi.saturating_sub(lo);
    let a_larger = a.mag.cmpgt_mask(b.mag);
    let p_opp = (a_larger & a.pos) | (!a_larger & b.pos);
    let mag = U8x64::mask_blend(same, diff, sum);
    let pos = (same & a.pos) | (!same & p_opp);
    let pos = pos | (mag.cmpeq_mask(zero) & (ta | tb));
    (Pm { mag, pos }, carry)
}

/// One lane's value: `None` = never touched.
pub fn decode_lane(m: u8, p: bool) -> Option<i64> {
    match (p, m) {
        (false, 0) => None,
        (false, m) => Some(-i64::from(m)),
        (true, m) => Some(i64::from(m)),
    }
}

/// A touched value in ±255 to its lane; zero is `+0`.
pub fn encode_lane(v: i64) -> (u8, bool) {
    assert!((-255..=255).contains(&v), "out of range: {v}");
    (v.unsigned_abs() as u8, v >= 0)
}

/// A keyed signed counter over `n` keys. See the module doc.
#[derive(Clone, Debug)]
pub struct PmLedger {
    blocks: Vec<Pm>,
    /// Per block: keys whose value is in `wide`.
    escaped: Vec<u64>,
    wide: Vec<i64>,
    /// Additions whose exact sum left ±255.
    pub carries: u64,
    /// Wide values moved back to a lane.
    pub demotions: u64,
}

impl PmLedger {
    /// `n` keys, all never touched.
    pub fn new(n: usize) -> Self {
        let b = n.div_ceil(64);
        PmLedger {
            blocks: vec![Pm::untouched(); b],
            escaped: vec![0; b],
            wide: vec![0; n],
            carries: 0,
            demotions: 0,
        }
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.wide.len()
    }

    /// True when the ledger has no keys.
    pub fn is_empty(&self) -> bool {
        self.wide.is_empty()
    }

    /// Key `k`'s value; `None` = never touched.
    pub fn value(&self, k: usize) -> Option<i64> {
        let (b, i) = (k / 64, k % 64);
        if self.escaped[b] >> i & 1 == 1 {
            return Some(self.wide[k]);
        }
        let blk = self.blocks[b];
        decode_lane(blk.mag.to_array()[i], blk.pos >> i & 1 == 1)
    }

    /// Keys whose value is wide now.
    pub fn escalated(&self) -> usize {
        self.escaped.iter().map(|m| m.count_ones() as usize).sum()
    }

    /// Keys that hold a value (touched, lane or wide).
    pub fn touched_keys(&self) -> usize {
        self.blocks
            .iter()
            .zip(&self.escaped)
            .map(|(b, e)| (touched(*b) | e).count_ones() as usize)
            .sum()
    }

    fn set_lane(&mut self, k: usize, v: Option<i64>) {
        let (b, i) = (k / 64, k % 64);
        let mut m = self.blocks[b].mag.to_array();
        let mut p = self.blocks[b].pos & !(1u64 << i);
        match v {
            None => m[i] = 0,
            Some(v) => {
                let (mag, pos) = encode_lane(v);
                m[i] = mag;
                p |= u64::from(pos) << i;
            }
        }
        self.blocks[b] = Pm {
            mag: U8x64::from_array(m),
            pos: p,
        };
    }

    fn escalate(&mut self, k: usize, exact: i64) {
        self.set_lane(k, None);
        self.escaped[k / 64] |= 1 << (k % 64);
        self.wide[k] = exact;
    }

    /// Key `k` (not wide) takes the exact value `v`: a lane if it fits, else
    /// wide, counted as a carry.
    fn put(&mut self, k: usize, v: i64) {
        if v.abs() > 255 {
            self.carries += 1;
            self.escalate(k, v);
        } else {
            self.set_lane(k, Some(v));
        }
    }

    /// Demote key `k` if it is wide and back within ±255.
    fn settle(&mut self, k: usize) {
        let (b, i) = (k / 64, k % 64);
        if self.escaped[b] >> i & 1 == 1 && self.wide[k].abs() <= 255 {
            self.escaped[b] &= !(1 << i);
            let w = self.wide[k];
            self.wide[k] = 0;
            self.set_lane(k, Some(w));
            self.demotions += 1;
        }
    }

    /// Add `w` to key `k` (one event; `w` is touched even when 0).
    pub fn add(&mut self, k: usize, w: i64) {
        let (b, i) = (k / 64, k % 64);
        if self.escaped[b] >> i & 1 == 1 {
            self.wide[k] += w;
            self.settle(k);
            return;
        }
        let old = self.value(k);
        if w.abs() > 255 {
            self.put(k, old.unwrap_or(0) + w);
            return;
        }
        let (m, p) = encode_lane(w);
        let mut one = [0u8; 64];
        one[i] = m;
        let lane = Pm {
            mag: U8x64::from_array(one),
            pos: u64::from(p) << i,
        };
        let (sum, carry) = add_pm(self.blocks[b], lane);
        if carry >> i & 1 == 1 {
            self.carries += 1;
            self.escalate(k, old.unwrap_or(0) + w);
        } else {
            self.blocks[b] = sum;
        }
    }

    /// Add every value of `delta` into `self`, 64 keys per `add_pm`. Lanes
    /// that carry, and keys wide on either side, go through the wide escape.
    pub fn absorb(&mut self, delta: &PmLedger) {
        assert_eq!(self.len(), delta.len(), "same key space");
        let zero = U8x64::splat(0);
        for b in 0..self.blocks.len() {
            let (se, de) = (self.escaped[b], delta.escaped[b]);
            let d = delta.blocks[b];
            // Lanes the vector sum must not see: self wide (its lane is the
            // never-touched code, not its value).
            let off = se;
            let d_lanes = Pm {
                mag: U8x64::mask_blend(off, d.mag, zero),
                pos: d.pos & !off,
            };
            let before = self.blocks[b];
            let (sum, carry) = add_pm(before, d_lanes);
            self.blocks[b] = sum;
            let base = b * 64;
            // carry lanes: exact sum from the inputs, escalate
            let mut c = carry;
            while c != 0 {
                let i = c.trailing_zeros() as usize;
                c &= c - 1;
                let k = base + i;
                let a = decode_lane(before.mag.to_array()[i], before.pos >> i & 1 == 1);
                let x = decode_lane(d.mag.to_array()[i], d.pos >> i & 1 == 1);
                self.carries += 1;
                self.escalate(k, a.unwrap_or(0) + x.unwrap_or(0));
            }
            // self wide, delta lane: add the lane exactly
            let mut s = se & touched(d);
            while s != 0 {
                let i = s.trailing_zeros() as usize;
                s &= s - 1;
                let k = base + i;
                let x = decode_lane(d.mag.to_array()[i], d.pos >> i & 1 == 1);
                self.wide[k] += x.unwrap_or(0);
                self.settle(k);
            }
            // delta wide: add its exact value
            let mut w = de;
            while w != 0 {
                let i = w.trailing_zeros() as usize;
                w &= w - 1;
                let k = base + i;
                let x = delta.wide[k];
                if self.escaped[b] >> i & 1 == 1 {
                    self.wide[k] += x;
                    self.settle(k);
                } else {
                    let a = self.value(k).unwrap_or(0);
                    self.put(k, a + x);
                }
            }
        }
    }

    /// Every key back to never touched; the counters are kept.
    pub fn clear(&mut self) {
        self.blocks.fill(Pm::untouched());
        self.escaped.fill(0);
        self.wide.fill(0);
    }
}
