//! Five physical representations of the same logical rows. Every arm is built
//! from a [`Fixture`] through `fixture::value` / `fixture::present` and answers
//! the same [`Query`] with the same `u128`.
//!
//! Contract for `Arm::run`: no allocation, no row tuples, no per-query `Vec`.
//! Columnar arms iterate lanes contiguously with plain scalar loops.
//!
//! Field 2 values are stored for every row; presence lives in a separate
//! bitmask (bit `i` set = present). Absent is not zero: only `SumPresentC`
//! consults the mask, all other queries read the raw stored value.

use std::cell::Cell;
use std::collections::HashMap;

use crate::fixture::{field_bits, mask_to, present, value, Fixture, FIELDS};

/// The predicate constant: `A == K` selects about 1/16 of rows.
pub const K: u64 = 7;

/// The workload. Every query answers a `u128`.
#[derive(Debug, Clone, Copy)]
pub enum Query<'a> {
    /// COUNT of rows with `A == K`. Demands {A}.
    CountWhere,
    /// Whether any row has `A == K` (early exit allowed). Demands {A}.
    Exists,
    /// SUM(B) over rows with `A == K`, exact in `u128`. Demands {A, B}.
    SumWhere,
    /// SUM(C) over rows with `A == K` and C present. Demands {A, C, presence(C)}.
    SumPresentC,
    /// `acc = acc.wrapping_add(v_f ^ f)` over all rows for fields `0..d`.
    DemandFold { d: usize },
    /// For each probe ordinal, XOR of fields 0..5, folded with wrapping add.
    Point { probes: &'a [usize] },
}

/// The fields a query reads (cold helper; allocates).
pub fn demanded_fields(q: &Query<'_>) -> Vec<usize> {
    match *q {
        Query::CountWhere | Query::Exists => vec![0],
        Query::SumWhere => vec![0, 1],
        Query::SumPresentC => vec![0, 2],
        Query::DemandFold { d } => (0..d.min(FIELDS)).collect(),
        Query::Point { .. } => (0..5).collect(),
    }
}

/// A physical representation.
pub trait Arm {
    fn name(&self) -> &'static str;
    fn bytes_resident(&self) -> usize;
    fn run(&self, q: &Query<'_>) -> u128;
    /// Storage width in bits per value of field `f` for columnar arms; `None`
    /// for row-shaped arms.
    fn storage_bits(&self, _f: usize) -> Option<u32> {
        None
    }
}

fn build_presence(fx: &Fixture) -> Vec<u64> {
    let mut m = vec![0u64; fx.n.div_ceil(64)];
    for i in 0..fx.n {
        if present(fx.seed, i, 2) {
            m[i >> 6] |= 1u64 << (i & 63);
        }
    }
    m
}

#[inline(always)]
fn bit(m: &[u64], i: usize) -> bool {
    (m[i >> 6] >> (i & 63)) & 1 == 1
}

// ---------------------------------------------------------------- Aos

/// Array of structs: one `[u64; 64]` per row.
pub struct Aos {
    rows: Vec<[u64; FIELDS]>,
    pres: Vec<u64>,
}

impl Aos {
    pub fn build(fx: &Fixture) -> Aos {
        let mut rows = Vec::with_capacity(fx.n);
        for i in 0..fx.n {
            let mut r = [0u64; FIELDS];
            for (f, slot) in r.iter_mut().enumerate() {
                *slot = value(fx.seed, i, f);
            }
            rows.push(r);
        }
        Aos {
            rows,
            pres: build_presence(fx),
        }
    }
}

impl Arm for Aos {
    fn name(&self) -> &'static str {
        "aos"
    }

    fn bytes_resident(&self) -> usize {
        self.rows.len() * FIELDS * 8 + self.pres.len() * 8
    }

    fn run(&self, q: &Query<'_>) -> u128 {
        match *q {
            Query::CountWhere => {
                let mut c = 0u64;
                for r in &self.rows {
                    c += (r[0] == K) as u64;
                }
                c as u128
            }
            Query::Exists => self.rows.iter().any(|r| r[0] == K) as u128,
            Query::SumWhere => {
                let mut s = 0u128;
                for r in &self.rows {
                    if r[0] == K {
                        s += r[1] as u128;
                    }
                }
                s
            }
            Query::SumPresentC => {
                let mut s = 0u128;
                for (i, r) in self.rows.iter().enumerate() {
                    if r[0] == K && bit(&self.pres, i) {
                        s += r[2] as u128;
                    }
                }
                s
            }
            Query::DemandFold { d } => {
                let d = d.min(FIELDS);
                let mut acc = 0u64;
                for r in &self.rows {
                    for (f, &v) in r.iter().take(d).enumerate() {
                        acc = acc.wrapping_add(v ^ f as u64);
                    }
                }
                acc as u128
            }
            Query::Point { probes } => {
                let mut acc = 0u64;
                for &p in probes {
                    let r = &self.rows[p];
                    acc = acc.wrapping_add(r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4]);
                }
                acc as u128
            }
        }
    }
}

// ---------------------------------------------------------------- SoaU64

/// Lane-major `u64` SoA: lane `f` occupies `[f*n .. (f+1)*n]` of one slab.
pub struct SoaU64 {
    data: Box<[u64]>,
    n: usize,
    pres: Vec<u64>,
}

impl SoaU64 {
    pub fn build(fx: &Fixture) -> SoaU64 {
        let n = fx.n;
        let mut data = vec![0u64; FIELDS * n];
        for f in 0..FIELDS {
            let lane = &mut data[f * n..(f + 1) * n];
            for (i, slot) in lane.iter_mut().enumerate() {
                *slot = value(fx.seed, i, f);
            }
        }
        SoaU64 {
            data: data.into_boxed_slice(),
            n,
            pres: build_presence(fx),
        }
    }

    /// Row count.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Lane `f`, borrowed in place.
    #[inline(always)]
    pub fn lane(&self, f: usize) -> &[u64] {
        &self.data[f * self.n..(f + 1) * self.n]
    }
}

impl Arm for SoaU64 {
    fn name(&self) -> &'static str {
        "soa-u64"
    }

    fn bytes_resident(&self) -> usize {
        self.data.len() * 8 + self.pres.len() * 8
    }

    fn storage_bits(&self, _f: usize) -> Option<u32> {
        Some(64)
    }

    fn run(&self, q: &Query<'_>) -> u128 {
        match *q {
            Query::CountWhere => {
                let mut c = 0u64;
                for &a in self.lane(0) {
                    c += (a == K) as u64;
                }
                c as u128
            }
            Query::Exists => self.lane(0).iter().any(|&a| a == K) as u128,
            Query::SumWhere => {
                let mut s = 0u128;
                for (&a, &b) in self.lane(0).iter().zip(self.lane(1)) {
                    if a == K {
                        s += b as u128;
                    }
                }
                s
            }
            Query::SumPresentC => {
                let mut s = 0u128;
                for (i, (&a, &c)) in self.lane(0).iter().zip(self.lane(2)).enumerate() {
                    if a == K && bit(&self.pres, i) {
                        s += c as u128;
                    }
                }
                s
            }
            Query::DemandFold { d } => {
                let mut acc = 0u64;
                for f in 0..d.min(FIELDS) {
                    let fx = f as u64;
                    for &v in self.lane(f) {
                        acc = acc.wrapping_add(v ^ fx);
                    }
                }
                acc as u128
            }
            Query::Point { probes } => {
                let (l0, l1, l2, l3, l4) = (
                    self.lane(0),
                    self.lane(1),
                    self.lane(2),
                    self.lane(3),
                    self.lane(4),
                );
                let mut acc = 0u64;
                for &p in probes {
                    acc = acc.wrapping_add(l0[p] ^ l1[p] ^ l2[p] ^ l3[p] ^ l4[p]);
                }
                acc as u128
            }
        }
    }
}

// ---------------------------------------------------------------- SoaNarrow

/// One lane in the narrowest of u8/u16/u32/u64 that holds the field width.
pub enum NarrowLane {
    U8(Box<[u8]>),
    U16(Box<[u16]>),
    U32(Box<[u32]>),
    U64(Box<[u64]>),
}

impl NarrowLane {
    fn build(fx: &Fixture, f: usize) -> NarrowLane {
        let (seed, n) = (fx.seed, fx.n);
        match field_bits(f) {
            0..=8 => NarrowLane::U8((0..n).map(|i| value(seed, i, f) as u8).collect()),
            9..=16 => NarrowLane::U16((0..n).map(|i| value(seed, i, f) as u16).collect()),
            17..=32 => NarrowLane::U32((0..n).map(|i| value(seed, i, f) as u32).collect()),
            _ => NarrowLane::U64((0..n).map(|i| value(seed, i, f)).collect()),
        }
    }

    /// Widened value at row `i`.
    #[inline(always)]
    pub fn get(&self, i: usize) -> u64 {
        match self {
            NarrowLane::U8(s) => s[i] as u64,
            NarrowLane::U16(s) => s[i] as u64,
            NarrowLane::U32(s) => s[i] as u64,
            NarrowLane::U64(s) => s[i],
        }
    }

    /// Container width in bits.
    pub fn width_bits(&self) -> u32 {
        match self {
            NarrowLane::U8(_) => 8,
            NarrowLane::U16(_) => 16,
            NarrowLane::U32(_) => 32,
            NarrowLane::U64(_) => 64,
        }
    }

    fn byte_size(&self) -> usize {
        match self {
            NarrowLane::U8(s) => s.len(),
            NarrowLane::U16(s) => s.len() * 2,
            NarrowLane::U32(s) => s.len() * 4,
            NarrowLane::U64(s) => s.len() * 8,
        }
    }
}

macro_rules! dispatch1 {
    ($a:expr, $f:ident $(, $extra:expr)*) => {
        match $a {
            NarrowLane::U8(x) => $f(x $(, $extra)*),
            NarrowLane::U16(x) => $f(x $(, $extra)*),
            NarrowLane::U32(x) => $f(x $(, $extra)*),
            NarrowLane::U64(x) => $f(x $(, $extra)*),
        }
    };
}

macro_rules! dispatch2 {
    ($a:expr, $b:expr, $f:ident $(, $extra:expr)*) => {
        match ($a, $b) {
            (NarrowLane::U8(x), NarrowLane::U8(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U8(x), NarrowLane::U16(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U8(x), NarrowLane::U32(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U8(x), NarrowLane::U64(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U16(x), NarrowLane::U8(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U16(x), NarrowLane::U16(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U16(x), NarrowLane::U32(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U16(x), NarrowLane::U64(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U32(x), NarrowLane::U8(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U32(x), NarrowLane::U16(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U32(x), NarrowLane::U32(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U32(x), NarrowLane::U64(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U64(x), NarrowLane::U8(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U64(x), NarrowLane::U16(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U64(x), NarrowLane::U32(y)) => $f(x, y $(, $extra)*),
            (NarrowLane::U64(x), NarrowLane::U64(y)) => $f(x, y $(, $extra)*),
        }
    };
}

fn count_eq<T: Copy + Into<u64>>(a: &[T]) -> u64 {
    let mut c = 0u64;
    for &x in a {
        c += (x.into() == K) as u64;
    }
    c
}

fn any_eq<T: Copy + Into<u64>>(a: &[T]) -> bool {
    a.iter().any(|&x| x.into() == K)
}

fn sum_pair<A: Copy + Into<u64>, B: Copy + Into<u64>>(a: &[A], b: &[B]) -> u128 {
    let mut s = 0u128;
    for (&x, &y) in a.iter().zip(b) {
        if x.into() == K {
            s += y.into() as u128;
        }
    }
    s
}

fn sum_pair_present<A: Copy + Into<u64>, C: Copy + Into<u64>>(
    a: &[A],
    c: &[C],
    pres: &[u64],
) -> u128 {
    let mut s = 0u128;
    for (i, (&x, &y)) in a.iter().zip(c).enumerate() {
        if x.into() == K && bit(pres, i) {
            s += y.into() as u128;
        }
    }
    s
}

fn fold_xor<T: Copy + Into<u64>>(s: &[T], fx: u64) -> u64 {
    let mut acc = 0u64;
    for &v in s {
        acc = acc.wrapping_add(v.into() ^ fx);
    }
    acc
}

/// Per-field narrowest-container SoA.
pub struct SoaNarrow {
    lanes: Vec<NarrowLane>,
    pres: Vec<u64>,
}

impl SoaNarrow {
    pub fn build(fx: &Fixture) -> SoaNarrow {
        SoaNarrow {
            lanes: (0..FIELDS).map(|f| NarrowLane::build(fx, f)).collect(),
            pres: build_presence(fx),
        }
    }

    /// The stored lane of field `f`.
    pub fn lane(&self, f: usize) -> &NarrowLane {
        &self.lanes[f]
    }
}

impl Arm for SoaNarrow {
    fn name(&self) -> &'static str {
        "soa-narrow"
    }

    fn bytes_resident(&self) -> usize {
        self.lanes.iter().map(NarrowLane::byte_size).sum::<usize>() + self.pres.len() * 8
    }

    fn storage_bits(&self, f: usize) -> Option<u32> {
        Some(self.lanes[f].width_bits())
    }

    fn run(&self, q: &Query<'_>) -> u128 {
        match *q {
            Query::CountWhere => dispatch1!(&self.lanes[0], count_eq) as u128,
            Query::Exists => dispatch1!(&self.lanes[0], any_eq) as u128,
            Query::SumWhere => dispatch2!(&self.lanes[0], &self.lanes[1], sum_pair),
            Query::SumPresentC => dispatch2!(
                &self.lanes[0],
                &self.lanes[2],
                sum_pair_present,
                &self.pres
            ),
            Query::DemandFold { d } => {
                let mut acc = 0u64;
                for f in 0..d.min(FIELDS) {
                    acc = acc.wrapping_add(dispatch1!(&self.lanes[f], fold_xor, f as u64));
                }
                acc as u128
            }
            Query::Point { probes } => {
                let mut acc = 0u64;
                for &p in probes {
                    let mut x = 0u64;
                    for l in &self.lanes[..5] {
                        x ^= l.get(p);
                    }
                    acc = acc.wrapping_add(x);
                }
                acc as u128
            }
        }
    }
}

// ---------------------------------------------------------------- SoaBitpack

#[inline(always)]
fn lane_mask(bits: u32) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Random-access decode of value `i` from a lane packed at `bits` per value,
/// little-endian bit order, values may straddle words.
#[inline(always)]
pub fn get(lane: &[u64], bits: u32, i: usize) -> u64 {
    let off = i * bits as usize;
    let w = off >> 6;
    let s = (off & 63) as u32;
    let mut v = lane[w] >> s;
    if s + bits > 64 {
        v |= lane[w + 1] << (64 - s);
    }
    v & lane_mask(bits)
}

fn put(lane: &mut [u64], bits: u32, i: usize, v: u64) {
    let v = mask_to(v, bits);
    let off = i * bits as usize;
    let w = off >> 6;
    let s = (off & 63) as u32;
    lane[w] |= v << s;
    if s + bits > 64 {
        lane[w + 1] |= v >> (64 - s);
    }
}

/// Sequential unpacker: keeps a running bit offset, no per-element division.
struct Unpacker<'a> {
    lane: &'a [u64],
    bits: u32,
    mask: u64,
    off: usize,
}

impl<'a> Unpacker<'a> {
    fn new(lane: &'a [u64], bits: u32) -> Self {
        Unpacker {
            lane,
            bits,
            mask: lane_mask(bits),
            off: 0,
        }
    }

    #[inline(always)]
    fn take(&mut self) -> u64 {
        let w = self.off >> 6;
        let s = (self.off & 63) as u32;
        let mut v = self.lane[w] >> s;
        if s + self.bits > 64 {
            v |= self.lane[w + 1] << (64 - s);
        }
        self.off += self.bits as usize;
        v & self.mask
    }
}

/// Per-field bit-packed SoA at exactly `field_bits(f)` bits per value.
pub struct SoaBitpack {
    lanes: Vec<Box<[u64]>>,
    bits: [u32; FIELDS],
    n: usize,
    pres: Vec<u64>,
}

impl SoaBitpack {
    pub fn build(fx: &Fixture) -> SoaBitpack {
        let n = fx.n;
        let mut bits = [0u32; FIELDS];
        let mut lanes = Vec::with_capacity(FIELDS);
        for (f, b) in bits.iter_mut().enumerate() {
            *b = field_bits(f);
            let mut lane = vec![0u64; (n * *b as usize).div_ceil(64)];
            for i in 0..n {
                put(&mut lane, *b, i, value(fx.seed, i, f));
            }
            lanes.push(lane.into_boxed_slice());
        }
        SoaBitpack {
            lanes,
            bits,
            n,
            pres: build_presence(fx),
        }
    }

    fn unpacker(&self, f: usize) -> Unpacker<'_> {
        Unpacker::new(&self.lanes[f], self.bits[f])
    }
}

impl Arm for SoaBitpack {
    fn name(&self) -> &'static str {
        "soa-bitpack"
    }

    fn bytes_resident(&self) -> usize {
        self.lanes.iter().map(|l| l.len() * 8).sum::<usize>() + self.pres.len() * 8
    }

    fn storage_bits(&self, f: usize) -> Option<u32> {
        Some(self.bits[f])
    }

    fn run(&self, q: &Query<'_>) -> u128 {
        match *q {
            Query::CountWhere => {
                let mut ua = self.unpacker(0);
                let mut c = 0u64;
                for _ in 0..self.n {
                    c += (ua.take() == K) as u64;
                }
                c as u128
            }
            Query::Exists => {
                let mut ua = self.unpacker(0);
                for _ in 0..self.n {
                    if ua.take() == K {
                        return 1;
                    }
                }
                0
            }
            Query::SumWhere => {
                let mut ua = self.unpacker(0);
                let mut ub = self.unpacker(1);
                let mut s = 0u128;
                for _ in 0..self.n {
                    let a = ua.take();
                    let b = ub.take();
                    if a == K {
                        s += b as u128;
                    }
                }
                s
            }
            Query::SumPresentC => {
                let mut ua = self.unpacker(0);
                let mut uc = self.unpacker(2);
                let mut s = 0u128;
                for i in 0..self.n {
                    let a = ua.take();
                    let c = uc.take();
                    if a == K && bit(&self.pres, i) {
                        s += c as u128;
                    }
                }
                s
            }
            Query::DemandFold { d } => {
                let mut acc = 0u64;
                for f in 0..d.min(FIELDS) {
                    let fx = f as u64;
                    let mut u = self.unpacker(f);
                    for _ in 0..self.n {
                        acc = acc.wrapping_add(u.take() ^ fx);
                    }
                }
                acc as u128
            }
            Query::Point { probes } => {
                let mut acc = 0u64;
                for &p in probes {
                    let mut x = 0u64;
                    for f in 0..5 {
                        x ^= get(&self.lanes[f], self.bits[f], p);
                    }
                    acc = acc.wrapping_add(x);
                }
                acc as u128
            }
        }
    }
}

// ---------------------------------------------------------------- HashRow

/// Null comparator (H4): rows keyed by global guid; every element access is an
/// identity lookup `map[&guid]`, with guids walked in canonical order.
pub struct HashRow {
    map: HashMap<u64, [u64; FIELDS]>,
    order: Vec<u64>,
    pres: HashMap<u64, bool>,
    lookups: Cell<u64>,
}

impl HashRow {
    pub fn build(fx: &Fixture) -> HashRow {
        let mut map = HashMap::with_capacity(fx.n);
        let mut pres = HashMap::with_capacity(fx.n);
        for (i, &g) in fx.guids.iter().enumerate() {
            let mut r = [0u64; FIELDS];
            for (f, slot) in r.iter_mut().enumerate() {
                *slot = value(fx.seed, i, f);
            }
            map.insert(g, r);
            pres.insert(g, present(fx.seed, i, 2));
        }
        HashRow {
            map,
            order: fx.guids.clone(),
            pres,
            lookups: Cell::new(0),
        }
    }

    /// Identity lookups performed by `run` since construction or the last reset.
    pub fn lookups(&self) -> u64 {
        self.lookups.get()
    }

    pub fn reset_lookups(&self) {
        self.lookups.set(0);
    }

    #[inline(always)]
    fn row(&self, g: u64) -> &[u64; FIELDS] {
        self.lookups.set(self.lookups.get() + 1);
        &self.map[&g]
    }

    #[inline(always)]
    fn is_present(&self, g: u64) -> bool {
        self.lookups.set(self.lookups.get() + 1);
        self.pres[&g]
    }
}

impl Arm for HashRow {
    fn name(&self) -> &'static str {
        "hash-row"
    }

    /// Estimate: table capacity times (key + row + control byte), plus the
    /// presence table and the order vector.
    fn bytes_resident(&self) -> usize {
        self.map.capacity() * (8 + FIELDS * 8 + 1)
            + self.pres.capacity() * (8 + 1 + 1)
            + self.order.len() * 8
    }

    fn run(&self, q: &Query<'_>) -> u128 {
        match *q {
            Query::CountWhere => {
                let mut c = 0u64;
                for &g in &self.order {
                    c += (self.row(g)[0] == K) as u64;
                }
                c as u128
            }
            Query::Exists => {
                for &g in &self.order {
                    if self.row(g)[0] == K {
                        return 1;
                    }
                }
                0
            }
            Query::SumWhere => {
                let mut s = 0u128;
                for &g in &self.order {
                    let r = self.row(g);
                    if r[0] == K {
                        s += r[1] as u128;
                    }
                }
                s
            }
            Query::SumPresentC => {
                let mut s = 0u128;
                for &g in &self.order {
                    let r = self.row(g);
                    if r[0] == K && self.is_present(g) {
                        s += r[2] as u128;
                    }
                }
                s
            }
            Query::DemandFold { d } => {
                let d = d.min(FIELDS);
                let mut acc = 0u64;
                for &g in &self.order {
                    let r = self.row(g);
                    for (f, &v) in r.iter().take(d).enumerate() {
                        acc = acc.wrapping_add(v ^ f as u64);
                    }
                }
                acc as u128
            }
            Query::Point { probes } => {
                let mut acc = 0u64;
                for &p in probes {
                    let r = self.row(self.order[p]);
                    acc = acc.wrapping_add(r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4]);
                }
                acc as u128
            }
        }
    }
}
