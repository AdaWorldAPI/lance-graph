//! Lane K — codebook indices over an mmapped file.
//!
//! Every row becomes two small integers before any aggregation happens:
//!
//! - **station index:** a dense id assigned in order of first appearance,
//!   per worker. The codebook entry is a slice of the input, so a station
//!   name is never copied into an owned string on the row path.
//! - **temperature code:** `tenths + 999`, so `-99.9 → 0` and `+99.9 → 1998`.
//!   The map preserves order and is affine, so min and max run on codes and
//!   the sum is corrected once per station at the end:
//!   `Σ tenths = Σ code − 999·n`. No row is decoded.
//!
//! Aggregation is then four dense arrays indexed by station id
//! (`min/max/sum/count`). After the pass, the per-worker codebooks merge
//! into one, and one sort of the distinct names (about 400) gives the
//! alphabetical output order. The rows are never reordered.
//!
//! Two temperature paths, so the codebook idea is measured on the
//! temperature too:
//!
//! - [`TempPath::Parse`]: lane S's branch-light parse, then `+ 999`.
//! - [`TempPath::Table`]: the field's bytes (3 to 5) packed into a `u64`
//!   and looked up in a 4,096-slot table built once from all 1,999 values
//!   (plus `-0.0`). A key the table does not hold falls back to the parse,
//!   so an unexpected spelling is slower, never wrong.
//!
//! The scan is lane S's SWAR delimiter find, unchanged, so the variables
//! against lane S are the grouping and the temperature path.
//!
//! ## mmap
//!
//! [`Mapped`] maps a file read-only with the platform `mmap` (declared
//! here, no dependency; Unix only). The `run` subcommand reads the file
//! before its timer starts, which hides what mmap saves. The `wall`
//! subcommand (`main.rs`) times from opening the file to the finished
//! result, so lane K (mmap) and lane S (`fs::read`) can be compared end to
//! end.
//!
//! ## Rules this lane keeps
//!
//! - Nothing depends on the station list: codebooks are built from the
//!   names actually seen, and every lookup verifies the full name bytes.
//! - At most 16,384 distinct names per worker (the codebook's slot count
//!   is fixed at 32,768); the 1BRC limit is 10,000.

use crate::Stats;
use std::collections::BTreeMap;

const SEMI_BCAST: u64 = 0x3B3B_3B3B_3B3B_3B3B;
const NL_BCAST: u64 = 0x0A0A_0A0A_0A0A_0A0A;
const ONES: u64 = 0x0101_0101_0101_0101;
const HIGH: u64 = 0x8080_8080_8080_8080;

/// Offset that makes every temperature code non-negative.
pub const TEMP_OFFSET: i32 = 999;

#[inline(always)]
fn match_mask(word: u64, bcast: u64) -> u64 {
    let x = word ^ bcast;
    x.wrapping_sub(ONES) & !x & HIGH
}

/// First index `>= i` where `data[idx]` is `bcast`'s byte (lane S's find).
#[inline(always)]
fn find(data: &[u8], mut i: usize, bcast: u64) -> usize {
    let len = data.len();
    while i + 8 <= len {
        let word = u64::from_le_bytes(data[i..i + 8].try_into().unwrap());
        let m = match_mask(word, bcast);
        if m != 0 {
            return i + ((m.trailing_zeros() >> 3) as usize);
        }
        i += 8;
    }
    let b = bcast as u8;
    while i < len && data[i] != b {
        i += 1;
    }
    i
}

/// Lane S's parse of `-?\d?\d.\d` to tenths.
#[inline(always)]
fn parse_tenths(s: &[u8]) -> i32 {
    let neg = s[0] == b'-';
    let d = if neg { &s[1..] } else { s };
    let v = if d.len() == 3 {
        (d[0] - b'0') as i32 * 10 + (d[2] - b'0') as i32
    } else {
        (d[0] - b'0') as i32 * 100 + (d[1] - b'0') as i32 * 10 + (d[3] - b'0') as i32
    };
    if neg {
        -v
    } else {
        v
    }
}

/// How the temperature field becomes a code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TempPath {
    Parse,
    Table,
}

/// The temperature codebook: every 1BRC temperature spelling → its code.
pub struct TempTable {
    keys: Box<[u64; TEMP_SLOTS]>,
    codes: Box<[u16; TEMP_SLOTS]>,
}

const TEMP_SLOTS: usize = 4096;

#[inline(always)]
fn temp_key(field: &[u8]) -> u64 {
    let mut k = [0u8; 8];
    k[..field.len()].copy_from_slice(field);
    u64::from_le_bytes(k)
}

#[inline(always)]
fn temp_slot(key: u64) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 52) as usize
}

impl TempTable {
    /// Build the table from every value in `-99.9..=99.9`, plus `-0.0`.
    pub fn build() -> Self {
        let mut t = TempTable {
            keys: Box::new([0; TEMP_SLOTS]),
            codes: Box::new([0; TEMP_SLOTS]),
        };
        for tenths in -999..=999i32 {
            let (whole, frac) = (tenths.abs() / 10, tenths.abs() % 10);
            let s = if tenths < 0 {
                format!("-{whole}.{frac}")
            } else {
                format!("{whole}.{frac}")
            };
            t.insert(s.as_bytes(), (tenths + TEMP_OFFSET) as u16);
        }
        t.insert(b"-0.0", TEMP_OFFSET as u16);
        t
    }

    fn insert(&mut self, field: &[u8], code: u16) {
        let key = temp_key(field);
        let mut s = temp_slot(key);
        while self.keys[s] != 0 && self.keys[s] != key {
            s = (s + 1) & (TEMP_SLOTS - 1);
        }
        self.keys[s] = key;
        self.codes[s] = code;
    }

    /// The code of `field`, or `None` when the spelling is not in the table.
    #[inline(always)]
    pub fn get(&self, field: &[u8]) -> Option<u16> {
        if field.len() > 8 {
            return None;
        }
        let key = temp_key(field);
        let mut s = temp_slot(key);
        loop {
            let k = self.keys[s];
            if k == key {
                return Some(self.codes[s]);
            }
            if k == 0 {
                return None;
            }
            s = (s + 1) & (TEMP_SLOTS - 1);
        }
    }
}

const NAME_SLOTS: usize = 1 << 15;
const MAX_NAMES: usize = NAME_SLOTS / 2;

#[inline(always)]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// One worker's codebook and its per-station state, indexed by station id.
/// Names are slices of the input, never copies.
pub struct Codebook<'a> {
    slots: Vec<u32>,
    hashes: Vec<u64>,
    pub names: Vec<&'a [u8]>,
    pub min: Vec<u16>,
    pub max: Vec<u16>,
    pub sum: Vec<u64>,
    pub count: Vec<u32>,
}

impl<'a> Codebook<'a> {
    pub fn new() -> Self {
        Codebook {
            slots: vec![0; NAME_SLOTS],
            hashes: Vec::new(),
            names: Vec::new(),
            min: Vec::new(),
            max: Vec::new(),
            sum: Vec::new(),
            count: Vec::new(),
        }
    }

    /// The station id of `name`, appending a new entry on first sight.
    #[inline(always)]
    pub fn id(&mut self, name: &'a [u8]) -> usize {
        let h = fnv1a64(name);
        let mut s = (h as usize) & (NAME_SLOTS - 1);
        loop {
            let v = self.slots[s];
            if v == 0 {
                assert!(
                    self.names.len() < MAX_NAMES,
                    "lane K: too many distinct stations"
                );
                let id = self.names.len();
                self.slots[s] = id as u32 + 1;
                self.hashes.push(h);
                self.names.push(name);
                self.min.push(u16::MAX);
                self.max.push(0);
                self.sum.push(0);
                self.count.push(0);
                return id;
            }
            let id = (v - 1) as usize;
            if self.hashes[id] == h && self.names[id] == name {
                return id;
            }
            s = (s + 1) & (NAME_SLOTS - 1);
        }
    }

    #[inline(always)]
    fn observe(&mut self, id: usize, code: u16) {
        if code < self.min[id] {
            self.min[id] = code;
        }
        if code > self.max[id] {
            self.max[id] = code;
        }
        self.sum[id] += code as u64;
        self.count[id] += 1;
    }
}

impl Default for Codebook<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fold one newline-aligned chunk into a fresh codebook.
pub fn accumulate<'a>(data: &'a [u8], path: TempPath, table: &TempTable) -> Codebook<'a> {
    let mut cb = Codebook::new();
    let len = data.len();
    let mut i = 0usize;
    while i < len {
        let semi = find(data, i, SEMI_BCAST);
        let name = &data[i..semi];
        let nl = find(data, semi + 1, NL_BCAST);
        let field = &data[semi + 1..nl];
        let code = match path {
            TempPath::Parse => (parse_tenths(field) + TEMP_OFFSET) as u16,
            TempPath::Table => match table.get(field) {
                Some(c) => c,
                None => (parse_tenths(field) + TEMP_OFFSET) as u16,
            },
        };
        let id = cb.id(name);
        cb.observe(id, code);
        i = nl + 1;
    }
    cb
}

/// Merge per-worker codebooks into one global codebook in alphabetical
/// order and decode once per station.
///
/// The order is computed on the distinct names only, so the cost does not
/// depend on the number of rows.
pub fn merge(books: &[Codebook<'_>]) -> BTreeMap<String, Stats> {
    merge_ordered(books).into_iter().collect()
}

/// [`merge`] as an alphabetically ordered list: the order comes from the one
/// sort of distinct names, not from a sorted container.
pub fn merge_ordered(books: &[Codebook<'_>]) -> Vec<(String, Stats)> {
    // Global ids in first-appearance order across workers.
    let mut global: Codebook<'_> = Codebook::new();
    for b in books {
        for (local, &name) in b.names.iter().enumerate() {
            if b.count[local] == 0 {
                continue;
            }
            let g = global.id(name);
            global.min[g] = global.min[g].min(b.min[local]);
            global.max[g] = global.max[g].max(b.max[local]);
            global.sum[g] += b.sum[local];
            global.count[g] += b.count[local];
        }
    }
    // One sort of the distinct names gives the output order; the per-station
    // state is permuted, the rows never are.
    let mut order: Vec<usize> = (0..global.names.len()).collect();
    order.sort_unstable_by(|&a, &b| global.names[a].cmp(global.names[b]));
    let mut out = Vec::with_capacity(order.len());
    for g in order {
        let n = global.count[g] as i64;
        let stats = Stats {
            min: global.min[g] as i32 - TEMP_OFFSET,
            max: global.max[g] as i32 - TEMP_OFFSET,
            sum: global.sum[g] as i64 - TEMP_OFFSET as i64 * n,
            count: global.count[g],
        };
        let name = std::str::from_utf8(global.names[g]).expect("station name is valid utf8");
        out.push((name.to_string(), stats));
    }
    out
}

/// Lane K over an in-memory buffer: `workers` newline-aligned chunks, one
/// codebook each, then the merge.
pub fn lane_k_codebook(data: &[u8], workers: usize, path: TempPath) -> BTreeMap<String, Stats> {
    let table = TempTable::build();
    let bounds = crate::chunk_bounds(data, workers.max(1));
    let books: Vec<Codebook<'_>> = std::thread::scope(|scope| {
        let table = &table;
        let handles: Vec<_> = bounds
            .iter()
            .map(|&(start, end)| {
                let slice = &data[start..end];
                scope.spawn(move || accumulate(slice, path, table))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("lane K worker panicked"))
            .collect()
    });
    merge(&books)
}

/// A read-only memory mapping of a whole file.
#[cfg(unix)]
pub struct Mapped {
    ptr: *const u8,
    len: usize,
}

#[cfg(unix)]
mod sys {
    use core::ffi::c_void;
    pub const PROT_READ: i32 = 1;
    pub const MAP_PRIVATE: i32 = 2;
    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
    }
}

#[cfg(unix)]
impl Mapped {
    /// Map `path` read-only. An empty file maps to an empty slice.
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        use std::os::fd::AsRawFd;
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Mapped {
                ptr: core::ptr::NonNull::<u8>::dangling().as_ptr(),
                len: 0,
            });
        }
        // SAFETY: a fresh read-only private mapping of an open file
        // descriptor; the result is checked against MAP_FAILED. The mapping
        // outlives the descriptor, which may be closed after this call.
        let p = unsafe {
            sys::mmap(
                core::ptr::null_mut(),
                len,
                sys::PROT_READ,
                sys::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if p as isize == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Mapped {
            ptr: p as *const u8,
            len,
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` points at `len` readable bytes for as long as `self`
        // lives (or is dangling with `len == 0`). The probe assumes nobody
        // truncates the file while it is mapped.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[cfg(unix)]
impl Drop for Mapped {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: unmaps exactly the region `open` mapped.
            unsafe {
                sys::munmap(self.ptr as *mut core::ffi::c_void, self.len);
            }
        }
    }
}

// SAFETY: the mapping is read-only and owned; sharing `&[u8]` views of it
// across threads is the same as sharing any immutable byte slice.
#[cfg(unix)]
unsafe impl Sync for Mapped {}
#[cfg(unix)]
unsafe impl Send for Mapped {}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus(rows: u64, seed: u64) -> (std::path::PathBuf, Vec<u8>) {
        let path = std::env::temp_dir().join(format!(
            "onebrc_probe_test_k_{}_{rows}_{seed}.txt",
            std::process::id()
        ));
        crate::gen::gen(&path, rows, seed).expect("gen");
        let data = std::fs::read(&path).expect("read corpus");
        (path, data)
    }

    #[test]
    fn the_temp_table_holds_every_value_and_decodes_like_the_parse() {
        let t = TempTable::build();
        for tenths in -999..=999i32 {
            let (w, f) = (tenths.abs() / 10, tenths.abs() % 10);
            let s = if tenths < 0 {
                format!("-{w}.{f}")
            } else {
                format!("{w}.{f}")
            };
            assert_eq!(
                t.get(s.as_bytes()),
                Some((tenths + TEMP_OFFSET) as u16),
                "{s}"
            );
            assert_eq!(
                parse_tenths(s.as_bytes()) + TEMP_OFFSET,
                (tenths + TEMP_OFFSET),
                "{s}"
            );
        }
        assert_eq!(t.get(b"-0.0"), Some(TEMP_OFFSET as u16));
        // Spellings outside the format are absent, so the lane falls back.
        assert_eq!(t.get(b"100.0"), None);
        assert_eq!(t.get(b"1.00"), None);
    }

    #[test]
    fn codes_preserve_order_so_min_and_max_need_no_decode() {
        let t = TempTable::build();
        let mut prev = None;
        for tenths in -999..=999i32 {
            let (w, f) = (tenths.abs() / 10, tenths.abs() % 10);
            let s = if tenths < 0 {
                format!("-{w}.{f}")
            } else {
                format!("{w}.{f}")
            };
            let c = t.get(s.as_bytes()).unwrap();
            if let Some(p) = prev {
                assert!(c > p);
            }
            prev = Some(c);
        }
    }

    #[test]
    fn lane_k_agrees_with_lane_a_on_both_temperature_paths() {
        let (path, data) = corpus(60_000, 41);
        std::fs::remove_file(&path).ok();
        let a = crate::lane_a_scalar(&data);
        assert!(!a.is_empty());
        for workers in [1, 3, 7] {
            for p in [TempPath::Parse, TempPath::Table] {
                assert_eq!(
                    a,
                    lane_k_codebook(&data, workers, p),
                    "workers {workers} {p:?}"
                );
            }
        }
    }

    #[test]
    fn negative_sums_survive_the_offset_correction() {
        // All-negative station: the sum is corrected by 999·n, not decoded per row.
        let data = b"Frost;-12.5\nFrost;-0.5\nFrost;-99.9\nHeat;99.9\nFrost;-0.0\n";
        let a = crate::lane_a_scalar(data);
        for p in [TempPath::Parse, TempPath::Table] {
            let k = lane_k_codebook(data, 1, p);
            assert_eq!(a, k);
            assert_eq!(k["Frost"].sum, -1129);
            assert_eq!(k["Frost"].min, -999);
            assert_eq!(k["Frost"].max, 0);
        }
    }

    #[test]
    fn station_ids_follow_first_appearance_and_output_is_alphabetical() {
        let data = b"Zeta;1.0\nAlpha;2.0\nMid;3.0\nAlpha;4.0\n";
        let t = TempTable::build();
        let cb = accumulate(data, TempPath::Parse, &t);
        let names: Vec<&[u8]> = cb.names.clone();
        assert_eq!(names, vec![&b"Zeta"[..], b"Alpha", b"Mid"]);
        let out = merge_ordered(&[cb]);
        let keys: Vec<&str> = out.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(keys, vec!["Alpha", "Mid", "Zeta"]);
        assert_eq!(out[0].1.count, 2);
    }

    #[test]
    fn codebook_entries_are_slices_of_the_input_not_copies() {
        let data = b"Oslo;1.0\nRome;2.0\n";
        let t = TempTable::build();
        let cb = accumulate(data, TempPath::Parse, &t);
        let base = data.as_ptr() as usize;
        for n in &cb.names {
            let p = n.as_ptr() as usize;
            assert!(
                p >= base && p < base + data.len(),
                "name not inside the input"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn the_mapping_reads_the_same_bytes_as_fs_read() {
        let (path, data) = corpus(5_000, 7);
        let m = Mapped::open(&path).expect("map");
        assert_eq!(m.as_slice(), &data[..]);
        assert_eq!(
            lane_k_codebook(m.as_slice(), 4, TempPath::Table),
            crate::lane_a_scalar(&data)
        );
        drop(m);
        std::fs::remove_file(&path).ok();
        let empty = std::env::temp_dir().join(format!("onebrc_probe_empty_{}", std::process::id()));
        std::fs::write(&empty, b"").unwrap();
        assert!(Mapped::open(&empty).unwrap().as_slice().is_empty());
        std::fs::remove_file(&empty).ok();
    }
}
