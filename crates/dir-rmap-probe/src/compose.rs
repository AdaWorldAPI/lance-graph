//! D-RFT-1, lab only: one typed composition over the DR256 register's shared
//! slot coordinate.
//!
//! STATUS: PROBE. Nothing here is a production path.
//!
//! The composition is user -> group membership ([`Page::member`], slots)
//! composed with role carriers ([`Qr::role_slots`], the slots whose reach
//! carries role `r`) over the SAME 256-slot coordinate, bound by one [`Qr`]
//! (one [`Witness`]). Per coordinate (user `i`, role `r`) the path mask is
//!
//! ```text
//! member[i] & valid[i] & role_slots[r]
//! ```
//!
//! Its population count is the Bag(N) multiplicity: the number of distinct
//! DIRECT groups through which `r` is reached. Its existence is the Distinct
//! answer: whether the path mask is non-empty. There is no value-directed
//! gather, no semijoin, and no tuple: the answer lives in coordinate lanes
//! indexed `i * 64 + r` and in a bit plane over the same coordinates, and
//! every count, existence and equality is a mask-risc terminal.
//!
//! The join key is a slot. A bit set in `member` but clear in `valid` is a
//! NULL join key: present payload, absent value. It matches nothing, as a
//! NULL key matches nothing in SQL.
//!
//! Parallel paths combine by [`Law`]. `Count` and `Exists` are the same
//! computation read at two strata (see [`Composed::answer`]). `Xor` is
//! refused: parity loses an even number of paths, so the diamond (two direct
//! groups reaching one role) would read zero, an answer neither stratum
//! gives.

use lance_graph_mask_risc::{
    Foreign, Out, Planes, Program, Scratch, Value, execute_into, words_for,
};
use lance_graph_r2il::{FoldDialect, Val, procs};

use crate::focus::{Row, SLOTS};
use crate::prep::{Refusal, Witness};
use crate::qr::{Page, Qr};

/// How parallel paths combine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Law {
    /// Number of paths (Bag(N)).
    Count,
    /// Whether any path exists (Distinct).
    Exists,
    /// Parity of the number of paths. Refused: an even number of parallel
    /// paths reads zero.
    Xor,
}

/// Which answer the two sides' cardinality witness guarantees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stratum {
    /// One side is functional (at most one member per user row, or at most
    /// one carrier slot per role), so no coordinate has more than one path.
    Distinct,
    /// Both sides fan out: a coordinate can have several paths.
    Bag,
}

/// Why a composition was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComposeError {
    /// The operands were bound to another version than `current`.
    Stale,
    /// The two operands were bound to different registers (witnesses).
    SealMismatch,
    /// [`Law::Xor`] is not a composition law here.
    XorLaw,
    /// A checked bag sum would have wrapped.
    Overflow,
    /// Any other refusal from the preparation layer.
    Refusal(Refusal),
}

impl From<Refusal> for ComposeError {
    /// `Refusal::Stale` is [`ComposeError::Stale`]; every other refusal is
    /// carried as [`ComposeError::Refusal`].
    fn from(r: Refusal) -> Self {
        match r {
            Refusal::Stale => ComposeError::Stale,
            other => ComposeError::Refusal(other),
        }
    }
}

/// Left operand: one page's membership rows plus their presence plane.
///
/// A bit set in `member` but clear in `valid` is a NULL join key: present
/// payload, absent value. It matches nothing.
pub struct Members<'a> {
    /// The register version the rows were bound against.
    pub witness: &'a Witness,
    /// Users in this page (at most 256).
    pub users: usize,
    /// Row `i`: user `i`'s direct security groups as slots.
    pub member: &'a [Row; SLOTS],
    /// Presence plane, same coordinates as `member`.
    pub valid: [Row; SLOTS],
}

impl<'a> Members<'a> {
    /// The page's rows with every key present (`valid == member`): 8 KiB of
    /// presence plane is the only thing copied.
    pub fn of(qr: &'a Qr, page: &'a Page, users: usize) -> Self {
        assert!(users <= SLOTS, "a page holds at most {SLOTS} users");
        Members {
            witness: &qr.witness,
            users,
            member: &page.member,
            valid: page.member,
        }
    }
}

/// Right operand: role carriers over the same slots, held by the register
/// in both orientations.
pub struct Carriers<'a> {
    /// The register version the carriers were bound against.
    pub witness: &'a Witness,
    /// Role -> the slots whose reach carries it.
    pub role_slots: &'a [Row; 64],
    /// Slot -> the roles its reach carries (one word per slot).
    pub role: &'a [u64; SLOTS],
}

impl<'a> Carriers<'a> {
    /// The register's role carriers.
    pub fn of(qr: &'a Qr) -> Self {
        Carriers {
            witness: &qr.witness,
            role_slots: &qr.role_slots,
            role: &qr.role,
        }
    }
}

/// The composition's answer, in coordinates.
pub struct Composed {
    /// Which answer the cardinality witness guarantees.
    pub stratum: Stratum,
    /// Users in the page.
    pub users: usize,
    /// Bag multiplicity per coordinate `i * 64 + r`.
    pub counts: Vec<u16>,
    /// Distinct support: bit `i * 64 + r` set when the count is non-zero.
    /// Word `i` is user `i`'s role set.
    pub support: Vec<u64>,
    /// Sum of `counts`, checked.
    pub bag: u64,
    /// Popcount of `support` (mask-risc `Count` over the support plane).
    pub distinct: u64,
    /// The cardinality witness: max path-mask fan on the left (members per
    /// user row).
    pub left_fan: u64,
    /// The cardinality witness: max carrier slots per role on the right.
    pub right_fan: u64,
}

impl Composed {
    /// The answer the caller demands under `law`.
    ///
    /// The law only selects which ANSWER is read (`Count` -> [`Self::bag`],
    /// `Exists` -> [`Self::distinct`]); it never selects a different
    /// computation. `Xor` is refused.
    pub fn answer(&self, law: Law) -> Result<u64, ComposeError> {
        match law {
            Law::Count => Ok(self.bag),
            Law::Exists => Ok(self.distinct),
            Law::Xor => Err(ComposeError::XorLaw),
        }
    }
}

/// Checked bag accumulation: `acc + n`, or [`ComposeError::Overflow`] where
/// the sum would wrap.
pub fn add_bag(acc: u64, n: u64) -> Result<u64, ComposeError> {
    acc.checked_add(n).ok_or(ComposeError::Overflow)
}

/// The scalar a one-result r2il procedure left on the stack.
fn take(stack: &mut Vec<Val>) -> u64 {
    match stack.pop() {
        Some(Val::Scalar(n)) if stack.is_empty() => {
            u64::try_from(n).expect("a count or a 0/1 answer is non-negative")
        }
        other => panic!("an r2il procedure leaves one scalar, got {other:?}"),
    }
}

/// An r2il procedure compiled once and replayed over planes of the same
/// shape: a prepared statement.
///
/// A [`FoldDialect`] builds the mask-risc [`Program`] once from an r2il
/// procedure over planes `0..k`; every later call runs that program over its
/// own `k` zero-copy plane slices, with one scratch reused. Per call the
/// cost is mask-risc execution alone, not program construction.
struct Replay {
    k: usize,
    n_rows: usize,
    program: Program,
    scratch: Scratch<'static>,
}

impl Replay {
    /// Compile `body` over `k` planes of `n_rows` rows. `keep` lends the
    /// dialect a sink, for a body that ends in `KEEP`.
    fn compile(
        k: usize,
        n_rows: usize,
        keep: bool,
        body: impl FnOnce(&mut FoldDialect<'_>, &mut Vec<Val>, &[u16]),
    ) -> Self {
        let zero = vec![0u64; words_for(n_rows)];
        let mut sink = vec![0u64; words_for(n_rows)];
        let masks = vec![zero.as_slice(); k];
        let planes = Planes {
            n_rows,
            masks: &masks,
            lanes: &[],
        };
        let ix: Vec<u16> = (0..k)
            .map(|p| u16::try_from(p).expect("at most 256 planes"))
            .collect();
        let mut d = if keep {
            FoldDialect::with_keep(&planes, &Foreign::NONE, &mut sink)
        } else {
            FoldDialect::new(&planes, &Foreign::NONE)
        };
        let mut stack = Vec::new();
        body(&mut d, &mut stack, &ix);
        let program = d.last_program().expect("the body folds").clone();
        let scratch = Scratch::for_program(&program, n_rows).expect("scratch");
        Replay {
            k,
            n_rows,
            program,
            scratch,
        }
    }

    /// `procs::count_and` over `k` planes.
    fn count_and(k: usize, n_rows: usize) -> Self {
        Self::compile(k, n_rows, false, |d, stack, ix| {
            procs::count_and(d, stack, ix).expect("compiles");
            take(stack);
        })
    }

    /// `procs::union` of `k` planes, then `procs::keep`.
    fn keep_union(k: usize, n_rows: usize) -> Self {
        Self::compile(k, n_rows, true, |d, stack, ix| {
            procs::union(d, stack, ix).expect("compiles");
            procs::keep(d, stack).expect("compiles");
        })
    }

    /// `procs::intersect` of `k` planes, then `procs::keep`.
    fn keep_intersect(k: usize, n_rows: usize) -> Self {
        Self::compile(k, n_rows, true, |d, stack, ix| {
            procs::intersect(d, stack, ix).expect("compiles");
            procs::keep(d, stack).expect("compiles");
        })
    }

    fn planes<'m>(&self, masks: &'m [&'m [u64]]) -> Planes<'m> {
        assert_eq!(masks.len(), self.k, "compiled for {} planes", self.k);
        Planes {
            n_rows: self.n_rows,
            masks,
            lanes: &[],
        }
    }

    /// Run a compiled count over `masks`.
    fn count(&mut self, masks: &[&[u64]]) -> u64 {
        let planes = self.planes(masks);
        match execute_into(
            &self.program,
            &planes,
            &Foreign::NONE,
            &mut self.scratch,
            Out::None,
        ) {
            Ok(Value::Count(n)) => n as u64,
            other => panic!("a count program answers a Count, got {other:?}"),
        }
    }

    /// Run a compiled keep over `masks`, writing the mask into `out`.
    fn keep(&mut self, masks: &[&[u64]], out: &mut [u64]) {
        let planes = self.planes(masks);
        match execute_into(
            &self.program,
            &planes,
            &Foreign::NONE,
            &mut self.scratch,
            Out::Mask(out),
        ) {
            Ok(Value::Mask(_)) => {}
            other => panic!("a keep program answers a Mask, got {other:?}"),
        }
    }
}

/// Population count of `words` over `n_rows` rows (`procs::count_and` of
/// one plane, i.e. `LOAD` then `PopCount`).
fn plane_count(words: &[u64], n_rows: usize) -> u64 {
    Replay::count_and(1, n_rows).count(&[words])
}

/// The set positions of `words`: the pointer table that inverts a plane to
/// its local positions (the one thing the inversion materializes).
fn positions(words: &[u64], out: &mut Vec<usize>) {
    out.clear();
    for (w, &word) in words.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            out.push(w * 64 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}

/// The composition's prepared programs: every r2il procedure it runs,
/// compiled once. Building one is the prepare step; [`Composer::compose`]
/// is execution alone and reuses them (unions are compiled once per arity,
/// on first use).
pub struct Composer {
    present_of: Replay,
    path: Replay,
    fan: Replay,
    union_of: Vec<Option<Replay>>,
}

impl Default for Composer {
    fn default() -> Self {
        Self::new()
    }
}

impl Composer {
    /// Compile the composition's procedures.
    pub fn new() -> Self {
        Composer {
            present_of: Replay::keep_intersect(2, SLOTS),
            path: Replay::count_and(3, SLOTS),
            fan: Replay::count_and(1, SLOTS),
            union_of: (0..=SLOTS).map(|_| None).collect(),
        }
    }

    /// Compose membership with role carriers.
    ///
    /// Order of refusal: `Xor` first (the law is wrong whatever the operands
    /// are), then the operands' own seal ([`ComposeError::SealMismatch`]), then
    /// the caller's `current` witness ([`ComposeError::Stale`]).
    ///
    /// `Law::Count` and `Law::Exists` return the same [`Composed`]; the caller
    /// reads [`Composed::bag`] or [`Composed::distinct`] (or
    /// [`Composed::answer`]).
    pub fn compose(
        &mut self,
        left: &Members<'_>,
        right: &Carriers<'_>,
        current: &Witness,
        law: Law,
    ) -> Result<Composed, ComposeError> {
        if law == Law::Xor {
            return Err(ComposeError::XorLaw);
        }
        if left.witness != right.witness {
            return Err(ComposeError::SealMismatch);
        }
        if left.witness != current {
            return Err(ComposeError::Stale);
        }
        assert!(left.users <= SLOTS, "a page holds at most {SLOTS} users");
        let users = left.users;

        // Support first. A user's roles are the union of the role words of its
        // present join keys (`member & valid`), so the only coordinates that can
        // hold a path are the bits of that union; every other coordinate has
        // count 0 by construction and is never evaluated. Each step is an r2il
        // procedure compiled once and replayed over zero-copy row slices:
        //   present_i  = keep(intersect(member_i, valid_i))       over 256 slots
        //   support_i  = keep(union(role[s] for s in present_i))   over 64 roles
        //   count(i,r) = count_and(member_i, valid_i, carriers_r)  for r in support_i
        // The two pointer tables (present slots, supported roles) are the
        // inversion from a plane to its local positions.
        let mut counts = vec![0u16; users * 64];
        let mut support = vec![0u64; users];
        let mut bag = 0u64;
        let mut left_fan = 0u64;
        let mut present = [0u64; 4];
        let (mut keys, mut roles): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        let mut words: Vec<&[u64]> = Vec::with_capacity(SLOTS);
        for i in 0..users {
            let (m, v) = (&left.member[i][..], &left.valid[i][..]);
            self.present_of.keep(&[m, v], &mut present);
            positions(&present, &mut keys);
            left_fan = left_fan.max(keys.len() as u64);
            if keys.is_empty() {
                continue;
            }
            words.clear();
            words.extend(keys.iter().map(|&s| std::slice::from_ref(&right.role[s])));
            self.union_of[keys.len()]
                .get_or_insert_with(|| Replay::keep_union(keys.len(), 64))
                .keep(&words, &mut support[i..=i]);
            positions(&support[i..=i], &mut roles);
            for &r in &roles {
                let c = self.path.count(&[m, v, &right.role_slots[r][..]]);
                // A path mask lives over 256 slots, so its count is at most 256.
                counts[i * 64 + r] = u16::try_from(c).expect("a count over 256 slots fits u16");
                bag = add_bag(bag, c)?;
            }
        }
        let distinct = if users == 0 {
            0
        } else {
            plane_count(&support, users * 64)
        };

        // Cardinality witness. Left: present join keys per user row (above).
        // Right: carrier slots per role.
        let mut right_fan = 0u64;
        for carriers in right.role_slots.iter() {
            right_fan = right_fan.max(self.fan.count(&[&carriers[..]]));
        }

        let stratum = if left_fan <= 1 || right_fan <= 1 {
            Stratum::Distinct
        } else {
            Stratum::Bag
        };
        Ok(Composed {
            stratum,
            users,
            counts,
            support,
            bag,
            distinct,
            left_fan,
            right_fan,
        })
    }
}

/// [`Composer::compose`] with a freshly prepared [`Composer`].
pub fn compose(
    left: &Members<'_>,
    right: &Carriers<'_>,
    current: &Witness,
    law: Law,
) -> Result<Composed, ComposeError> {
    Composer::new().compose(left, right, current, law)
}

/// Whether two planes differ over `n_rows` rows: `procs::differ`, i.e.
/// `IntXor` then `ANY`. Both planes must be `n_rows.div_ceil(64)` words long.
pub fn differ(a: &[u64], b: &[u64], n_rows: usize) -> bool {
    let words = n_rows.div_ceil(64);
    assert_eq!(a.len(), words, "plane a spans n_rows");
    assert_eq!(b.len(), words, "plane b spans n_rows");
    let masks = [a, b];
    let planes = Planes {
        n_rows,
        masks: &masks,
        lanes: &[],
    };
    let mut d = FoldDialect::new(&planes, &Foreign::NONE);
    let mut stack = Vec::new();
    procs::differ(&mut d, &mut stack, 0, 1).expect("two resident planes");
    take(&mut stack) == 1
}
