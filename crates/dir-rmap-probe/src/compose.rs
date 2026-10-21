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
//! indexed `i * 64 + r` and in a bit plane over the same coordinates.
//!
//! The register already holds both answers' inputs, and the composition
//! reads them instead of recomputing them:
//!
//! - **Distinct** is the page's resolved role set ([`Page::roles`]), when
//!   every key is present. Nothing is computed.
//! - **Bag** is the bit product of the page's membership matrix (users ×
//!   256 slots) and the register's carrier matrix ([`Qr::role_slots`],
//!   roles × 256 slots), evaluated only on the support:
//!   `count(i, r) = popcount(member[i] & role_slots[r])` for each role `r`
//!   in user `i`'s role set, through `ndarray::simd`. Every other
//!   coordinate is zero by construction.
//!
//! The int8 tile GEMM (`ndarray::simd::gemm_u8_i8`) computes the same
//! product densely; measured at about 19 µs per page (unpack included) on an
//! AMX host, it is slower than the bit product, because the 0/1 bits must first
//! be unpacked to bytes (`examples/rft1_bench.rs` prints both).
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

use lance_graph_mask_risc::{Foreign, Planes};
use lance_graph_r2il::{FoldDialect, Val, procs};
use ndarray::simd::{mask_and_assign, mask_or_assign, popcount_batch_u64};

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

/// Left operand: one page's membership rows, the page's resolved role sets,
/// and an optional presence plane.
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
    /// Row `i`: user `i`'s roles, resolved with the page. Valid only while
    /// every key is present.
    pub roles: &'a [u64; SLOTS],
    /// Presence plane, same coordinates as `member`. `None` means every key
    /// is present; nothing is copied for that case.
    pub valid: Option<&'a [Row; SLOTS]>,
}

impl<'a> Members<'a> {
    /// The page's rows with every key present. Borrows the page; copies
    /// nothing.
    ///
    /// The witness is the page's own, never the register's: a page resolved
    /// against another register version then fails `compose`'s seal check
    /// instead of being read through a different slot mapping. `qr` only
    /// fixes the lifetime the rows are tied to.
    pub fn of(_qr: &'a Qr, page: &'a Page, users: usize) -> Self {
        assert!(users <= SLOTS, "a page holds at most {SLOTS} users");
        Members {
            witness: &page.witness,
            users,
            member: &page.member,
            roles: &page.roles,
            valid: None,
        }
    }

    /// The same rows read through a presence plane: a key whose `valid` bit
    /// is clear is NULL. The page's resolved roles no longer apply, so the
    /// support is taken from the counts.
    pub fn with_valid(mut self, valid: &'a [Row; SLOTS]) -> Self {
        self.valid = Some(valid);
        self
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
    /// The right-side cardinality witness: max carrier slots per role,
    /// counted once per bind.
    pub right_fan: u64,
}

impl<'a> Carriers<'a> {
    /// The register's role carriers, with the right fan counted once.
    pub fn of(qr: &'a Qr) -> Self {
        let right_fan = qr
            .role_slots
            .iter()
            .map(|slots| popcount_batch_u64(slots))
            .max()
            .unwrap_or(0);
        Carriers {
            witness: &qr.witness,
            role_slots: &qr.role_slots,
            role: &qr.role,
            right_fan,
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
    /// Popcount of `support`.
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

/// Set positions of a word, lowest first: the pointer table that inverts a
/// plane to its local positions.
fn ones(mut word: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (word != 0).then(|| {
            let p = word.trailing_zeros() as usize;
            word &= word - 1;
            p
        })
    })
}

/// The composition's executor. It holds no state; the register is the
/// prepared form.
#[derive(Default)]
pub struct Composer;

impl Composer {
    /// A composer.
    pub fn new() -> Self {
        Composer
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
        let mut counts = vec![0u16; users * 64];
        let mut support = vec![0u64; users];
        let (mut bag, mut left_fan) = (0u64, 0u64);
        for i in 0..users {
            // Present keys: the row itself, or the row under its presence plane.
            let mut present = left.member[i];
            if let Some(valid) = left.valid {
                mask_and_assign(&mut present, &valid[i]);
            }
            left_fan = left_fan.max(popcount_batch_u64(&present));
            // Distinct: the page's resolved roles; under a presence plane,
            // the union of the present slots' role words.
            support[i] = match left.valid {
                None => left.roles[i],
                Some(_) => {
                    let mut roles = [0u64];
                    for (w, &word) in present.iter().enumerate() {
                        for b in ones(word) {
                            mask_or_assign(
                                &mut roles,
                                std::slice::from_ref(&right.role[w * 64 + b]),
                            );
                        }
                    }
                    roles[0]
                }
            };
            // Bag: one AND + popcount per supported role.
            for r in ones(support[i]) {
                let mut path = present;
                mask_and_assign(&mut path, &right.role_slots[r]);
                let c = popcount_batch_u64(&path);
                counts[i * 64 + r] = u16::try_from(c).expect("a count over 256 slots fits u16");
                bag = add_bag(bag, c)?;
            }
        }
        let distinct = popcount_batch_u64(&support);
        let right_fan = right.right_fan;
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
