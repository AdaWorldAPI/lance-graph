//! D-SAG-6-1 fixture: closures, static and dynamic trait dispatch, a capturing
//! closure, a predicate that can panic, a mask popcount, two same-named
//! functions, a macro-generated function and a derived impl.

pub trait Pred {
    fn test(&self, x: u32) -> bool;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Below(pub u32);
pub struct Even;

impl Pred for Below {
    fn test(&self, x: u32) -> bool {
        x < self.0
    }
}
impl Pred for Even {
    fn test(&self, x: u32) -> bool {
        x % 2 == 0
    }
}

/// Pure filter-count with a non-capturing-by-mut closure: the D-SAG-6 fixture shape.
pub fn count_below(xs: &[u32], t: u32) -> usize {
    xs.iter().filter(|&&x| x < t).count()
}

/// Static dispatch through a generic bound.
pub fn count_generic<P: Pred>(xs: &[u32], p: &P) -> usize {
    xs.iter().filter(|&&x| p.test(x)).count()
}

/// Dynamic dispatch: the callee is not known statically.
pub fn count_dyn(xs: &[u32], p: &dyn Pred) -> usize {
    xs.iter().filter(|&&x| p.test(x)).count()
}

/// The closure captures mutable state: a fusion to popcount would drop the side effect (F02).
pub fn count_with_tally(xs: &[u32], t: u32, seen: &mut u64) -> usize {
    xs.iter()
        .filter(|&&x| {
            *seen += 1;
            x < t
        })
        .count()
}

/// The predicate can panic (division by zero) (F03).
pub fn count_div(xs: &[u32], d: u32) -> usize {
    xs.iter().filter(|&&x| x / d > 1).count()
}

/// Mask form: popcount over u64 words.
pub fn popcount_mask(words: &[u64]) -> usize {
    words.iter().map(|w| w.count_ones() as usize).sum()
}

/// Builds the mask `count_below` would count.
pub fn mask_below(xs: &[u32], t: u32) -> Vec<u64> {
    let mut m = vec![0u64; xs.len().div_ceil(64)];
    for (i, &x) in xs.iter().enumerate() {
        if x < t {
            m[i / 64] |= 1 << (i % 64);
        }
    }
    m
}

pub mod a {
    pub fn helper(x: u32) -> u32 {
        x + 1
    }
}
pub mod b {
    pub fn helper(x: u32) -> u32 {
        x * 2
    }
}

pub fn use_helpers(x: u32) -> u32 {
    a::helper(x) + b::helper(x)
}

macro_rules! make_twice {
    ($name:ident) => {
        pub fn $name(x: u32) -> u32 {
            x * 2
        }
    };
}
make_twice!(twice);

pub fn call_twice(x: u32) -> u32 {
    twice(x)
}
