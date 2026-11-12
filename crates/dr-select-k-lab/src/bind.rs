//! The one binding of a selection, and the views of a residual over it.

use crate::counters::Counters;
use crate::source::{Selection, Source, Staged, Version};

/// The order the local rows `0..K` are in. It is a witness: an operation
/// that needs key order may rely on it only when it says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Source row order. Says nothing about keys.
    Source,
    /// Key ascending, ties by guid.
    KeyThenGuid,
}

/// Why a binding was not made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindRefusal {
    /// The selection was produced at another version of the source.
    NotWitnessedForVersion { selection: Version, source: Version },
}

/// K local rows `0..K`, bound once. `rows` is the one local-to-source
/// mapping (the IndexVector); no second copy of it exists.
pub struct Bound {
    version: Version,
    order: Order,
    rows: Vec<u32>,
    staged: Staged,
}

impl Bound {
    /// Bind `selection` into local coordinates and stage its K rows. The
    /// only place after selection where the source is touched.
    pub fn bind(
        source: &Source,
        selection: Selection,
        counters: &Counters,
    ) -> Result<Bound, BindRefusal> {
        if selection.version != source.version() {
            return Err(BindRefusal::NotWitnessedForVersion {
                selection: selection.version,
                source: source.version(),
            });
        }
        let staged = source.take(&selection.rows, counters);
        Counters::bump(&counters.binds, 1);
        Ok(Bound {
            version: selection.version,
            order: selection.order,
            rows: selection.rows,
            staged,
        })
    }

    pub fn k(&self) -> usize {
        self.rows.len()
    }

    pub fn version(&self) -> Version {
        self.version
    }

    pub fn order(&self) -> Order {
        self.order
    }

    /// Local row -> source row. Borrowed; the IndexVector.
    pub fn rows(&self) -> &[u32] {
        &self.rows
    }

    /// The staged K-local lanes.
    pub fn staged(&self) -> &Staged {
        &self.staged
    }
}

/// A residual selection over the K local rows, in one physical form. Built
/// once per residual predicate; the form is what the lab compares.
pub enum Residual {
    /// Every local row. Nothing is stored.
    All,
    /// Local positions, ascending. So their order is the local order.
    Positions(Vec<u32>),
    /// One bit per local row, `ceil(K / 64)` words, tail bits zero.
    Mask(Vec<u64>),
}

/// A borrowed view of a [`Residual`].
#[derive(Debug, Clone, Copy)]
pub enum View<'r> {
    All,
    Positions(&'r [u32]),
    Mask(&'r [u64]),
}

impl Residual {
    pub fn view(&self) -> View<'_> {
        match self {
            Residual::All => View::All,
            Residual::Positions(p) => View::Positions(p),
            Residual::Mask(m) => View::Mask(m),
        }
    }

    /// Bytes this form retains.
    pub fn retained_bytes(&self) -> usize {
        match self {
            Residual::All => 0,
            Residual::Positions(p) => p.len() * 4,
            Residual::Mask(m) => m.len() * 8,
        }
    }
}
