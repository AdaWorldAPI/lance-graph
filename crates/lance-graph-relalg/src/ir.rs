//! The IR. It describes a selection over one table's row space; it never
//! holds rows.

/// An ordered comparison of an `I32` lane against a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Cmp {
    /// `lane < t`
    Lt,
    /// `lane <= t`
    Le,
    /// `lane > t`
    Gt,
    /// `lane >= t`
    Ge,
    /// `lane == t`
    Eq,
    /// `lane != t`
    Ne,
}

/// One predicate on one lane of the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Atom {
    /// `lanes[lane] op t` on an `I32` lane.
    I32 { lane: u16, op: Cmp, t: i32 },
    /// `lanes[lane] == v` (`eq`) or `!= v` (`!eq`) on a `U32` lane.
    U32 { lane: u16, eq: bool, v: u32 },
}

/// A set of rows of the table, described, not computed.
///
/// `And` is intersection (σ of a conjunction, ∩), `Or` is union (∪), and
/// `And[a, Not b]` is difference (−). Projection and aggregation are the
/// [`Measure`] a [`Query`] ends with.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rel {
    /// Every row.
    All,
    /// No row.
    Empty,
    /// A resident mask plane, `Planes::masks[i]`.
    Plane(u16),
    /// The rows where one predicate holds.
    Atom(Atom),
    /// Intersection.
    And(Vec<Rel>),
    /// Union.
    Or(Vec<Rel>),
    /// Complement within the table.
    Not(Box<Rel>),
}

impl Rel {
    /// `a − b`.
    pub fn diff(a: Rel, b: Rel) -> Rel {
        Rel::And(vec![a, Rel::Not(Box::new(b))])
    }

    /// `lanes[lane] op t` on an `I32` lane.
    pub fn i32(lane: u16, op: Cmp, t: i32) -> Rel {
        Rel::Atom(Atom::I32 { lane, op, t })
    }

    /// `lanes[lane] == v` on a `U32` lane.
    pub fn eq_u32(lane: u16, v: u32) -> Rel {
        Rel::Atom(Atom::U32 { lane, eq: true, v })
    }

    /// `lanes[lane] != v` on a `U32` lane.
    pub fn ne_u32(lane: u16, v: u32) -> Rel {
        Rel::Atom(Atom::U32 { lane, eq: false, v })
    }

    /// How many predicates the description names (a lowering emits one
    /// predicate pass per atom).
    pub fn atoms(&self) -> usize {
        match self {
            Rel::Atom(_) => 1,
            Rel::And(cs) | Rel::Or(cs) => cs.iter().map(Rel::atoms).sum(),
            Rel::Not(c) => c.atoms(),
            _ => 0,
        }
    }
}

/// What a query produces from its rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Measure {
    /// How many rows.
    Count,
    /// Σ of an `I32` lane.
    Sum(u16),
    /// Minimum of an `I32` lane.
    Min(u16),
    /// Maximum of an `I32` lane.
    Max(u16),
    /// The rows themselves, written into a caller-owned mask.
    Keep,
}

/// A selection and what to produce from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Query {
    /// Which rows.
    pub rel: Rel,
    /// What to produce.
    pub measure: Measure,
}
