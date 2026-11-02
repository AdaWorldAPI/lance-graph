//! D-DRCA-3: DR #4, method routing over one admitted register (lab only).
//!
//! DR #4 is the route from a mapped coordinate set to a method operation
//! (D-DRDT §5, lance-graph #1835). DR owns the coordinates and the fold; DT
//! owns what a method means. So every DT-owned value here is **opaque**:
//!
//! - [`MethodKey`]: the method's identity. Minted by the DT owner (D-CMH
//!   §2.1); this lab never mints one and never derives one from a name.
//! - [`GuardWitness`]: evidence that DT's guard was evaluated. DR compares it
//!   for equality and never evaluates a guard.
//! - [`EffectOrderRef`]: DT's reference to the method's ordered effects. DR
//!   never runs, reorders or compares effects by meaning.
//!
//! # What routing preserves
//!
//! One [`RoutedMethod`] per input method, in input order: the key, the
//! signature revision, the read and write field sets, the guard witness and
//! the effect-order reference. Two methods with equal read/write sets and a
//! different effect order are two entries. So are two methods whose effects
//! commute: whether they may be merged is decided by the law owner, not by
//! routing.
//!
//! # What routing refuses
//!
//! In order: a stale register frame, a resolution made under another frame
//! issue, a foreign class, a duplicate key, coordinates outside 64 × 64, an
//! algebra DR does not have, a presence read over a three-valued register,
//! and a guard that was declared but not witnessed (or witnessed differently).
//! See [`Dr4Refusal`].
//!
//! # What this module does not do
//!
//! It is not a DT resolver (that is D-CMH-1's `PreparedClass`), it stores
//! nothing, and it executes no effect. [`Resolution`] is DT's answer, handed
//! in by the caller.

use lance_graph_contract::class_view::ClassId;

use crate::hhtl16;
use crate::{BoundDr64, Frame, FrameStore, Population, Refusal, Truth};

/// A method's identity, minted by the DT owner. Opaque to DR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MethodKey(pub [u8; 16]);

/// Evidence that a method's guard was evaluated. Opaque to DR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GuardWitness(pub [u8; 16]);

/// DT's reference to a method's ordered effects. Opaque to DR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EffectOrderRef(pub [u8; 16]);

/// The algebra a method declares for its operation over the register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Algebra {
    /// Boolean presence over a two-valued, distinct register.
    Presence,
    /// Value and validity, three-valued; a two-valued register is known
    /// everywhere.
    PresenceK3,
    /// A numeric fold. DR has no admitted numeric fold over DR64, so this
    /// refuses.
    NumericFold,
}

/// Whether DT declared a guard for the method, and which.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GuardDecl {
    Unguarded,
    Required(GuardWitness),
}

/// One method as DT resolved it, as DR receives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MethodRef {
    pub class: ClassId,
    pub key: MethodKey,
    pub signature_rev: u32,
    /// Field positions read and written (a `FieldMask`-width lab view).
    pub reads: u64,
    pub writes: u64,
    pub guard: GuardDecl,
    pub effects: EffectOrderRef,
    pub algebra: Algebra,
}

/// DT's answer for one class, and the frame it was resolved under.
#[derive(Clone, Debug)]
pub struct Resolution {
    pub class: ClassId,
    pub under: Frame,
    pub methods: Vec<MethodRef>,
}

/// The operation routed over the register's coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Cell { x: u8, y: u8 },
    Row { x: u8 },
    Column { y: u8 },
}

/// What one method's algebra produced at the routed coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Bool(bool),
    Mask(u64),
    /// `value` is masked by `known`: an UNKNOWN cell never reads TRUE.
    K3 {
        value: u64,
        known: u64,
    },
}

/// One routed method: its DT references, unchanged, and its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutedMethod {
    pub key: MethodKey,
    pub signature_rev: u32,
    pub reads: u64,
    pub writes: u64,
    /// The witness the guard was checked against; `None` when unguarded.
    pub guard: Option<GuardWitness>,
    pub effects: EffectOrderRef,
    pub value: Value,
}

/// The routing result. No public constructor.
#[derive(Clone, Debug)]
pub struct Routed {
    frame: Frame,
    op: Op,
    entries: Vec<RoutedMethod>,
}

impl Routed {
    /// The frame the result was computed under.
    pub fn frame(&self) -> &Frame {
        &self.frame
    }
    pub fn op(&self) -> Op {
        self.op
    }
    /// One entry per input method, in input order.
    pub fn entries(&self) -> &[RoutedMethod] {
        &self.entries
    }
}

/// Why a route was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dr4Refusal {
    /// The register's frame is not its store's current seal.
    Frame(Refusal),
    /// DT resolved the methods under another frame issue.
    ResolvedUnderOtherFrame,
    /// The resolution, or one of its methods, is for another class.
    ForeignClass,
    /// One method key appears twice.
    DuplicateMethod,
    /// Coordinates outside 64 × 64.
    OutOfRegister,
    /// The method declares an algebra DR does not have.
    UnsupportedAlgebra,
    /// A presence read over a three-valued or bag register.
    NotBoolean,
    /// A declared guard has no witness.
    GuardDropped,
    /// A declared guard was witnessed with another value.
    GuardMismatch,
}

/// Route DT's methods for one class over one admitted register.
///
/// `observed` holds the guard witnesses the caller obtained, keyed by method.
pub fn route(
    store: &FrameStore,
    register: &BoundDr64<'_>,
    resolution: &Resolution,
    op: Op,
    observed: &[(MethodKey, GuardWitness)],
) -> Result<Routed, Dr4Refusal> {
    store.check(&register.frame).map_err(Dr4Refusal::Frame)?;
    if !resolution.under.same_issue(&register.frame) {
        return Err(Dr4Refusal::ResolvedUnderOtherFrame);
    }
    let class = register.method.owner_class;
    if resolution.class != class || resolution.methods.iter().any(|m| m.class != class) {
        return Err(Dr4Refusal::ForeignClass);
    }
    for (i, m) in resolution.methods.iter().enumerate() {
        if resolution.methods[..i].iter().any(|p| p.key == m.key) {
            return Err(Dr4Refusal::DuplicateMethod);
        }
    }
    let in_range = match op {
        Op::Cell { x, y } => x < 64 && y < 64,
        Op::Row { x } => x < 64,
        Op::Column { y } => y < 64,
    };
    if !in_range {
        return Err(Dr4Refusal::OutOfRegister);
    }
    let boolean = register.method.truth == Truth::TwoValued
        && register.method.population == Population::Distinct;

    let mut entries = Vec::with_capacity(resolution.methods.len());
    for m in &resolution.methods {
        let value = match m.algebra {
            Algebra::NumericFold => return Err(Dr4Refusal::UnsupportedAlgebra),
            Algebra::Presence if !boolean => return Err(Dr4Refusal::NotBoolean),
            Algebra::Presence => presence(register, op),
            Algebra::PresenceK3 => presence_k3(register, op),
        };
        let guard = match m.guard {
            GuardDecl::Unguarded => None,
            GuardDecl::Required(want) => {
                let got = observed
                    .iter()
                    .find(|(k, _)| *k == m.key)
                    .map(|(_, w)| *w)
                    .ok_or(Dr4Refusal::GuardDropped)?;
                if got != want {
                    return Err(Dr4Refusal::GuardMismatch);
                }
                Some(got)
            }
        };
        entries.push(RoutedMethod {
            key: m.key,
            signature_rev: m.signature_rev,
            reads: m.reads,
            writes: m.writes,
            guard,
            effects: m.effects,
            value,
        });
    }
    Ok(Routed {
        frame: register.frame.clone(),
        op,
        entries,
    })
}

/// Boolean read; coordinates are already in range.
fn presence(r: &BoundDr64<'_>, op: Op) -> Value {
    match op {
        Op::Cell { x, y } => Value::Bool((r.rows[x as usize] >> y) & 1 == 1),
        Op::Row { x } => Value::Mask(r.rows[x as usize]),
        Op::Column { y } => Value::Mask(hhtl16::column(r.rows, y).unwrap_or(0)),
    }
}

/// Value and validity; a two-valued register is known everywhere.
fn presence_k3(r: &BoundDr64<'_>, op: Op) -> Value {
    let all = [u64::MAX; 64];
    let validity = r.validity.unwrap_or(&all);
    let (value, known) = match op {
        Op::Cell { x, y } => {
            let k = (validity[x as usize] >> y) & 1;
            ((r.rows[x as usize] >> y) & k, k)
        }
        Op::Row { x } => (r.rows[x as usize], validity[x as usize]),
        Op::Column { y } => (
            hhtl16::column(r.rows, y).unwrap_or(0),
            hhtl16::column(validity, y).unwrap_or(0),
        ),
    };
    Value::K3 {
        value: value & known,
        known,
    }
}
