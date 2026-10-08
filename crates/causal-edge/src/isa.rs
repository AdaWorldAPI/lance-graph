//! The CE64 instruction set.
//!
//! `CausalEdge64` is the register; this module is the decoder and the
//! semantics. Three rules hold here and nowhere else:
//!
//! 1. **The decoder is strict.** The 4-bit inference field holds 16 values,
//!    but only five of them name an instruction this crate implements
//!    ([`Opcode::decode`]). Every other value is [`IsaFault::Unsupported`]:
//!    it is never executed as some other instruction and re-stamped with its
//!    own code afterwards. In particular Counterfactual (`-6`),
//!    Intervention (`+6`) and the reserved `±7` fault until each has an
//!    implementation of its own.
//! 2. **One truth function per instruction** ([`truth`]). `forward`, `learn`
//!    and `syllogize` all call these; none carries its own formula.
//! 3. **Every instruction declares its field contract** ([`Contract`]): which
//!    operand fields it reads, which output fields it computes, which it
//!    passes through unchanged from an operand, and which it overwrites with a
//!    constant. `tests/ce64_isa_contract.rs` measures each declaration.
//!
//! ## Total and partial instructions
//!
//! - **Total**: every valid operand pair has a defined result. The five
//!   [`Opcode`]s are total; their register methods return `Self`.
//!   `deduction`, `induction`, `abduction` and `synthesis` compose the payload
//!   through a [`Compose`] algebra; `revision` is truth-only and needs none.
//! - **Partial**: the instruction has preconditions; a violation returns an
//!   [`IsaFault`]. `CausalEdge64::counterfactual` and `::intervention` are
//!   partial with a precondition no operand satisfies yet (no implementation),
//!   so they always fault. They exist so the slot is visible and so a caller
//!   that names them gets a refusal, never another instruction.
//!
//! ## Layers
//!
//! - **Instruction** (this module and the register methods): truth arithmetic
//!   and field transitions.
//! - **[`Compose`]**: the payload algebra for S/P/O. It never selects an
//!   opcode or sees any reasoning state.
//! - **Reasoning** (recipes, planner, revision policies such as Gadamer's):
//!   chooses operands, instruction, order, admissibility and when to stop.
//!   It lives above this crate.
//!
//! The register space is larger than the decoder on purpose. A value written
//! by a producer the ISA does not execute (a counterfactual tag, a lens
//! reading) stays representable; it just cannot be executed.

use crate::edge::InferenceType;

/// An executable CE64 instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    /// `A->B, B->C ⊢ A->C`.
    Deduction,
    /// `A->B, A->C ⊢ B->C`.
    Induction,
    /// `A->B, C->B ⊢ A->C`.
    Abduction,
    /// Pool two bodies of evidence for the same statement.
    Revision,
    /// Arithmetic mean of frequency and of confidence.
    Synthesis,
}

/// The payload algebra an instruction composes S, P and O through: one
/// 256 x 256 table per plane, indexed `lhs * 256 + rhs`.
///
/// The ISA does not interpret the payload bytes itself; the caller declares
/// what they mean by the tables it passes.
#[derive(Debug, Clone, Copy)]
pub struct Compose<'a> {
    /// S-plane composition.
    pub s: &'a [u8; 256 * 256],
    /// P-plane composition.
    pub p: &'a [u8; 256 * 256],
    /// O-plane composition.
    pub o: &'a [u8; 256 * 256],
}

/// A register value the ISA does not execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsaFault {
    /// The inference field holds a code with no implementation here.
    Unsupported {
        /// The signed 4-bit value as stored (v1 layout: the 3-bit code).
        mantissa: i8,
    },
}

impl core::fmt::Display for IsaFault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IsaFault::Unsupported { mantissa } => {
                write!(
                    f,
                    "CE64 ISA: inference code {mantissa} has no implementation"
                )
            }
        }
    }
}

impl std::error::Error for IsaFault {}

impl Opcode {
    /// Every executable instruction.
    pub const ALL: [Opcode; 5] = [
        Opcode::Deduction,
        Opcode::Induction,
        Opcode::Abduction,
        Opcode::Revision,
        Opcode::Synthesis,
    ];

    /// The signed v2 inference code this instruction is stored as.
    #[inline]
    pub const fn encoding(self) -> i8 {
        match self {
            Opcode::Deduction => 1,
            Opcode::Induction => 2,
            Opcode::Abduction => -1,
            Opcode::Revision => 4,
            Opcode::Synthesis => 5,
        }
    }

    /// Strict v2 decode: exactly the five encodings above execute.
    ///
    /// `0` (neutral), `-8`, `-2` (contraposition), `±3` (exemplification /
    /// negative analogy), `-4`, `-5` (decomposition), `±6` (intervention /
    /// counterfactual) and `±7` have no implementation and fault.
    /// [`InferenceType::from_mantissa`] still maps all 16 values to its
    /// nearest variant; that is a reading for display, not a decoder.
    #[inline]
    pub const fn decode(mantissa: i8) -> Result<Self, IsaFault> {
        match mantissa {
            1 => Ok(Opcode::Deduction),
            2 => Ok(Opcode::Induction),
            -1 => Ok(Opcode::Abduction),
            4 => Ok(Opcode::Revision),
            5 => Ok(Opcode::Synthesis),
            m => Err(IsaFault::Unsupported { mantissa: m }),
        }
    }

    /// Strict decode of the v1 3-bit code (Deduction..Synthesis are 0..4).
    #[inline]
    pub const fn decode_v1(code: InferenceType) -> Result<Self, IsaFault> {
        match code {
            InferenceType::Deduction => Ok(Opcode::Deduction),
            InferenceType::Induction => Ok(Opcode::Induction),
            InferenceType::Abduction => Ok(Opcode::Abduction),
            InferenceType::Revision => Ok(Opcode::Revision),
            InferenceType::Synthesis => Ok(Opcode::Synthesis),
            other => Err(IsaFault::Unsupported {
                mantissa: other as i8,
            }),
        }
    }

    /// The `InferenceType` this instruction is.
    #[inline]
    pub const fn inference_type(self) -> InferenceType {
        match self {
            Opcode::Deduction => InferenceType::Deduction,
            Opcode::Induction => InferenceType::Induction,
            Opcode::Abduction => InferenceType::Abduction,
            Opcode::Revision => InferenceType::Revision,
            Opcode::Synthesis => InferenceType::Synthesis,
        }
    }

    /// The canonical truth function of this instruction, on `(f, c)` pairs.
    ///
    /// Revision with no evidence on either side returns `(0.5, 0.0)`
    /// (unknown); [`truth::revision`] reports that case as `None` for callers
    /// that must leave their state untouched instead.
    #[inline]
    pub fn truth(self, f1: f32, c1: f32, f2: f32, c2: f32) -> (f32, f32) {
        match self {
            Opcode::Deduction => truth::deduction(f1, c1, f2, c2),
            Opcode::Induction => truth::induction(f1, c1, f2, c2),
            Opcode::Abduction => truth::abduction(f1, c1, f2, c2),
            Opcode::Revision => truth::revision(f1, c1, f2, c2).unwrap_or((0.5, 0.0)),
            Opcode::Synthesis => truth::synthesis(f1, c1, f2, c2),
        }
    }
}

/// The canonical NARS truth functions (evidence horizon `k = 1`).
///
/// These are the only formulas in this crate. The arithmetic and its order
/// are exactly what `forward`, `learn` and `syllogize` computed inline before
/// they delegated here, so every packed result is bit-identical.
pub mod truth {
    /// Evidence weight `w = c / (1 - c)`, saturating to `f32::MAX` at
    /// `c >= 0.999`.
    ///
    /// Known defect, pinned in `tests/ce64_isa_golden.rs`: when BOTH operands
    /// of a revision saturate, the pooled weight overflows to infinity and the
    /// result is `NaN`, which packs as `f = c = 0`.
    #[inline]
    pub fn evidence_weight(c: f32) -> f32 {
        if c >= 0.999 {
            f32::MAX
        } else {
            c / (1.0 - c)
        }
    }

    /// Deduction: `f = f1·f2`, `c = f1·f2·c1·c2`.
    #[inline]
    pub fn deduction(f1: f32, c1: f32, f2: f32, c2: f32) -> (f32, f32) {
        let f = f1 * f2;
        (f, f * c1 * c2)
    }

    /// Induction: `f = f2`, `c = w/(w+1)` with `w = f1·c1·c2`.
    #[inline]
    pub fn induction(f1: f32, c1: f32, f2: f32, c2: f32) -> (f32, f32) {
        let w = f1 * c1 * c2;
        (f2, w / (w + 1.0))
    }

    /// Abduction: `f = f1`, `c = w/(w+1)` with `w = f2·c1·c2`.
    #[inline]
    pub fn abduction(f1: f32, c1: f32, f2: f32, c2: f32) -> (f32, f32) {
        let w = f2 * c1 * c2;
        (f1, w / (w + 1.0))
    }

    /// Revision: pool the evidence weights. `None` when neither side carries
    /// evidence.
    ///
    /// The threshold is `ws > f32::EPSILON`. For confidences decoded from a
    /// `u8` the smallest non-zero weight is `(1/255)/(254/255) ≈ 0.0039`, so
    /// `ws` is either exactly 0 or far above the threshold.
    #[inline]
    pub fn revision(f1: f32, c1: f32, f2: f32, c2: f32) -> Option<(f32, f32)> {
        let w1 = evidence_weight(c1);
        let w2 = evidence_weight(c2);
        let ws = w1 + w2;
        if ws > f32::EPSILON {
            Some(((f1 * w1 + f2 * w2) / ws, ws / (ws + 1.0)))
        } else {
            None
        }
    }

    /// Synthesis: the mean of each component. Not a NARS rule.
    #[inline]
    pub fn synthesis(f1: f32, c1: f32, f2: f32, c2: f32) -> (f32, f32) {
        ((f1 + f2) / 2.0, (c1 + c2) / 2.0)
    }
}

/// A CE64 field, v2 layout.
#[cfg(feature = "causal-edge-v2-layout")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Field {
    /// Bits 0..23: the S, P, O payload.
    Spo,
    /// Bits 24..31.
    Frequency,
    /// Bits 32..39.
    Confidence,
    /// Bits 40..42.
    Pearl,
    /// Bits 43..45: the S/P/O sign triple.
    Direction,
    /// Bits 46..49: the inference code.
    Inference,
    /// Bits 50..52.
    Plasticity,
    /// Bits 53..58.
    Witness,
    /// Bits 59..63.
    Epistemic,
}

#[cfg(feature = "causal-edge-v2-layout")]
impl Field {
    /// Every field, in bit order. Together they cover all 64 bits once.
    pub const ALL: [Field; 9] = [
        Field::Spo,
        Field::Frequency,
        Field::Confidence,
        Field::Pearl,
        Field::Direction,
        Field::Inference,
        Field::Plasticity,
        Field::Witness,
        Field::Epistemic,
    ];

    /// `(shift, width)`.
    pub const fn span(self) -> (u32, u32) {
        match self {
            Field::Spo => (0, 24),
            Field::Frequency => (24, 8),
            Field::Confidence => (32, 8),
            Field::Pearl => (40, 3),
            Field::Direction => (43, 3),
            Field::Inference => (46, 4),
            Field::Plasticity => (50, 3),
            Field::Witness => (53, 6),
            Field::Epistemic => (59, 5),
        }
    }

    /// The field's bits in place.
    pub const fn mask(self) -> u64 {
        let (s, w) = self.span();
        ((1u64 << w) - 1) << s
    }

    /// The field's value.
    pub const fn get(self, word: u64) -> u64 {
        (word & self.mask()) >> self.span().0
    }
}

/// Which operand a field comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operand {
    /// The receiver: `forward`'s running edge, `learn`'s self, `syllogize`'s
    /// first premise.
    A,
    /// The argument: `forward`'s weight, `learn`'s observation, `syllogize`'s
    /// second premise.
    B,
}

/// What one instruction does to each field.
///
/// Every output field appears in exactly one of `computes`, `passes` and
/// `constants`.
#[cfg(feature = "causal-edge-v2-layout")]
#[derive(Debug, Clone, Copy)]
pub struct Contract {
    /// The instruction.
    pub name: &'static str,
    /// Operand fields that influence an output other than by passing through.
    pub reads: &'static [(Operand, Field)],
    /// Output fields computed from the reads.
    pub computes: &'static [Field],
    /// Output fields copied unchanged from one operand.
    pub passes: &'static [(Field, Operand)],
    /// Output fields overwritten with a constant, whatever the operands hold.
    pub constants: &'static [(Field, u64)],
    /// Whether the instruction composes S/P/O through a [`Compose`] algebra.
    pub needs_compose: bool,
    /// When the instruction refuses.
    pub fails: &'static str,
}

#[cfg(feature = "causal-edge-v2-layout")]
pub mod contracts {
    //! The declared contracts, one per instruction.
    use super::{Contract, Field::*, Operand::*};

    /// `a.forward(b)`. The instruction comes from `b`'s inference field;
    /// `a`'s is ignored. The output's inference field is `b`'s, unchanged.
    pub const FORWARD: Contract = Contract {
        name: "forward",
        reads: &[
            (A, Spo),
            (A, Frequency),
            (A, Confidence),
            (A, Pearl),
            (B, Spo),
            (B, Frequency),
            (B, Confidence),
            (B, Pearl),
            (B, Inference),
        ],
        computes: &[Spo, Frequency, Confidence, Pearl],
        passes: &[(Inference, B), (Direction, B), (Plasticity, B)],
        constants: &[(Witness, 0), (Epistemic, 0)],
        needs_compose: true,
        fails: "B's inference code has no implementation (Opcode::decode)",
    };

    /// `a.learn(b)`: revision of `a` by the observation `b`.
    pub const LEARN: Contract = Contract {
        name: "learn",
        reads: &[
            (A, Spo),
            (A, Frequency),
            (A, Confidence),
            (A, Plasticity),
            (B, Spo),
            (B, Frequency),
            (B, Confidence),
        ],
        computes: &[Spo, Frequency, Confidence, Plasticity],
        passes: &[
            (Pearl, A),
            (Direction, A),
            (Inference, A),
            (Witness, A),
            (Epistemic, A),
        ],
        constants: &[],
        needs_compose: false,
        fails: "never; no evidence on either side leaves A unchanged",
    };

    /// `a.syllogize(b)`: a fresh conclusion; the rule follows from the figure.
    pub const SYLLOGIZE: Contract = Contract {
        name: "syllogize",
        reads: &[
            (A, Spo),
            (A, Frequency),
            (A, Confidence),
            (A, Pearl),
            (B, Spo),
            (B, Frequency),
            (B, Confidence),
            (B, Pearl),
        ],
        computes: &[Spo, Frequency, Confidence, Pearl, Inference],
        passes: &[],
        constants: &[
            (Direction, 0),
            (Plasticity, 0b111),
            (Witness, 0),
            (Epistemic, 0),
        ],
        needs_compose: false,
        fails: "None (not an error) when the two edges share no term or are the same statement",
    };

    /// `a.revision(b)`: pure truth revision of the same statement.
    pub const REVISION: Contract = Contract {
        name: "revision",
        reads: &[
            (A, Frequency),
            (A, Confidence),
            (B, Frequency),
            (B, Confidence),
        ],
        computes: &[Frequency, Confidence],
        passes: &[
            (Spo, A),
            (Pearl, A),
            (Direction, A),
            (Inference, A),
            (Plasticity, A),
            (Witness, A),
            (Epistemic, A),
        ],
        constants: &[],
        needs_compose: false,
        fails: "never; no evidence on either side gives (0.5, 0.0)",
    };

    /// All declared contracts.
    pub const ALL: [Contract; 4] = [FORWARD, LEARN, SYLLOGIZE, REVISION];
}
