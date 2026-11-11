//! D-CG-BEH-2 (EXO-155): the ON x WHEN x DO contract of a registered DT64
//! method, read as separate facts, and a query that says which obligations a
//! change makes stale.
//!
//! Read-only. The crate registers nothing in a production store, adds no
//! polyfill word and executes no method. Its own bindings live in
//! [`fixture`] under a lab-local classid.
//!
//! # Three grades of evidence, never merged
//!
//! - [`Grade::SourceStructure`]: read from source text (which dispatch arm a
//!   line sits in). It says where code is, not what it computes.
//! - [`Grade::Dt64Registration`]: read from a `BindStore` through
//!   `method_ref`, `admit` and `resolve`. It says what is registered.
//! - [`Grade::ExecutedParity`]: an executed comparison against an oracle.
//!   This crate produces none; it only names where one is required.
//!
//! A code-to-opcode mapping is source structure. It is not a statement that
//! the code computes what the word means.
//!
//! # Fail closed
//!
//! [`facts::extract`] returns a [`facts::Missing`] when the store does not
//! admit the reference, the view does not give the bound basis, or a position
//! does not resolve. [`query::diff`] refuses facts that contradict the
//! binding digest. [`query::impl_change`] refuses a dispatch reading taken
//! from other source text. None of them answers "nothing changed" in place of
//! a missing witness.

#![forbid(unsafe_code)]

pub mod dispatch;
pub mod facts;
pub mod fixture;
pub mod query;

/// How a statement was obtained. See the crate docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Grade {
    SourceStructure,
    Dt64Registration,
    ExecutedParity,
}

impl Grade {
    pub fn name(self) -> &'static str {
        match self {
            Grade::SourceStructure => "SOURCE_STRUCTURE",
            Grade::Dt64Registration => "DT64_REGISTRATION",
            Grade::ExecutedParity => "EXECUTED_PARITY",
        }
    }
}

/// One hop of an evidence path, with the grade it was obtained at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evidence {
    pub grade: Grade,
    pub hop: String,
}

impl Evidence {
    pub(crate) fn registration(hop: impl Into<String>) -> Evidence {
        Evidence {
            grade: Grade::Dt64Registration,
            hop: hop.into(),
        }
    }
    pub(crate) fn source(hop: impl Into<String>) -> Evidence {
        Evidence {
            grade: Grade::SourceStructure,
            hop: hop.into(),
        }
    }
}

/// FNV-1a 64 over `bytes`. A change detector for this crate's own snapshots;
/// it is not the store's basis or binding digest and authenticates nothing.
pub(crate) fn fnv64(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub(crate) const FNV64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
