//! P9: which behavioural representation exists for a method.
//!
//! The five categories of the P9 brief. A method is put in the first one it
//! satisfies; categories 2-5 are never reported as solved.
//!
//! `ActionDef` has no field that names an `ogar-loco::Program` (it carries
//! `body_source: Option<String>`, verbatim text), so no `ActionDef` can be in
//! category 1 today. [`classify`] says so for every input rather than
//! guessing a link.

use ogar_vocab::ActionDef;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Coverage {
    /// 1. An OGAR Program body plus its action semantics.
    ProgramAndAction,
    /// 2. An ActionDef exists; no executable Program is linked.
    ActionNoProgram,
    /// 3. A native method behind an explicit adapter or extern contract.
    NativeAdapter,
    /// 4. Only source text or an extracted footprint.
    SourceOrFootprint,
    /// 5. No sufficient behavioural representation.
    Insufficient,
}

/// What an `ActionDef` carries beyond its identity, for the census detail.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Carries {
    pub kausal: bool,
    pub body_source: bool,
    pub reads: bool,
    pub writes: bool,
    pub calls: bool,
    pub raises: bool,
    pub on_enter: bool,
}

pub fn carries(a: &ActionDef) -> Carries {
    Carries {
        kausal: a.kausal.is_some(),
        body_source: a.body_source.is_some(),
        reads: !a.reads.is_empty(),
        writes: !a.writes.is_empty(),
        calls: !a.calls.is_empty(),
        raises: !a.raises.is_empty(),
        on_enter: a.on_enter.is_some(),
    }
}

/// The category of a method OGAR represents as an `ActionDef`.
pub fn classify(_a: &ActionDef) -> Coverage {
    // No ActionDef field can link a Program; see the module doc.
    Coverage::ActionNoProgram
}

/// Per-source tally.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tally {
    pub methods: usize,
    pub by_category: [usize; 5],
    pub with_kausal: usize,
    pub with_body_source: usize,
    pub with_writes: usize,
    pub with_reads: usize,
    pub with_on_enter: usize,
}

pub fn tally(actions: &[ActionDef]) -> Tally {
    let mut t = Tally {
        methods: actions.len(),
        ..Tally::default()
    };
    for a in actions {
        t.by_category[classify(a) as usize] += 1;
        let c = carries(a);
        t.with_kausal += usize::from(c.kausal);
        t.with_body_source += usize::from(c.body_source);
        t.with_writes += usize::from(c.writes);
        t.with_reads += usize::from(c.reads);
        t.with_on_enter += usize::from(c.on_enter);
    }
    t
}
