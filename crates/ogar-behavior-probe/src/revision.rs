//! P2: a deterministic revision for an `ogar-loco::Program`.
//!
//! Two digests, both FNV-1a 64 (the same function `dr::LocalMap::digest`
//! uses), both lab-local:
//!
//! - [`raw_body_rev`] reads the functions in stored index order.
//! - [`body_rev`] first renumbers the functions reachable from the entry in
//!   first-reference order, then hashes. Two programs that differ only in
//!   which index a lowerer assigned to a body get the same revision; a
//!   function no path reaches is not part of the revision.
//!
//! What the digest covers, explicitly:
//! - the vocabulary tag ([`VocabTag`]). `ogar-loco::Vocabulary` has no name
//!   and no version (vocabulary.rs, trait `Vocabulary`), so the caller must
//!   supply one. That is a gap, recorded in the README;
//! - per function: the lane shape, the call count, and every written call
//!   (function byte plus the three value bytes). Bytes past `len` are never
//!   read, because only [`FunctionBody::calls`] is read.
//!
//! What it does NOT cover: anything on the `ActionDef` envelope
//! (`kausal`, `reads`, `writes`, `calls`, `raises`, `on_enter`, the guard
//! failure policy, the timeout). See [`envelope_rev`] and the P2 tests.

use ogar_loco::{branches_of, Call, CheckedVocabulary, FnIndex, FunctionBody, Program, Vocabulary};
use ogar_vocab::ActionDef;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

struct Fnv(u64);

impl Fnv {
    fn new(domain: &[u8]) -> Fnv {
        let mut h = Fnv(FNV_OFFSET);
        h.bytes(domain);
        h
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= u64::from(x);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }
    /// Length-prefixed, so `["ab","c"]` and `["a","bc"]` differ.
    fn str(&mut self, s: &str) {
        self.bytes(&(s.len() as u64).to_le_bytes());
        self.bytes(s.as_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
}

/// The interpretation a program's bytes are read under. Supplied by the
/// caller because the vocabulary trait carries no identity of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocabTag {
    pub name: &'static str,
    pub rev: u32,
}

/// Why a canonical revision could not be computed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RevError {
    /// A body reference names no function, or names the entry
    /// (`Program::references_are_resolvable` is false).
    UnresolvableReference,
    /// More than 256 reachable functions: an index no longer fits a byte.
    TooManyFunctions,
}

fn hash_body(h: &mut Fnv, body: &FunctionBody, calls: &[Call]) {
    h.str(&format!("{:?}", body.shape()));
    h.u64(calls.len() as u64);
    for c in calls {
        h.bytes(&[c.function.0]);
        h.bytes(&c.values);
    }
}

/// The revision of `program` read in stored index order.
pub fn raw_body_rev(program: &Program, tag: VocabTag) -> u64 {
    let mut h = Fnv::new(b"ogar-behavior-probe/raw-body/1");
    h.str(tag.name);
    h.u64(u64::from(tag.rev));
    h.u64(program.functions.len() as u64);
    for body in &program.functions {
        let calls: Vec<Call> = body.calls().collect();
        hash_body(&mut h, body, &calls);
    }
    h.0
}

/// The revision of `program` after renumbering the reachable functions in
/// first-reference order from the entry. Body-reference bytes are rewritten
/// to the new numbering before hashing; every other byte is hashed as stored.
pub fn body_rev<V: Vocabulary>(
    program: &Program,
    vocab: &CheckedVocabulary<V>,
    tag: VocabTag,
) -> Result<u64, RevError> {
    if program.is_empty() || !program.references_are_resolvable(vocab) {
        return Err(RevError::UnresolvableReference);
    }
    // old index -> new index, assigned in visit order.
    let mut new_of: Vec<Option<u8>> = vec![None; program.len()];
    let mut order: Vec<usize> = vec![0];
    new_of[0] = Some(0);
    let mut i = 0;
    while i < order.len() {
        let old = order[i];
        for (_, _, targets) in branches_of(vocab, &program.functions[old]) {
            for t in targets {
                let t = usize::from(t);
                if new_of[t].is_none() {
                    let next = u8::try_from(order.len()).map_err(|_| RevError::TooManyFunctions)?;
                    new_of[t] = Some(next);
                    order.push(t);
                }
            }
        }
        i += 1;
    }
    let mut h = Fnv::new(b"ogar-behavior-probe/body/1");
    h.str(tag.name);
    h.u64(u64::from(tag.rev));
    h.u64(order.len() as u64);
    for &old in &order {
        let body = &program.functions[old];
        let calls: Vec<Call> = body
            .calls()
            .map(|mut c| {
                let refs = usize::from(vocab.body_refs(c.function));
                for slot in 0..refs {
                    let target = usize::from(c.values[slot]);
                    c.values[slot] = new_of[target].expect("reachable by construction");
                }
                c
            })
            .collect();
        hash_body(&mut h, body, &calls);
    }
    Ok(h.0)
}

/// A digest of the `ActionDef` envelope: everything a method's observable
/// behaviour depends on that is NOT in the program body. Lab-local; it uses
/// the `Debug` rendering of the OGAR types, so it is stable only for one
/// `ogar-vocab` build. It exists to show which changes [`body_rev`] cannot
/// see, not as a proposed identity.
pub fn envelope_rev(a: &ActionDef) -> u64 {
    let mut h = Fnv::new(b"ogar-behavior-probe/envelope/1");
    h.str(&a.identity);
    h.str(&format!("{:?}", a.kausal));
    for list in [&a.reads, &a.writes, &a.calls, &a.raises] {
        h.u64(list.len() as u64);
        for s in list {
            h.str(s);
        }
    }
    h.str(&format!("{:?}", a.on_enter));
    h.str(&format!("{:?}", a.guard_failure_policy));
    h.str(&format!("{:?}", a.state_timeout_millis));
    h.str(&format!("{:?}", a.default_modal));
    h.str(&format!("{:?}", a.default_temporal));
    h.0
}

/// Whether `f` is a byte the vocabulary reads as a body reference.
pub fn is_reference<V: Vocabulary>(vocab: &CheckedVocabulary<V>, f: FnIndex) -> bool {
    vocab.body_refs(f) > 0
}
