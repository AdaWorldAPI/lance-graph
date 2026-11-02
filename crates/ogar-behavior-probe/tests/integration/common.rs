//! Shared fixtures: real OGAR data, nothing hand-invented.

use ogar_behavior_probe::projection::Basis;
use ogar_from_ruff::mint::{compile_graph_python, CompiledClass};
use ogar_loco::vocabulary::conformance::{validate, CheckedVocabulary};
use ogar_loco::{FnIndex, Vocabulary};
use ogar_vocab::ports::OdooPort;

/// Verbatim Odoo 19 `addons/account/models/account_payment_term.py`, the
/// same bytes OGAR's own kausal-parity test reads (see
/// `fixtures/account_payment_term.PROVENANCE.md`).
pub const ACCOUNT_PAYMENT_TERM_PY: &str = include_str!("../../fixtures/account_payment_term.py");

/// The two models of the fixture, lifted exactly as
/// `ogar-from-ruff/tests/odoo_kausal_parity_probe.rs` lifts them.
pub fn compiled() -> Vec<CompiledClass> {
    let graph = ruff_python_spo::extract_from_source(ACCOUNT_PAYMENT_TERM_PY);
    compile_graph_python::<OdooPort>(&graph)
}

pub fn class<'a>(c: &'a [CompiledClass], name: &str) -> &'a CompiledClass {
    c.iter()
        .find(|c| c.class.name == name)
        .expect("model compiles")
}

/// The lab basis: attribute names, then association names, in lift order.
pub fn basis(c: &CompiledClass) -> Basis {
    let mut names: Vec<String> = c.class.attributes.iter().map(|a| a.name.clone()).collect();
    names.extend(c.class.associations.iter().map(|a| a.name.clone()));
    Basis::new(names)
}

/// The shared core and nothing else (ogar-loco's own `EmptyVocab` shape).
pub struct CoreOnly;
impl Vocabulary for CoreOnly {
    fn domain_stack_arity(&self, _f: FnIndex) -> Option<u8> {
        None
    }
    fn domain_body_refs(&self, _f: FnIndex) -> u8 {
        0
    }
}

pub fn core() -> CheckedVocabulary<CoreOnly> {
    validate(CoreOnly).expect("the shared core conforms")
}
