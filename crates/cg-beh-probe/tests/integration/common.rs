use cg_beh_probe::dispatch::{self, Dispatch};
use cg_beh_probe::facts::Snapshot;
use cg_beh_probe::fixture::{self, FoundryVariant, TestWorld, Variant};
use cg_beh_probe::query::{MethodDelta, OldRef};
use foundry_parity_lab::bind::World;
use lance_graph_dt64::{BindStore, MethodId, Refusal};

pub const REV: &str = "test-revision";

pub fn lab(v: Variant) -> (TestWorld, Snapshot) {
    let w = fixture::world(v);
    let mut s = Snapshot::new(REV, "lab").unwrap();
    s.add_class(
        &w.store,
        &w.view,
        fixture::CLASS,
        &fixture::METHODS,
        &fixture::READ,
    )
    .unwrap();
    (w, s)
}

pub fn foundry(v: FoundryVariant) -> (World, Snapshot) {
    let w = fixture::foundry(v);
    let mut s = Snapshot::new(REV, "foundry").unwrap();
    s.add_class(
        &w.store,
        &w.view,
        fixture::FOUNDRY_CLASS,
        &fixture::foundry_ids(),
        &fixture::FOUNDRY_READ,
    )
    .unwrap();
    (w, s)
}

pub fn delta(ds: &[MethodDelta], method: u32) -> &MethodDelta {
    ds.iter().find(|d| d.method == method).unwrap()
}

/// What `new` answers a reference `old` issued for `method`, as the query's
/// [`OldRef`] vocabulary.
pub fn observed(old: &BindStore, new: &BindStore, class: u32, method: u32) -> OldRef {
    let r = old.method_ref(class, MethodId(method)).unwrap();
    match new.admit(&r) {
        Err(Refusal::SignatureRevisionMismatch { .. }) => OldRef::Signature,
        Err(
            Refusal::GuardMismatch { .. }
            | Refusal::GuardDropped { .. }
            | Refusal::GuardAdded { .. },
        ) => OldRef::Guard,
        Err(Refusal::BindingMismatch { .. }) => OldRef::Binding,
        Err(Refusal::ForeignStore { .. }) => OldRef::IssuerOnly,
        other => panic!("unexpected admission answer {other:?}"),
    }
}

pub fn dispatch_source() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../lance-graph-r2il/src/lib.rs"
    ))
    .unwrap()
}

pub fn real_dispatch() -> (String, Dispatch) {
    let src = dispatch_source();
    let d = dispatch::read(&src).unwrap();
    (src, d)
}
