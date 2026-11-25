//! The admission gate (S2 of `vrl-vector-openobserve-census-v1`): a program
//! is admitted only if upstream VRL compiles it against the declared schema
//! and the recipe lowers it. Upstream decides what VRL means; the lowering
//! decides what runs in-fold. A program either side refuses never runs.
//!
//! Upstream is the pin the crate is built with: `upstream-vector` (Vector's
//! vrl) when enabled, else `upstream-o2` (OpenObserve's).

use std::collections::BTreeMap;

use lance_graph_contract::class_view::{ClassId, ClassView};

#[cfg(all(feature = "upstream-o2", not(feature = "upstream-vector")))]
use vrl_o2 as vrl;
#[cfg(feature = "upstream-vector")]
use vrl_vector as vrl;

use vrl::compiler::state::ExternalEnv;
use vrl::compiler::{compile_with_external, CompileConfig};
use vrl::value::kind::{Collection, Field};
use vrl::value::Kind;

use crate::{Recipe, Refusal};

/// A field's value type as the source declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclaredKind {
    Integer,
    Text,
    Boolean,
}

/// One field of the declared schema: its view label, its type, and whether
/// a row may lack it. An optional field may be missing or null, so an order
/// compare over it is fallible and upstream refuses it with E100.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Declared {
    pub label: String,
    pub kind: DeclaredKind,
    pub optional: bool,
}

impl Declared {
    pub fn new(label: &str, kind: DeclaredKind, optional: bool) -> Declared {
        Declared {
            label: label.to_owned(),
            kind,
            optional,
        }
    }

    fn kind(&self) -> Kind {
        let k = match self.kind {
            DeclaredKind::Integer => Kind::integer(),
            DeclaredKind::Text => Kind::bytes(),
            DeclaredKind::Boolean => Kind::boolean(),
        };
        if self.optional {
            k.or_null().or_undefined()
        } else {
            k
        }
    }
}

/// The event shape the recipe reads: `key` at the top, every other field at
/// `.properties.<label>.value`.
fn external(schema: &[Declared]) -> ExternalEnv {
    let mut props: BTreeMap<Field, Kind> = BTreeMap::new();
    let mut top: BTreeMap<Field, Kind> = BTreeMap::new();
    for d in schema {
        if d.label == "key" {
            top.insert("key".into(), d.kind());
        } else {
            let mut v: BTreeMap<Field, Kind> = BTreeMap::new();
            v.insert("value".into(), d.kind());
            props.insert(d.label.as_str().into(), Kind::object(Collection::from(v)));
        }
    }
    top.insert("properties".into(), Kind::object(Collection::from(props)));
    ExternalEnv::new_with_kind(
        Kind::object(Collection::from(top)),
        Kind::object(Collection::empty()),
    )
}

impl Recipe {
    /// Compile `src` as [`Recipe::compile`] does, after upstream VRL has
    /// compiled it against `schema`. Upstream refusing is
    /// [`Refusal::Upstream`] with its diagnostic codes; the lowering's
    /// refusals are unchanged.
    pub fn admit<V: ClassView + ?Sized>(
        src: &str,
        view: &V,
        class: ClassId,
        schema: &[Declared],
    ) -> Result<Recipe, Refusal> {
        let fns = vrl::stdlib::all();
        if let Err(diagnostics) =
            compile_with_external(src, &fns, &external(schema), CompileConfig::default())
        {
            let mut codes: Vec<usize> = diagnostics.iter().map(|d| d.code).collect();
            codes.sort_unstable();
            codes.dedup();
            return Err(Refusal::Upstream { codes });
        }
        Recipe::compile(src, view, class)
    }
}
