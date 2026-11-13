//! D-VAS-C3: a guard's residual lane is a field position of the class it is
//! bound to, and `register` refuses a guard that names a position outside the
//! class basis. `Sym` leaves name no field and are not judged here.

use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_contract::set_algebra::{Cmp, Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::{
    view_class, BindStore, ClassSchema, Extent, Guard, MethodBinding, MethodId, Refusal, SchemaRev,
    SigRev, ValueDemand,
};

struct View {
    class: ClassId,
    fields: Vec<FieldRef>,
}

impl ClassView for View {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == self.class {
            &self.fields
        } else {
            &[]
        }
    }
    fn template(&self, _class: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _class: ClassId) -> u8 {
        0
    }
}

const CLASS: u32 = 0x0A1E_0001;
const FIELDS: usize = 4;

fn guarded(expr: SetExpr) -> MethodBinding {
    MethodBinding {
        id: MethodId(1),
        sig_rev: SigRev(1),
        name: "guarded".into(),
        on: WideFieldMask::from_positions(&[0]),
        when: Some(Guard::new(expr, Logic::TwoValued)),
        reads: WideFieldMask::from_positions(&[0, 1]),
        effects: vec![],
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }
}

fn register(expr: SetExpr) -> Result<(), Refusal> {
    let view = View {
        class: view_class(CLASS),
        fields: (0..FIELDS)
            .map(|i| FieldRef::new(format!("urn:f{i}"), format!("f{i}")))
            .collect(),
    };
    let schema = ClassSchema {
        schema_rev: SchemaRev(1),
        field_count: FIELDS,
        extent: Extent::DR64_CUBE,
    };
    BindStore::new().register(&view, CLASS, schema, vec![guarded(expr)])
}

/// Positive: every lane a guard names inside the basis registers, at every
/// nesting depth and for both residual kinds.
#[test]
fn a_guard_over_fields_of_the_basis_registers() {
    let last = (FIELDS - 1) as u16;
    for expr in [
        SetExpr::eq_u32(last, 7),
        SetExpr::i32(0, Cmp::Lt, 3),
        SetExpr::And(vec![
            SetExpr::negated(SetExpr::i32(last, Cmp::Ge, 1)),
            SetExpr::Or(vec![SetExpr::ne_u32(1, 2), SetExpr::Sym(900)]),
        ]),
        SetExpr::All,
    ] {
        assert_eq!(register(expr.clone()), Ok(()), "{expr:?}");
    }
}

/// Negative: one position past the basis is refused, wherever in the guard
/// it sits, for both residual kinds, and past what a `u8` position can name.
#[test]
fn a_guard_naming_a_field_outside_the_basis_is_refused() {
    let past = FIELDS as u16;
    for expr in [
        SetExpr::eq_u32(past, 7),
        SetExpr::i32(past, Cmp::Lt, 3),
        SetExpr::i32(300, Cmp::Lt, 3),
        SetExpr::And(vec![
            SetExpr::i32(0, Cmp::Lt, 3),
            SetExpr::negated(SetExpr::Or(vec![SetExpr::Sym(1), SetExpr::ne_u32(past, 2)])),
        ]),
    ] {
        assert!(
            matches!(
                register(expr.clone()),
                Err(Refusal::GuardLaneOutOfBasis {
                    class: CLASS,
                    method: MethodId(1),
                    lane
                }) if usize::from(lane) >= FIELDS
            ),
            "registered or refused for another reason: {expr:?}"
        );
    }
}
