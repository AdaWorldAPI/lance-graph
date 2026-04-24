//! The `Component` trait — ECS-style component definition transcoded to Arrow.
//!
//! Per ADR-0001 Decision 1 (Archetype Transcode, not bridge): a `Component`
//! is a Rust-side type that declares how it projects into an Arrow
//! `Field`. The transcode surface is Arrow because every downstream
//! consumer of this crate lands in a Lance dataset; the `arrow_field`
//! method is what a `Processor` keys its `matches(schema)` check against.
//!
//! This trait deliberately stays Sized-agnostic at the scaffold stage —
//! only associated functions, no self-receiver. Implementors declare
//! static metadata (field shape, type ID) and the runtime machinery
//! lives elsewhere.

use arrow::datatypes::Field;

/// An ECS-style component that knows how to project itself into an Arrow
/// `Field`. Components do not carry row data at this stage — they declare
/// SHAPE. Row data flows through `RecordBatch`es handed to `Processor`.
///
/// **BBB-invariant:** component types defined by implementors live
/// INSIDE-BBB. They do not cross the external membrane (see
/// `lance_graph_contract::external_membrane`). The scalar projection
/// "a component tick happened" is carried by `CognitiveEventRow`'s
/// existing columns (`cycle_fp_hi/lo`, `MetaWord`); this crate does
/// not extend that row.
pub trait Component {
    /// Arrow field descriptor for this component. Called once at
    /// `Processor::matches` time, not per-row. Implementors should
    /// return a `Field` with a stable name and dtype.
    fn arrow_field() -> Field;

    /// Stable string identifier for this component type. Used by the
    /// `CommandBroker` drain path to address entities-by-component
    /// without relying on Rust's `TypeId` (which is not stable across
    /// builds). Convention: `"<crate>::<type>"`.
    fn type_id() -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::DataType;

    /// Test-only component used to assert that the trait is implementable
    /// and that its metadata is reachable without constructing a value.
    struct MockComponent;

    impl Component for MockComponent {
        fn arrow_field() -> Field {
            Field::new("mock_component", DataType::Int64, false)
        }

        fn type_id() -> &'static str {
            "lance_graph_archetype::tests::MockComponent"
        }
    }

    #[test]
    fn mock_component_has_arrow_field() {
        let field = MockComponent::arrow_field();
        assert_eq!(field.name(), "mock_component");
        assert_eq!(field.data_type(), &DataType::Int64);
        assert!(!field.is_nullable());
    }

    #[test]
    fn mock_component_type_id_is_stable() {
        assert_eq!(
            MockComponent::type_id(),
            "lance_graph_archetype::tests::MockComponent"
        );
    }
}
