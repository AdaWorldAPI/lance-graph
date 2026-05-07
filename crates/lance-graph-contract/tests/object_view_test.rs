//! D-PARITY-V2-4 acceptance — ObjectView + NotificationSpec POD primitives.
//! Zero-dep crate: `Clone + Debug + PartialEq` only (matches MulThresholdProfile).

use lance_graph_contract::ontology::{
    DisplayTemplate, FieldRef, NotificationChannel, NotificationSpec, NotificationTrigger,
    ObjectView,
};

#[test]
fn object_view_three_fields_card_template() {
    let view = ObjectView::new(
        DisplayTemplate::Card,
        vec![
            FieldRef::new("ogit:name", "Name"),
            FieldRef::new("ogit:tax_id", "Tax ID"),
            FieldRef::new("ogit:created_at", "Created"),
        ],
    );

    assert_eq!(view.display_template, DisplayTemplate::Card);
    assert_eq!(view.fields.len(), 3);
    assert_eq!(view.fields[0].predicate_iri, "ogit:name");
    assert_eq!(view.fields[1].label, "Tax ID");
    assert!(view.primary_label.is_none());

    // Clone + PartialEq roundtrip.
    let clone = view.clone();
    assert_eq!(view, clone);
}

#[test]
fn object_view_with_primary_label() {
    let mut view = ObjectView::new(DisplayTemplate::Detail, vec![FieldRef::new("p", "L")]);
    view.primary_label = Some("p".to_string());
    assert_eq!(view.primary_label.as_deref(), Some("p"));
    assert_eq!(view.display_template, DisplayTemplate::Detail);
}

#[test]
fn notification_spec_created_webhook() {
    let spec = NotificationSpec::new(
        NotificationTrigger::Created,
        NotificationChannel::Webhook,
        "object {{name}} was created",
    );

    assert_eq!(spec.trigger, NotificationTrigger::Created);
    assert_eq!(spec.channel, NotificationChannel::Webhook);
    assert_eq!(spec.template, "object {{name}} was created");

    // Clone + PartialEq roundtrip.
    let clone = spec.clone();
    assert_eq!(spec, clone);
}

#[test]
fn notification_trigger_and_channel_variants_distinct() {
    assert_ne!(NotificationTrigger::Created, NotificationTrigger::Updated);
    assert_ne!(NotificationTrigger::Deleted, NotificationTrigger::ThresholdCrossed);
    assert_ne!(NotificationChannel::Inline, NotificationChannel::Email);
}
