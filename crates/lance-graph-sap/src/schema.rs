//! The harvested 23-field CATS schema, ordered by the ABAP leaf declarations.
//! Ordinals are append-only. DDIC scale, ALPHA exits and localized labels are
//! absent: none is defined by the pinned source. Widths come from the C# mirror.
use lance_graph_contract::class_view::{ClassId, ClassView, WideFieldMask};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use lance_graph_mask_risc::LaneKind;
use lance_graph_quack::Col;

pub const FIELD_COUNT: usize = 23;
/// Native type is cold source/domain metadata, not an execution type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldDescriptor {
    pub ordinal: Col,
    pub abap_group: &'static str,
    pub technical_name: &'static str,
    pub csharp_name: &'static str,
    pub native_type: &'static str,
    pub optional: bool,
    pub width: Option<usize>,
    pub carrier: LaneKind,
}

pub const FIELDS: [FieldDescriptor; FIELD_COUNT] = [
    FieldDescriptor {
        ordinal: Col(0),
        abap_group: "ty_s_entry_header",
        technical_name: "entry_id",
        csharp_name: "EntryID",
        native_type: "string",
        optional: false,
        width: None,
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(1),
        abap_group: "ty_s_entry_header",
        technical_name: "source_system",
        csharp_name: "SourceSystem",
        native_type: "string",
        optional: false,
        width: None,
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(2),
        abap_group: "ty_s_entry_header",
        technical_name: "entry_type",
        csharp_name: "EntryType",
        native_type: "string",
        optional: false,
        width: None,
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(3),
        abap_group: "ty_s_entry_header",
        technical_name: "tenant_id",
        csharp_name: "TenantID",
        native_type: "string",
        optional: false,
        width: None,
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(4),
        abap_group: "ty_s_entry_header",
        technical_name: "timestamp_utc",
        csharp_name: "TimestampUTC",
        native_type: "string",
        optional: false,
        width: None,
        carrier: LaneKind::U64,
    },
    FieldDescriptor {
        ordinal: Col(5),
        abap_group: "ty_s_time_entry_details",
        technical_name: "employee_number",
        csharp_name: "EmployeeNumber",
        native_type: "pernr_d",
        optional: false,
        width: Some(8),
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(6),
        abap_group: "ty_s_time_entry_details",
        technical_name: "customer_number",
        csharp_name: "CustomerNumber",
        native_type: "kunnr",
        optional: true,
        width: Some(10),
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(7),
        abap_group: "ty_s_time_entry_details",
        technical_name: "project_code",
        csharp_name: "ProjectCode",
        native_type: "ps_posid",
        optional: true,
        width: Some(24),
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(8),
        abap_group: "ty_s_time_entry_details",
        technical_name: "task_code",
        csharp_name: "TaskCode",
        native_type: "aufnr",
        optional: true,
        width: Some(12),
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(9),
        abap_group: "ty_s_time_entry_details",
        technical_name: "work_date_utc",
        csharp_name: "WorkDateUTC",
        native_type: "string",
        optional: false,
        width: None,
        carrier: LaneKind::U64,
    },
    FieldDescriptor {
        ordinal: Col(10),
        abap_group: "ty_s_time_entry_details",
        technical_name: "hours_logged",
        csharp_name: "HoursLogged",
        native_type: "catsquantity",
        optional: false,
        width: None,
        carrier: LaneKind::I32,
    },
    FieldDescriptor {
        ordinal: Col(11),
        abap_group: "ty_s_time_entry_details",
        technical_name: "activity_type",
        csharp_name: "ActivityType",
        native_type: "lstar",
        optional: false,
        width: Some(6),
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(12),
        abap_group: "ty_s_time_entry_details",
        technical_name: "billing_indicator",
        csharp_name: "BillingIndicator",
        native_type: "string",
        optional: false,
        width: None,
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(13),
        abap_group: "ty_s_time_sanitization",
        technical_name: "sanitized_input_date",
        csharp_name: "SanitizedInputDate",
        native_type: "string",
        optional: false,
        width: None,
        carrier: LaneKind::U64,
    },
    FieldDescriptor {
        ordinal: Col(14),
        abap_group: "ty_s_time_sanitization",
        technical_name: "original_input_date",
        csharp_name: "OriginalInputDate",
        native_type: "string",
        optional: false,
        width: None,
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(15),
        abap_group: "ty_s_time_sanitization",
        technical_name: "date_validation_status",
        csharp_name: "DateValidationStatus",
        native_type: "string",
        optional: false,
        width: None,
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(16),
        abap_group: "ty_s_time_sanitization",
        technical_name: "sanitization_reason",
        csharp_name: "SanitizationReason",
        native_type: "string",
        optional: true,
        width: None,
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(17),
        abap_group: "ty_s_security_metadata",
        technical_name: "encryption_indicator",
        csharp_name: "EncryptionIndicator",
        native_type: "abap_bool",
        optional: false,
        width: None,
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(18),
        abap_group: "ty_s_security_metadata",
        technical_name: "data_hash",
        csharp_name: "DataHash",
        native_type: "string",
        optional: false,
        width: None,
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(19),
        abap_group: "ty_s_security_metadata",
        technical_name: "compliance_label",
        csharp_name: "ComplianceLabel",
        native_type: "string",
        optional: false,
        width: None,
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(20),
        abap_group: "ty_s_additional_metadata",
        technical_name: "notes",
        csharp_name: "Notes",
        native_type: "string",
        optional: true,
        width: None,
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(21),
        abap_group: "ty_s_additional_metadata",
        technical_name: "approver_employee_num",
        csharp_name: "ApproverEmployeeNumber",
        native_type: "pernr_d",
        optional: true,
        width: Some(8),
        carrier: LaneKind::U32,
    },
    FieldDescriptor {
        ordinal: Col(22),
        abap_group: "ty_s_additional_metadata",
        technical_name: "approval_timestamp_utc",
        csharp_name: "ApprovalTimestampUTC",
        native_type: "string",
        optional: true,
        width: None,
        carrier: LaneKind::U64,
    },
];
/// Class identity is supplied by the caller's registry; this kit reserves no
/// global class ID. DOLCE category is likewise caller metadata, not guessed.
#[derive(Debug)]
pub struct CatsSchema {
    pub class: ClassId,
    category: u8,
    fields: Vec<FieldRef>,
}
impl CatsSchema {
    /// Explicit cold/UI projection boundary. The shared constructor is the
    /// same one used by od_ontology::view_mask::mint_wide_mask. Unknown names
    /// are refused before that constructor (which otherwise ignores them).
    pub fn realize_projection(&self, names: &[&str]) -> Option<WideFieldMask> {
        let universe: Vec<_> = FIELDS.iter().map(|f| f.technical_name).collect();
        let present: Option<Vec<_>> = names
            .iter()
            .map(|name| {
                self.resolve(name)
                    .map(|col| FIELDS[usize::from(col.0)].technical_name)
            })
            .collect();
        WideFieldMask::from_universe_present(&universe, &present?).ok()
    }
    pub fn new(class: ClassId, category: u8) -> Self {
        Self {
            class,
            category,
            fields: FIELDS
                .iter()
                .map(|f| {
                    FieldRef::new(
                        format!("urn:simaf:cats:{}:{}", f.abap_group, f.technical_name),
                        f.technical_name,
                    )
                })
                .collect(),
        }
    }
    /// Cold binding only. There is no name lookup in an execution program.
    pub fn resolve(&self, name: &str) -> Option<Col> {
        FIELDS
            .iter()
            .find(|f| f.technical_name == name || f.csharp_name == name)
            .map(|f| f.ordinal)
    }
}
impl ClassView for CatsSchema {
    fn fields(&self, class: ClassId) -> &[FieldRef] {
        if class == self.class {
            &self.fields
        } else {
            &[]
        }
    }
    fn template(&self, _: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _: ClassId) -> u8 {
        self.category
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_order_and_aliases_are_one_basis() {
        let schema = CatsSchema::new(42, 0);
        let tsv = include_str!("../schema.tsv");
        for (i, (f, line)) in FIELDS.iter().zip(tsv.lines().skip(1)).enumerate() {
            let cells: Vec<_> = line.split('\t').collect();
            assert_eq!(usize::from(f.ordinal.0), i);
            assert_eq!(cells[0], i.to_string());
            assert_eq!(cells[2], f.technical_name);
            assert_eq!(schema.resolve(cells[2]), schema.resolve(cells[4]));
            assert_eq!(schema.fields(42)[i].label, f.technical_name);
        }
        assert_eq!(schema.fields(42).len(), FIELD_COUNT);
        assert!(schema.fields(43).is_empty());
        assert_eq!(schema.resolve("invented"), None);
    }
}
