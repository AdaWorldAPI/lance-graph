//! Auto-generated from /home/user/odoo/addons/account_payment/ by `tools/odoo-blueprint-extractor`.
//! Do NOT edit by hand — re-run the extractor.
//! Plan: `.claude/plans/odoo-source-extraction-v1.md` (D-ODOO-EXT-2).

use crate::odoo_blueprint::*;

pub const EXT_ACCOUNT_JOURNAL: OdooEntity = OdooEntity {
    model_name: "account.journal",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[

    ],
    methods: &[
        OdooMethod {
            name: "_get_available_payment_method_lines",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_unlink_except_linked_to_payment_provider",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[

    ],
    state_machine: None,
    constraints: &[

    ],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_payment/models/account_journal.py", line_range: (8, 25) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_MOVE: OdooEntity = OdooEntity {
    model_name: "account.move",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[
        OdooField {
            name: "transaction_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("payment.transaction"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "authorized_transaction_ids",
            kind: OdooFieldKind::Computed,
            target: Some("payment.transaction"),
            required: false,
            computed: Some("_compute_authorized_transaction_ids"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "transaction_count",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_transaction_count"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "amount_paid",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_amount_paid"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_authorized_transaction_ids",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_transaction_count",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_amount_paid",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_has_to_be_paid",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_online_payment_error",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "get_portal_last_transaction",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "payment_action_capture",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "payment_action_void",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "action_view_payment_transactions",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_default_payment_link_values",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_generate_portal_payment_qr",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_portal_payment_link",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["transaction_ids"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["transaction_ids"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["transaction_ids"],
        },
    ],
    state_machine: None,
    constraints: &[

    ],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_payment/models/account_move.py", line_range: (13, 182) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_PAYMENT: OdooEntity = OdooEntity {
    model_name: "account.payment",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[
        OdooField {
            name: "payment_transaction_id",
            kind: OdooFieldKind::Many2one,
            target: Some("payment.transaction"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "payment_token_id",
            kind: OdooFieldKind::Many2one,
            target: Some("payment.token"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "amount_available_for_refund",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_amount_available_for_refund"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "suitable_payment_token_ids",
            kind: OdooFieldKind::Computed,
            target: Some("payment.token"),
            required: false,
            computed: Some("_compute_suitable_payment_token_ids"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "use_electronic_payment_method",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_use_electronic_payment_method"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "source_payment_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.payment"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "refunds_count",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_refunds_count"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_amount_available_for_refund",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_suitable_payment_token_ids",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_use_electronic_payment_method",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_refunds_count",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_onchange_set_payment_token_id",
            kind: OdooMethodKind::Onchange,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "action_post",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
            triggers: &[],
        },
        OdooMethod {
            name: "action_refund_wizard",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
            triggers: &[],
        },
        OdooMethod {
            name: "action_view_refunds",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
            triggers: &[],
        },
        OdooMethod {
            name: "_create_payment_transaction",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_prepare_payment_transaction_vals",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_payment_refund_wizard_values",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["payment_method_line_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["payment_method_line_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiOnchange,
            targets: &["partner_id", "payment_method_line_id", "journal_id"],
        },
    ],
    state_machine: None,
    constraints: &[

    ],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_payment/models/account_payment.py", line_range: (7, 230) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_PAYMENT_METHOD: OdooEntity = OdooEntity {
    model_name: "account.payment.method",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[

    ],
    methods: &[
        OdooMethod {
            name: "_get_payment_method_information",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModel,
            targets: &[],
        },
    ],
    state_machine: None,
    constraints: &[

    ],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_payment/models/account_payment_method.py", line_range: (7, 20) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_PAYMENT_METHOD_LINE: OdooEntity = OdooEntity {
    model_name: "account.payment.method.line",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[
        OdooField {
            name: "payment_provider_id",
            kind: OdooFieldKind::Computed,
            target: Some("payment.provider"),
            required: false,
            computed: Some("_compute_payment_provider_id"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "payment_provider_state",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_name",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_payment_provider_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_unlink_except_active_provider",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "action_open_provider_form",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["payment_provider_id.name"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["payment_method_id"],
        },
    ],
    state_machine: None,
    constraints: &[

    ],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_payment/models/account_payment_method_line.py", line_range: (7, 85) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_PAYMENT_PROVIDER: OdooEntity = OdooEntity {
    model_name: "payment.provider",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[
        OdooField {
            name: "journal_id",
            kind: OdooFieldKind::Computed,
            target: Some("account.journal"),
            required: false,
            computed: Some("_compute_journal_id"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_ensure_payment_method_line",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_payment_method_outstanding_account_id",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_journal_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_inverse_journal_id",
            kind: OdooMethodKind::Inverse,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_provider_payment_method",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_setup_provider",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_setup_payment_method",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_check_existing_payment",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_remove_provider",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["code", "state", "company_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModel,
            targets: &[],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModel,
            targets: &[],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModel,
            targets: &[],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModel,
            targets: &[],
        },
    ],
    state_machine: None,
    constraints: &[

    ],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_payment/models/payment_provider.py", line_range: (7, 147) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_PAYMENT_TRANSACTION: OdooEntity = OdooEntity {
    model_name: "payment.transaction",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[
        OdooField {
            name: "payment_id",
            kind: OdooFieldKind::Many2one,
            target: Some("account.payment"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "invoice_ids",
            kind: OdooFieldKind::Many2many,
            target: Some("account.move"),
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "invoices_count",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_invoices_count"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_invoices_count",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "action_view_invoices",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_reference_prefix",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_post_process",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_create_payment",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_log_message_on_linked_documents",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_invoices_to_notify",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["invoice_ids"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModel,
            targets: &[],
        },
    ],
    state_machine: None,
    constraints: &[

    ],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_payment/models/payment_transaction.py", line_range: (6, 242) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

