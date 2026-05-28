//! Auto-generated from /home/user/odoo/addons/account_peppol/ by `tools/odoo-blueprint-extractor`.
//! Do NOT edit by hand — re-run the extractor.
//! Plan: `.claude/plans/odoo-source-extraction-v1.md` (D-ODOO-EXT-2).

use crate::odoo_blueprint::*;

pub const EXT_ACCOUNT_EDI_COMMON: OdooEntity = OdooEntity {
    model_name: "account.edi.common",
    kind: OdooEntityKind::Abstract,
    description: "",
    fields: &[

    ],
    methods: &[
        OdooMethod {
            name: "_add_logs_import_invoice_ubl_cii",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_log_import_invoice_ubl_cii",
            kind: OdooMethodKind::Helper,
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
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_peppol/models/account_edi_common.py", line_range: (4, 18) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_EDI_PROXY_CLIENT_USER: OdooEntity = OdooEntity {
    model_name: "account_edi_proxy_client.user",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[
        OdooField {
            name: "proxy_type",
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
            name: "_get_proxy_urls",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_peppol_error_message",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_call_peppol_proxy",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_mark_connection_out_of_sync",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_out_of_sync_reconnect_this_database",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_out_of_sync_disconnect_this_database",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_can_send_domain",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_cron_peppol_get_new_documents",
            kind: OdooMethodKind::Cron,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_cron_peppol_get_message_status",
            kind: OdooMethodKind::Cron,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_cron_peppol_get_participant_status",
            kind: OdooMethodKind::Cron,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_cron_peppol_webhook_keepalive",
            kind: OdooMethodKind::Cron,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_proxy_identification",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_import_invoice",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_get_new_documents",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_process_new_messages",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_post_process_new_messages",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_get_message_status",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_get_documents_for_status",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_process_messages_status",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_get_participant_status",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_company_details",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_register_sender",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_register_sender_as_receiver",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_deregister_participant",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_deregister_participant_to_sender",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_auto_register_services",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_auto_deregister_services",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_get_services",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_generate_webhook_token",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_user_from_token",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_reset_webhook",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
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
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_peppol/models/account_edi_proxy_user.py", line_range: (18, 586) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_EDI_UBL: OdooEntity = OdooEntity {
    model_name: "account.edi.ubl",
    kind: OdooEntityKind::Abstract,
    description: "",
    fields: &[

    ],
    methods: &[
        OdooMethod {
            name: "_ubl_add_values_company",
            kind: OdooMethodKind::Helper,
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
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_peppol/models/account_edi_ubl_xml.py", line_range: (4, 15) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_EDI_XML_UBL_BIS3: OdooEntity = OdooEntity {
    model_name: "account.edi.xml.ubl_bis3",
    kind: OdooEntityKind::Abstract,
    description: "",
    fields: &[

    ],
    methods: &[
        OdooMethod {
            name: "_invoice_constraints_peppol_en16931_ubl",
            kind: OdooMethodKind::Helper,
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
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_peppol/models/account_edi_xml_ubl_bis3.py", line_range: (4, 29) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_JOURNAL: OdooEntity = OdooEntity {
    model_name: "account.journal",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[
        OdooField {
            name: "account_peppol_proxy_state",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "is_peppol_journal",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_check_type_for_peppol_journal",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_show_refresh_out_einvoices_status_button",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_show_fetch_in_einvoices_button",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "button_fetch_in_einvoices",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "button_refresh_out_einvoices_status",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["type"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["account_peppol_proxy_state"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["is_peppol_journal", "account_peppol_proxy_state"],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "Python constraint on type",
            source_method: Some("_check_type_for_peppol_journal"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_peppol/models/account_journal.py", line_range: (5, 67) }],
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
            name: "peppol_message_uuid",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_move_state",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_peppol_move_state"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_is_sent",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_peppol_is_sent"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
    ],
    methods: &[
        OdooMethod {
            name: "action_send_and_print",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
            triggers: &[],
        },
        OdooMethod {
            name: "action_cancel_peppol_documents",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_display_send_button",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_peppol_move_state",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_peppol_is_sent",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_notify_by_email_prepare_rendering_context",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["state"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["peppol_move_state"],
        },
    ],
    state_machine: None,
    constraints: &[

    ],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_peppol/models/account_move.py", line_range: (8, 91) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_MOVE_SEND: OdooEntity = OdooEntity {
    model_name: "account.move.send",
    kind: OdooEntityKind::Abstract,
    description: "",
    fields: &[

    ],
    methods: &[
        OdooMethod {
            name: "_get_default_sending_methods",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_alerts",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_default_invoice_edi_format",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_mail_layout",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_do_peppol_pre_send",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_is_applicable_to_company",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_is_applicable_to_move",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_hook_if_errors",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_call_web_service_after_invoice_pdf_render",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "action_what_is_peppol_activate",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
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
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_peppol/models/account_move_send.py", line_range: (15, 296) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_RES_COMPANY: OdooEntity = OdooEntity {
    model_name: "res.company",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[
        OdooField {
            name: "account_peppol_contact_email",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_account_peppol_contact_email"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_migration_key",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_phone_number",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_account_peppol_phone_number"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_proxy_state",
            kind: OdooFieldKind::Selection,
            target: None,
            required: true,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_edi_user",
            kind: OdooFieldKind::Computed,
            target: Some("account_edi_proxy_client.user"),
            required: false,
            computed: Some("_compute_account_peppol_edi_user"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_eas",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_endpoint",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_purchase_journal_id",
            kind: OdooFieldKind::Computed,
            target: Some("account.journal"),
            required: false,
            computed: Some("_compute_peppol_purchase_journal_id"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_external_provider",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_can_send",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_peppol_can_send"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_parent_company_id",
            kind: OdooFieldKind::Computed,
            target: Some("res.company"),
            required: false,
            computed: Some("_compute_peppol_parent_company_id"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_metadata",
            kind: OdooFieldKind::Other,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_metadata_updated_at",
            kind: OdooFieldKind::Datetime,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_activate_self_billing_sending",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_self_billing_reception_journal_id",
            kind: OdooFieldKind::Computed,
            target: Some("account.journal"),
            required: false,
            computed: Some("_compute_peppol_self_billing_reception_journal_id"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_get_active_peppol_parent_company",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_have_unauthorized_peppol_parent_company",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_reset_peppol_configuration",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_check_phonenumbers_import",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_sanitize_peppol_phone_number",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_check_peppol_endpoint_number",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_check_account_peppol_phone_number",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_check_peppol_endpoint",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_check_peppol_purchase_journal_id",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_account_peppol_edi_user",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_peppol_parent_company_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_peppol_purchase_journal_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_inverse_peppol_purchase_journal_id",
            kind: OdooMethodKind::Inverse,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_peppol_self_billing_reception_journal_id",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_inverse_peppol_self_billing_reception_journal_id",
            kind: OdooMethodKind::Inverse,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_account_peppol_contact_email",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_account_peppol_phone_number",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_peppol_can_send",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_sanitize_peppol_endpoint_in_values",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "create",
            kind: OdooMethodKind::ApiModelCreateMulti,
            return_kind: OdooReturnKind::Record,
            triggers: &[],
        },
        OdooMethod {
            name: "write",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_modules_document_types",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_supported_document_types",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_peppol_edi_mode",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_peppol_webhook_endpoint",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_company_info_on_peppol",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_account_peppol_send_welcome_email",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModel,
            targets: &[],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["account_peppol_phone_number"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["peppol_endpoint"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["peppol_purchase_journal_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["account_edi_proxy_client_ids"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["peppol_eas", "peppol_endpoint"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["account_peppol_proxy_state"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["account_peppol_proxy_state"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["email"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["phone"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["account_peppol_proxy_state"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModel,
            targets: &[],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiModelCreateMulti,
            targets: &[],
        },
    ],
    state_machine: None,
    constraints: &[
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "Python constraint on account_peppol_phone_number",
            source_method: Some("_check_account_peppol_phone_number"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "Python constraint on peppol_endpoint",
            source_method: Some("_check_peppol_endpoint"),
        },
        OdooConstraint {
            kind: OdooConstraintKind::Python,
            condition: "Python constraint on peppol_purchase_journal_id",
            source_method: Some("_check_peppol_purchase_journal_id"),
        },
    ],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_peppol/models/res_company.py", line_range: (54, 448) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &["ogit:regulation/de/ustg/13", "ogit:regulation/eu/en16931"],
    },
};

pub const EXT_RES_CONFIG_SETTINGS: OdooEntity = OdooEntity {
    model_name: "res.config.settings",
    kind: OdooEntityKind::Transient,
    description: "",
    fields: &[
        OdooField {
            name: "account_peppol_edi_user",
            kind: OdooFieldKind::Many2one,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_edi_mode",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_contact_email",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_account_peppol_contact_email"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_eas",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_edi_identification",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_endpoint",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_migration_key",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_phone_number",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_proxy_state",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_peppol_purchase_journal_id",
            kind: OdooFieldKind::Many2one,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_external_provider",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_use_parent_company",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_peppol_use_parent_company"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_parent_company_name",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_peppol_use_parent_company"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "account_is_token_out_of_sync",
            kind: OdooFieldKind::Boolean,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_participation_role",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_peppol_participation_role"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
    ],
    methods: &[
        OdooMethod {
            name: "_compute_peppol_use_parent_company",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_peppol_participation_role",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_inverse_peppol_participation_role",
            kind: OdooMethodKind::Inverse,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_account_peppol_contact_email",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_inverse_account_peppol_contact_email",
            kind: OdooMethodKind::Inverse,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "action_open_peppol_form",
            kind: OdooMethodKind::Action,
            return_kind: OdooReturnKind::Action,
            triggers: &[],
        },
        OdooMethod {
            name: "button_open_peppol_config_wizard",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "button_peppol_disconnect_branch_from_parent",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "button_peppol_register_sender_as_receiver",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "button_reconnect_this_database",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "button_disconnect_this_database",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "button_peppol_deregister",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["company_id.peppol_parent_company_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["account_peppol_proxy_state"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["company_id.account_peppol_contact_email"],
        },
    ],
    state_machine: None,
    constraints: &[

    ],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_peppol/models/res_config_settings.py", line_range: (4, 165) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_RES_PARTNER: OdooEntity = OdooEntity {
    model_name: "res.partner",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[
        OdooField {
            name: "invoice_sending_method",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_eas",
            kind: OdooFieldKind::Selection,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "available_peppol_sending_methods",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_available_peppol_sending_methods"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "available_peppol_edi_formats",
            kind: OdooFieldKind::Computed,
            target: None,
            required: false,
            computed: Some("_compute_available_peppol_edi_formats"),
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "peppol_verification_state",
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
            name: "_onchange_verify_peppol_status",
            kind: OdooMethodKind::Onchange,
            return_kind: OdooReturnKind::Dict,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_available_peppol_sending_methods",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_available_peppol_edi_formats",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_available_peppol_eas",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_log_verification_state_update",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_participant_info",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_check_peppol_participant_exists",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_peppol_lookup_participant",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_check_document_type_support",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_update_peppol_state_per_company",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "create",
            kind: OdooMethodKind::ApiModelCreateMulti,
            return_kind: OdooReturnKind::Record,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_peppol_endpoint",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_peppol_eas",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "button_account_peppol_check_partner_endpoint",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_peppol_verification_state",
            kind: OdooMethodKind::ApiModel,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_frontend_writable_fields",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_partners_to_skip_peppol_computation",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiOnchange,
            targets: &["invoice_edi_format", "peppol_endpoint", "peppol_eas"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["company_id"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["invoice_sending_method"],
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
            kind: OdooDecoratorKind::ApiModelCreateMulti,
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
        odoo_source: &[OdooSourceRef { path: "/home/user/odoo/addons/account_peppol/models/res_partner.py", line_range: (19, 303) }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

