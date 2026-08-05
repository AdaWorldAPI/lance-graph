//! Auto-generated from /home/user/odoo/addons/l10n_de/ by `tools/odoo-blueprint-extractor`.
//! Do NOT edit by hand — re-run the extractor.
//! Plan: `.claude/plans/odoo-source-extraction-v1.md` (D-ODOO-EXT-2).

use crate::odoo_blueprint::*;

// NOTE: 2 duplicate model_name(s) merged (richest class kept).

pub const EXT_ACCOUNT_ACCOUNT: OdooEntity = OdooEntity {
    model_name: "account.account",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[],
    methods: &[OdooMethod {
        name: "write",
        kind: OdooMethodKind::Override,
        return_kind: OdooReturnKind::Boolean,
        triggers: &[],
    }],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef {
            path: "odoo/addons/l10n_de/models/account_account.py",
            line_range: (5, 19),
        }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_JOURNAL: OdooEntity = OdooEntity {
    model_name: "account.journal",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[],
    methods: &[OdooMethod {
        name: "_prepare_liquidity_account_vals",
        kind: OdooMethodKind::ApiModel,
        return_kind: OdooReturnKind::Unit,
        triggers: &[],
    }],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiModel,
        targets: &[],
    }],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef {
            path: "odoo/addons/l10n_de/models/account_journal.py",
            line_range: (6, 18),
        }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_MOVE: OdooEntity = OdooEntity {
    model_name: "account.move",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[],
    methods: &[
        OdooMethod {
            name: "_compute_show_delivery_date",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_post",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[OdooDecorator {
        kind: OdooDecoratorKind::ApiDepends,
        targets: &["country_code", "move_type"],
    }],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef {
            path: "odoo/addons/l10n_de/models/account_move.py",
            line_range: (4, 19),
        }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_CHART_TEMPLATE: OdooEntity = OdooEntity {
    model_name: "account.chart.template",
    kind: OdooEntityKind::Abstract,
    description: "",
    fields: &[],
    methods: &[
        OdooMethod {
            name: "_get_de_skr03_template_data",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_de_skr03_res_company",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_de_skr03_reconcile_model",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_get_de_skr03_account_account",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef {
            path: "odoo/addons/l10n_de/models/template_de_skr03.py",
            line_range: (6, 149),
        }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_ACCOUNT_TAX: OdooEntity = OdooEntity {
    model_name: "account.tax",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[OdooField {
        name: "l10n_de_datev_code",
        kind: OdooFieldKind::Char,
        target: None,
        required: false,
        computed: None,
        depends: &[],
        semantic_role: OdooSemanticRole::Other,
    }],
    methods: &[],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef {
            path: "odoo/addons/l10n_de/models/datev.py",
            line_range: (4, 7),
        }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_PRODUCT_TEMPLATE: OdooEntity = OdooEntity {
    model_name: "product.template",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[],
    methods: &[OdooMethod {
        name: "_get_product_accounts",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Unit,
        triggers: &[],
    }],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef {
            path: "odoo/addons/l10n_de/models/datev.py",
            line_range: (10, 37),
        }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};

pub const EXT_IR_ACTIONS_REPORT: OdooEntity = OdooEntity {
    model_name: "ir.actions.report",
    kind: OdooEntityKind::Model,
    description: "",
    fields: &[],
    methods: &[OdooMethod {
        name: "_get_rendering_context",
        kind: OdooMethodKind::Helper,
        return_kind: OdooReturnKind::Unit,
        triggers: &[],
    }],
    decorators: &[],
    state_machine: None,
    constraints: &[],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef {
            path: "odoo/addons/l10n_de/models/ir_actions_report.py",
            line_range: (4, 10),
        }],
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
            name: "l10n_de_stnr",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
        OdooField {
            name: "l10n_de_widnr",
            kind: OdooFieldKind::Char,
            target: None,
            required: false,
            computed: None,
            depends: &[],
            semantic_role: OdooSemanticRole::Other,
        },
    ],
    methods: &[
        OdooMethod {
            name: "write",
            kind: OdooMethodKind::Override,
            return_kind: OdooReturnKind::Boolean,
            triggers: &[],
        },
        OdooMethod {
            name: "_compute_force_restrictive_audit_trail",
            kind: OdooMethodKind::Compute,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "_validate_l10n_de_stnr",
            kind: OdooMethodKind::Constrain,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
        OdooMethod {
            name: "get_l10n_de_stnr_national",
            kind: OdooMethodKind::Helper,
            return_kind: OdooReturnKind::Unit,
            triggers: &[],
        },
    ],
    decorators: &[
        OdooDecorator {
            kind: OdooDecoratorKind::ApiDepends,
            targets: &["country_code"],
        },
        OdooDecorator {
            kind: OdooDecoratorKind::ApiConstrains,
            targets: &["state_id", "l10n_de_stnr"],
        },
    ],
    state_machine: None,
    constraints: &[OdooConstraint {
        kind: OdooConstraintKind::Python,
        condition: "Python constraint on state_id, l10n_de_stnr",
        source_method: Some("_validate_l10n_de_stnr"),
    }],
    provenance: OdooProvenance {
        l_doc: "",
        l_doc_lines: (0, 0),
        odoo_source: &[OdooSourceRef {
            path: "odoo/addons/l10n_de/models/res_company.py",
            line_range: (10, 60),
        }],
        confidence: OdooConfidence::Extracted,
        regulation_iri: &[],
    },
};
