# Odoo high-signal concerns (SoC-deduplicated)

From 3555 ruff-py-dto bundles, 3555 delegation-resolved, 2965 unique SoC-concerns after synergy dedup.

SoC key = `(name_root, primary_decorator, (reads, writes, invokes, raises, reads_env, traverses))`. Two methods are the same concern iff this tuple matches.

| # | n | root | decorator | reads | writes | invokes | raises | env | families |
| ---: | ---: | --- | --- | --- | --- | --- | --- | :---: | --- |
| 1 | 88 | `display_name` | `@api.depends` | — | — | — | — | — | account_account, account_account_tag, account_incoterms, account_journal, account_move, … (+81) |
| 2 | 24 | `name` | `@api.depends` | — | — | — | — | — | account_bank_statement, account_payment_method, account_payment_method_line, calendar_recurrence, crm_lead, … (+16) |
| 3 | 16 | `company_id` | `@api.depends` | — | — | — | — | — | account_move, account_partial_reconcile, account_payment, account_payment_register_withholding_line, account_payment_withholding_line, … (+11) |
| 4 | 13 | `currency_id` | `@api.depends` | — | — | — | — | — | account_bank_statement, account_bank_statement_line, account_move, account_move_line, account_payment, … (+8) |
| 5 | 10 | `allowed_uom_ids` | `@api.depends` | — | — | — | — | — | account_move_line, mrp_production, purchase_order_line, repair, sale_order_line, … (+4) |
| 6 | 10 | `show_reset_to_draft_button` | `@api.depends` | — | — | — | — | — | account_move |
| 7 | 8 | `product_uom_id` | `@api.depends` | — | — | — | — | — | mrp_bom, mrp_unbuild, product_supplierinfo, purchase_requisition, sale_order_line, … (+3) |
| 8 | 6 | `country_proxy_keys` | `@api.depends` | filtered | — | filtered | — | — | res_bank, res_partner_bank |
| 9 | 6 | `display_name` | `@api.depends` | — | — | — | — | ✓ | discuss_channel, im_livechat_channel_member_history, product_public_category, production_lot, sale_order_line, … (+1) |
| 10 | 6 | `duration` | `@api.depends` | — | — | — | — | — | discuss_channel, hr_resume_line, maintenance, mrp_production, mrp_workcenter, … (+1) |
| 11 | 6 | `location_id` | `@api.depends` | — | — | — | — | — | repair, stock_move, stock_move_line, stock_orderpoint, stock_picking |
| 12 | 6 | `price_unit` | `@api.depends` | — | — | — | — | — | account_move_line, hr_expense, project_sale_line_employee_map, purchase_requisition, sale_order_line |
| 13 | 6 | `website_url` | `@api.depends` | — | — | — | — | ✓ | event_event, event_sponsor, event_track, forum_tag, slide_channel, … (+1) |
| 14 | 5 | `amount` | `@api.depends` | — | — | — | — | — | account_payment, account_payment_register, account_withholding_line, fleet_vehicle_log_services, l10n_ar_payment_register_withholding |
| 15 | 5 | `color` | `@api.depends` | — | — | — | — | — | gamification_goal, hr_resume_line, payment_provider, product_template, project_update |
| 16 | 5 | `complete_name` | `@api.depends` | — | — | — | — | — | analytic_plan, hr_department, product_category, stock_location, stock_package |
| 17 | 5 | `currency_rate` | `@api.depends` | — | — | — | — | ✓ | account_move_line, pos_order, purchase_order, sale_order |
| 18 | 5 | `display_name` | `@api.depends` | filtered | — | filtered | — | — | account_journal, calendar_event, hr_timesheet, l10n_in_ewaybill, res_partner |
| 19 | 5 | `display_qr_setting` | `@api.depends` | filtered | — | filtered | — | — | res_bank, res_partner_bank |
| 20 | 5 | `need_cancel_request` | `@api.depends` | — | — | — | — | — | account_move |
| 21 | 5 | `partner_id` | `@api.depends` | — | — | — | — | — | account_analytic_line, event_track_visitor, hr_timesheet, project_project, repair |
| 22 | 5 | `product_qty` | `@api.depends` | — | — | — | — | — | mrp_production, mrp_unbuild, repair, sale_order_line, stock_move |
| 23 | 5 | `qty_delivered` | `@api.depends` | — | — | — | — | — | pos_order, sale_order, sale_order_line |
| 24 | 5 | `show_delivery_date` | `@api.depends` | — | — | — | — | — | account_move |
| 25 | 5 | `state` | `@api.depends` | — | — | — | — | — | account_lock_exception, account_payment, mrp_production, mrp_workorder, project_task |
| 26 | 4 | `avatar_128` | `@api.depends` | — | — | — | — | — | discuss_channel, hr_employee, im_livechat_channel_member_history, resource_resource |
| 27 | 4 | `department_id` | `@api.depends` | — | — | — | — | — | hr_leave, hr_leave_allocation, hr_timesheet, resource |
| 28 | 4 | `description_picking` | `@api.depends` | — | — | — | — | — | stock, stock_move, stock_picking |
| 29 | 4 | `packaging_uom_id` | `@api.depends` | — | — | — | — | — | stock, stock_move |
| 30 | 4 | `product_uom_qty` | `@api.depends` | — | — | — | — | — | mrp_production, project_milestone, purchase_order_line, sale_order_line |
| 31 | 4 | `render_model` | `@api.depends` | — | — | — | — | — | card_campaign, mail_template, mailing, sms_template |
| 32 | 4 | `show_reset_to_draft_button` | `@api.depends` | filtered | — | filtered | — | — | account_move |
| 33 | 4 | `warehouse_id` | `@api.depends` | — | — | — | — | ✓ | sale_order, sale_order_line, stock_orderpoint, stock_picking |
| 34 | 3 | `amount_invoiced` | `@api.depends` | — | — | — | — | — | sale_order, sale_order_line |
| 35 | 3 | `amount_to_invoice` | `@api.depends` | — | — | — | — | — | sale_order, sale_order_line |
| 36 | 3 | `analytic_distribution` | `@api.depends` | — | — | — | — | ✓ | account_move_line, hr_expense, purchase_order_line |
| 37 | 3 | `code` | `@api.depends` | — | — | — | — | — | account_code_mapping, hr_contract_type, loyalty_rule |
| 38 | 3 | `community_menu` | `@api.depends` | — | — | — | — | — | event_event, event_type |
| 39 | 3 | `company_consistency` | `@api.constrains` | — | — | — | UserError | ✓ | account_journal, account_tax, analytic_account |
| 40 | 3 | `company_registry` | `@api.depends` | _check_vat_number, _split_vat, filtered | — | _check_vat_number, _split_vat, filtered | — | — | res_partner |
| 41 | 3 | `company_registry_placeholder` | `@api.depends` | — | — | — | — | — | company, res_partner |
| 42 | 3 | `contact_email` | `@api.depends` | — | — | — | — | — | event_booth, event_booth_registration, event_track |
| 43 | 3 | `contact_phone` | `@api.depends` | — | — | — | — | — | event_booth, event_booth_registration, event_track |
| 44 | 3 | `country_id` | `@api.depends` | — | — | — | — | — | account_tax, hr_leave_type |
| 45 | 3 | `date` | `@api.depends` | — | — | — | — | — | account_bank_statement, event_track, hr_attendance |
| 46 | 3 | `dates` | `@api.depends` | — | — | — | — | — | calendar_event, mrp_workorder, production_lot |
| 47 | 3 | `effective_date` | `@api.depends` | — | — | — | — | — | purchase_order, sale_order, stock |
| 48 | 3 | `email_from` | `@api.depends` | filtered | — | filtered | — | — | mail_test_ticket, mailing_models, test_mail_models |
| 49 | 3 | `epson_printer_ip` | `@api.onchange` | — | — | — | — | — | pos_config, pos_printer, res_config_settings |
| 50 | 3 | `expense_policy` | `@api.depends` | filtered | — | filtered | — | — | product_template |
| 51 | 3 | `fiscal_country_codes` | `@api.depends` | — | — | — | — | — | account_payment_term, partner, product |
| 52 | 3 | `has_image` | `@api.depends` | — | — | — | — | — | pos_category, pos_preset, product_tag |
| 53 | 3 | `is_storno` | `@api.depends` | — | — | — | — | — | account_move, account_move_line |
| 54 | 3 | `l10n_ro_edi_stock_enable` | `@api.depends` | — | — | — | — | — | stock_picking, stock_picking_batch |
| 55 | 3 | `location_id` | `@api.depends` | browse | — | browse | — | — | stock_move, stock_scrap |
| 56 | 3 | `margin` | `@api.depends` | — | — | — | — | — | pos_order, sale_order_line |
| 57 | 3 | `no_cyclic_dependencies` | `@api.constrains` | _has_cycle | — | _has_cycle | ValidationError | — | mrp_routing, mrp_workorder, project_task |
| 58 | 3 | `outstanding_account_id` | `@api.depends` | — | — | — | — | — | account_payment |
| 59 | 3 | `parent_id` | `@api.constrains` | _has_cycle | — | _has_cycle | ValidationError | — | forum_post, hr_department, project_task |
| 60 | 3 | `partner_id` | `@api.depends` | filtered | — | filtered | — | — | hr_timesheet, project_project, stock_move |
| 61 | 3 | `picking_ids` | `@api.depends` | — | — | — | — | — | purchase_order, sale, sale_order |
| 62 | 3 | `picking_type_id` | `@api.depends` | — | — | — | — | — | stock_move |
| 63 | 3 | `prepayment_percent` | `@api.constrains` | — | — | — | ValidationError | — | res_company, sale_order, sale_order_template |
| 64 | 3 | `price` | `@api.depends` | — | — | — | — | — | event_booth_category, event_type_ticket, product_supplierinfo |
| 65 | 3 | `product_updatable` | `@api.depends` | — | — | — | — | — | sale, sale_order_line |
| 66 | 3 | `project_id` | `@api.depends` | — | — | — | — | — | hr_timesheet, mrp_production, test_base_automation |
| 67 | 3 | `purchase_order_count` | `@api.depends` | — | — | — | — | — | mrp_production, sale_order |
| 68 | 3 | `qty_delivered_method` | `@api.depends` | — | — | — | — | — | sale_order_line |
| 69 | 3 | `repair_count` | `@api.depends` | — | — | — | — | — | production, purchase_order, sale_order |
| 70 | 3 | `sale_order_count` | `@api.depends` | — | — | — | — | — | mrp_production, purchase_order |
| 71 | 3 | `scheduled_date` | `@api.depends` | — | — | — | — | — | event_mail_registration, stock_picking, stock_picking_batch |
| 72 | 3 | `seats` | `@api.depends` | ids | — | — | — | ✓ | event_event, event_slot, event_ticket |
| 73 | 3 | `sequence` | `@api.depends` | — | — | — | — | — | account_move_line, hr_leave_accrual_plan_level, uom_uom |
| 74 | 3 | `show_fetch_in_einvoices_button` | `@api.depends` | filtered | — | filtered | — | — | account_journal |
| 75 | 3 | `show_lots_text` | `@api.depends` | — | — | — | — | — | stock_picking, stock_picking_batch |
| 76 | 3 | `show_taxable_supply_date` | `@api.depends` | filtered | — | filtered | — | — | account_move |
| 77 | 3 | `skill_ids` | `@api.depends` | — | — | — | — | — | hr_applicant, hr_employee, hr_job |
| 78 | 3 | `tax_totals` | `@api.depends_context` | — | — | — | — | ✓ | account_move, purchase_order, sale_order |
| 79 | 3 | `team_id` | `@api.depends` | — | — | — | — | ✓ | crm_iap_lead_mining_request, crm_lead, sale_order |
| 80 | 3 | `translated_product_name` | `@api.depends` | — | — | — | — | — | account_move_line, purchase_order_line, sale_order_line |
| 81 | 3 | `url` | `@api.depends` | — | — | — | — | — | base_automation, event_sponsor, website_menu |
| 82 | 3 | `website_absolute_url` | `@api.depends` | — | — | — | — | — | event_sponsor, slide_channel, slide_slide |
| 83 | 2 | `account_id` | `@api.depends` | — | — | — | — | — | account_withholding_line, hr_expense |
| 84 | 2 | `account_peppol_contact_email` | `@api.depends` | — | — | — | — | — | res_company, res_config_settings |
| 85 | 2 | `amount` | `@api.depends` | — | — | — | — | ✓ | purchase_order_line, sale_order_line |
| 86 | 2 | `amount_currency` | `@api.depends` | — | — | — | — | — | account_bank_statement_line, account_move_line |
| 87 | 2 | `amount_paid` | `@api.depends` | — | — | — | — | — | account_move, sale_order |
| 88 | 2 | `amount_to_invoice_at_date` | `@api.depends` | — | — | — | — | — | purchase_order_line, sale_order_line |
| 89 | 2 | `analytic_distribution` | `@api.depends` | filtered | — | filtered | — | ✓ | purchase_order_line, sale_order_line |
| 90 | 2 | `answers_integrity` | `@api.constrains` | — | — | — | ValidationError | — | event_quiz, slide_question |
| 91 | 2 | `apply_grid` | `@api.onchange` | grid, grid_update, order_line, state, … (+1) | — | update | ValidationError | ✓ | purchase, sale_order |
| 92 | 2 | `authorized_transaction_ids` | `@api.depends` | — | — | — | — | — | account_move, sale_order |
| 93 | 2 | `available_model_ids` | `@api.depends` | filtered | — | filtered | — | ✓ | ir_actions_server |
| 94 | 2 | `available_quantity` | `@api.depends` | — | — | — | — | — | stock_quant |
| 95 | 2 | `available_today` | `@api.depends` | — | — | — | — | — | lunch_alert, lunch_supplier |
| 96 | 2 | `bank_id` | `@api.depends` | filtered | — | filtered | — | — | l10n_latam_check, l10n_latam_payment_register_check |
| 97 | 2 | `base_amount` | `@api.depends` | — | — | — | — | — | account_withholding_line, l10n_ar_payment_register_withholding |
| 98 | 2 | `base_unit_name` | `@api.depends` | — | — | — | — | — | product_product, product_template |
| 99 | 2 | `base_unit_price` | `@api.depends` | — | — | — | — | — | product_product, product_template |
| 100 | 2 | `bom_id` | `@api.depends` | — | — | — | — | ✓ | mrp_production, mrp_unbuild |
| 101 | 2 | `booth_menu` | `@api.depends` | — | — | — | — | — | event_event, event_type |
| 102 | 2 | `can_approve` | `@api.depends` | — | — | — | — | — | hr_leave, hr_leave_allocation |
| 103 | 2 | `can_be_reinvoiced` | `@api.depends` | — | — | — | — | — | hr_expense, hr_expense_split |
| 104 | 2 | `can_publish` | `@api.depends` | — | — | — | — | — | slide_channel, slide_slide |
| 105 | 2 | `can_refuse` | `@api.depends` | — | — | — | — | — | hr_leave, hr_leave_allocation |
| 106 | 2 | `can_validate` | `@api.depends` | — | — | — | — | — | hr_leave, hr_leave_allocation |
| 107 | 2 | `cash_control` | `@api.depends` | — | — | — | — | — | pos_config, pos_session |
| 108 | 2 | `cashier` | `@api.depends` | — | — | — | — | — | pos_order, pos_payment |
| 109 | 2 | `category_recursion` | `@api.constrains` | _has_cycle | — | _has_cycle | ValidationError | — | pos_category, product_category |
| 110 | 2 | `clean_issuer_vat` | `@api.onchange` | filtered | — | filtered | — | — | l10n_latam_check, l10n_latam_payment_register_check |
| 111 | 2 | `closing_date` | `@api.constrains` | — | — | — | ValidationError | — | calendar_event, event_event |
| 112 | 2 | `co2_emission_unit` | `@api.depends` | — | — | — | — | — | fleet_vehicle, fleet_vehicle_model |
| 113 | 2 | `comodel_currency_id` | `@api.depends` | — | — | — | — | — | account_payment_register_withholding_line, account_payment_withholding_line |
| 114 | 2 | `comodel_date` | `@api.depends` | — | — | — | — | — | account_payment_register_withholding_line, account_payment_withholding_line |
| 115 | 2 | `comodel_payment_type` | `@api.depends` | — | — | — | — | — | account_payment_register_withholding_line, account_payment_withholding_line |
| 116 | 2 | `company_consistency` | `@api.constrains` | — | — | — | ValidationError | — | stock_location, stock_rule |
| 117 | 2 | `company_informations` | `@api.depends` | company_id | — | — | — | — | res_config_settings |
| 118 | 2 | `constrains_assignment_domain` | `@api.constrains` | — | — | — | ValidationError | ✓ | crm_team, crm_team_member |
| 119 | 2 | `contact_name` | `@api.depends` | — | — | — | — | — | event_booth, event_booth_registration |
| 120 | 2 | `cost_method` | `@api.depends_context` | — | — | — | — | — | product, stock_quant |
| 121 | 2 | `currency_id` | `@api.depends_context` | — | — | — | — | — | account_payment_term, project_project |
| 122 | 2 | `currency_id` | `@api.depends` | — | — | — | — | ✓ | product_combo, product_template |
| 123 | 2 | `date_closed` | `@api.depends` | — | — | — | — | — | event_registration, hr_applicant |
| 124 | 2 | `date_deadline` | `@api.depends` | — | — | — | — | — | mrp_production, stock_picking |
| 125 | 2 | `date_from_date_to` | `@api.constrains` | — | — | — | UserError | — | hr_leave_allocation, loyalty_program |
| 126 | 2 | `deadline_date` | `@api.depends` | — | — | — | — | — | stock, stock_orderpoint |
| 127 | 2 | `debit_sequence` | `@api.depends` | — | — | — | — | — | account_journal |
| 128 | 2 | `default_code` | `@api.onchange` | default_code, id | — | — | — | ✓ | product_product, product_template |
| 129 | 2 | `description` | `@api.depends` | — | — | — | — | — | event_type_ticket, hr_leave_allocation |
| 130 | 2 | `dest_address_id` | `@api.depends` | filtered | — | filtered | — | — | purchase, purchase_order |
| 131 | 2 | `display_assign_serial` | `@api.depends` | — | — | — | — | — | stock_move |
| 132 | 2 | `display_withholding` | `@api.depends` | grouped | — | grouped | — | ✓ | account_payment, account_payment_register |
| 133 | 2 | `driver_employee_id` | `@api.depends` | driver_id | — | — | — | ✓ | fleet_vehicle, fleet_vehicle_assignation_log |
| 134 | 2 | `driver_id` | `@api.depends` | — | — | — | — | — | fleet_vehicle_odometer, stock_picking_batch |
| 135 | 2 | `duplicated_order_ids` | `@api.depends` | filtered | — | filtered | — | — | purchase_order, sale_order |
| 136 | 2 | `duration_display` | `@api.depends` | — | — | — | — | — | hr_leave, hr_leave_allocation |
| 137 | 2 | `duration_expected` | `@api.depends` | — | — | — | — | — | mrp_production, mrp_workorder |
| 138 | 2 | `email` | `@api.depends` | — | — | — | — | — | event_registration, mail_group_member |
| 139 | 2 | `email_from` | `@api.depends` | — | — | — | — | — | crm_lead, test_mail_models_mail |
| 140 | 2 | `email_normalized` | `@api.depends` | — | — | — | — | — | mail_gateway_allowed, mail_group_member |
| 141 | 2 | `email_phone` | `@api.depends` | filtered | — | filtered | — | — | website_visitor |
| 142 | 2 | `employee_id` | `@api.depends` | — | — | — | — | — | hr_expense, res_partner_bank |
| 143 | 2 | `employee_overtime` | `@api.depends` | employee_id | — | — | — | — | hr_leave, hr_leave_allocation |
| 144 | 2 | `encryption` | `@api.onchange` | smtp_authentication | — | — | — | — | ir_mail_server |
| 145 | 2 | `end_date` | `@api.depends` | — | — | — | — | — | event_track, stock_picking_batch |
| 146 | 2 | `exhibitor_menu` | `@api.depends` | — | — | — | — | — | event_event, event_type |
| 147 | 2 | `expected_currency_rate` | `@api.depends` | — | — | — | — | — | account_move |
| 148 | 2 | `factor` | `@api.depends` | — | — | — | — | — | account_tax, uom_uom |
| 149 | 2 | `field_is_one_day` | `@api.depends` | — | — | — | — | — | event_event, event_track |
| 150 | 2 | `fiscal_position_id` | `@api.depends` | — | — | — | — | — | account_move, sale_order |
| 151 | 2 | `fiscal_position_id` | `@api.depends` | — | — | — | — | ✓ | account_move, sale_order |
| 152 | 2 | `force_restrictive_audit_trail` | `@api.depends` | — | — | — | — | — | company, res_company |
| 153 | 2 | `forecasted_issue` | `@api.depends` | — | — | — | — | — | mrp_production, purchase_order_line |
| 154 | 2 | `form_field_ids` | `@api.depends` | filtered, form_field_ids | form_field_ids | filtered | — | ✓ | product_document, quotation_document |
| 155 | 2 | `from_employee_id` | `@api.depends` | — | — | — | — | — | hr_expense, hr_leave |
| 156 | 2 | `full_url` | `@api.depends` | — | — | — | — | — | hr_job, spreadsheet_dashboard_share |
| 157 | 2 | `generate_lead` | `@api.depends` | — | — | — | — | — | survey_question, survey_survey |
| 158 | 2 | `grid_up` | `@api.onchange` | _get_matrix, grid, grid_product_tmpl_id, grid_update | grid, grid_update | _get_matrix | — | — | purchase, sale_order |
| 159 | 2 | `hide_reservation_method` | `@api.depends` | — | — | — | — | — | stock_picking |
| 160 | 2 | `highlight_send_button` | `@api.depends` | — | — | — | — | — | account_move |
| 161 | 2 | `im_status` | `@api.depends` | — | — | — | — | — | mail_guest, res_users |
| 162 | 2 | `image_1920` | `@api.depends` | — | — | — | — | — | event_booth_category, slide_slide |
| 163 | 2 | `incoming_picking_count` | `@api.depends` | — | — | — | — | — | purchase, purchase_order |
| 164 | 2 | `incoterm_location` | `@api.depends` | — | — | — | — | — | account_invoice, account_move |
| 165 | 2 | `invoiced` | `@api.depends` | — | — | — | — | — | declaration_of_intent, sale_order |
| 166 | 2 | `is_domestic` | `@api.depends` | — | — | — | — | — | account_tax, partner |
| 167 | 2 | `is_dropship` | `@api.depends` | — | — | — | — | — | stock, stock_move |
| 168 | 2 | `is_editable` | `@api.depends` | — | — | — | — | — | discuss_channel, l10n_in_ewaybill |
| 169 | 2 | `is_favorite` | `@api.depends_context` | — | — | — | — | — | lunch_product, spreadsheet_dashboard |
| 170 | 2 | `is_homepage` | `@api.depends` | — | — | — | — | — | website_page_properties |
| 171 | 2 | `is_locked` | `@api.depends` | — | — | — | — | — | stock_move |
| 172 | 2 | `is_manager` | `@api.depends` | — | — | — | — | — | hr_attendance, hr_attendance_overtime |
| 173 | 2 | `is_membership_multi` | `@api.depends` | is_membership_multi | is_membership_multi | — | — | ✓ | crm_team, crm_team_member |
| 174 | 2 | `is_mondialrelay` | `@api.depends` | — | — | — | — | — | delivery_carrier, res_partner |
| 175 | 2 | `is_sold_out` | `@api.depends` | — | — | — | — | — | event_slot, event_ticket |
| 176 | 2 | `issuer_vat` | `@api.depends` | filtered | — | filtered | — | — | l10n_latam_check, l10n_latam_payment_register_check |
| 177 | 2 | `journal_id` | `@api.depends` | filtered | — | filtered | — | — | account_move, sale_order |
| 178 | 2 | `journal_id` | `@api.depends` | — | — | — | — | ✓ | account_payment, payment_provider |
| 179 | 2 | `l10n_es_edi_verifactu_qr_code` | `@api.depends` | — | — | — | — | — | account_move, pos_order |
| 180 | 2 | `l10n_es_edi_verifactu_state` | `@api.depends` | — | — | — | — | — | account_move, pos_order |
| 181 | 2 | `l10n_es_edi_verifactu_warning` | `@api.depends` | — | — | — | — | — | account_move, pos_order |
| 182 | 2 | `l10n_es_tbai_state` | `@api.depends` | — | — | — | — | — | account_move, pos_order |
| 183 | 2 | `l10n_in_ewaybill_details` | `@api.depends` | — | — | — | — | — | account_move, stock_picking |
| 184 | 2 | `l10n_it_edi_doi_date` | `@api.depends` | — | — | — | — | — | account_move, sale_order |
| 185 | 2 | `l10n_it_edi_doi_id` | `@api.depends` | — | — | — | — | ✓ | account_move, sale_order |
| 186 | 2 | `l10n_it_partner_pa` | `@api.depends` | — | — | — | — | — | account_move, sale_order |
| 187 | 2 | `l10n_latam_document_number` | `@api.onchange` | filtered | — | filtered | — | — | account_move |
| 188 | 2 | `l10n_my_identification_number_placeholder` | `@api.depends` | — | — | — | — | — | res_company, res_partner |
| 189 | 2 | `l10n_ro_edi_stock_available_operation_scopes` | `@api.depends` | — | — | — | — | — | stock_picking, stock_picking_batch |
| 190 | 2 | `l10n_ro_edi_stock_current_document_state` | `@api.depends` | — | — | — | — | — | stock_picking, stock_picking_batch |
| 191 | 2 | `l10n_ro_edi_stock_current_document_uit` | `@api.depends` | — | — | — | — | — | stock_picking, stock_picking_batch |
| 192 | 2 | `l10n_ro_edi_stock_default_location_type` | `@api.depends` | — | — | — | — | — | stock_picking, stock_picking_batch |
| 193 | 2 | `l10n_ro_edi_stock_enable_amend` | `@api.depends` | — | — | — | — | — | stock_picking, stock_picking_batch |
| 194 | 2 | `l10n_ro_edi_stock_enable_fetch` | `@api.depends` | — | — | — | — | — | stock_picking, stock_picking_batch |
| 195 | 2 | `l10n_ro_edi_stock_enable_send` | `@api.depends` | — | — | — | — | — | stock_picking, stock_picking_batch |
| 196 | 2 | `l10n_ro_edi_stock_fields_readonly` | `@api.depends` | — | — | — | — | — | stock_picking, stock_picking_batch |
| 197 | 2 | `l10n_ro_edi_stock_reset_variable_selection_fields` | `@api.onchange` | l10n_ro_edi_stock_end_loc_type, l10n_ro_edi_stock_operation_scope, l10n_ro_edi_stock_start_loc_type | l10n_ro_edi_stock_end_loc_type, l10n_ro_edi_stock_operation_scope, l10n_ro_edi_stock_start_loc_type | — | — | — | stock_picking, stock_picking_batch |
| 198 | 2 | `l10n_tr_ctsp_number` | `@api.constrains` | — | — | — | ValidationError | — | account_move_line, product_product |
| 199 | 2 | `last_activity` | `@api.depends` | — | — | — | — | — | hr_employee, hr_employee_public |
| 200 | 2 | `lead_count` | `@api.depends` | ids | — | — | — | ✓ | crm_iap_lead_mining_request, event_event |
| 201 | 2 | `lead_count` | `@api.depends` | — | — | — | — | — | event_registration, website_visitor |
| 202 | 2 | `leaves` | `@api.depends` | employee_id, holiday_status_id | — | — | — | — | hr_leave, hr_leave_allocation |
| 203 | 2 | `limit_available_currency_ids` | `@api.constrains` | filtered | — | filtered | ValidationError | — | payment_provider |
| 204 | 2 | `location_dest_id` | `@api.depends` | browse | — | browse | — | — | stock_move |
| 205 | 2 | `mailing_domain` | `@api.constrains` | — | — | — | ValidationError | ✓ | mailing_filter, mailing_mailing |
| 206 | 2 | `maintenance_team_id` | `@api.depends` | — | — | — | — | — | maintenance |
| 207 | 2 | `medium_id` | `@api.depends` | — | — | — | — | ✓ | mailing, mailing_mailing |
| 208 | 2 | `message_partner_ids` | `@api.depends` | — | — | — | — | — | hr_timesheet, mail_thread |
| 209 | 2 | `move_id` | `@api.constrains` | — | — | — | ValidationError | — | account_payment |
| 210 | 2 | `move_type` | `@api.depends` | — | — | — | — | — | stock, stock_picking |
| 211 | 2 | `mrp_production_ids` | `@api.depends` | — | — | — | — | — | sale_order, stock_picking |
| 212 | 2 | `name` | `@api.depends` | — | — | — | — | ✓ | account_payment, chatbot_script_step |
| 213 | 2 | `name` | `@api.onchange` | name | name | — | — | — | l10n_latam_check, l10n_latam_payment_register_check |
| 214 | 2 | `name_placeholder` | `@api.depends` | — | — | — | — | — | account_journal, account_move |
| 215 | 2 | `name_short` | `@api.depends` | — | — | — | — | — | sale_order_line |
| 216 | 2 | `nemhandel_edi_user` | `@api.depends` | — | — | — | — | — | res_company, res_config_settings |
| 217 | 2 | `nemhandel_move_state` | `@api.depends` | — | — | — | — | — | account_move |
| 218 | 2 | `no_followup` | `@api.depends` | — | — | — | — | — | account_move, account_move_line |
| 219 | 2 | `note` | `@api.depends` | — | — | — | — | — | event_event, mail_activity_plan_template |
| 220 | 2 | `notification_type` | `@api.depends` | notification_type | notification_type | — | — | — | event_mail, event_type_mail |
| 221 | 2 | `online_payment_method_id` | `@api.depends` | — | — | — | — | — | pos_order |
| 222 | 2 | `order_deadline_passed` | `@api.depends` | — | — | — | — | — | lunch_order, lunch_supplier |
| 223 | 2 | `order_line_company_id` | `@api.constrains` | — | — | — | ValidationError | — | purchase_order, sale_order |
| 224 | 2 | `original_amounts` | `@api.depends` | — | — | — | — | — | account_payment_register_withholding_line, account_payment_withholding_line |
| 225 | 2 | `overtime_deductible` | `@api.depends` | — | — | — | — | — | hr_leave, hr_leave_allocation |
| 226 | 2 | `owner` | `@api.depends` | — | — | — | — | — | equipment |
| 227 | 2 | `partner_bank_id` | `@api.depends` | — | — | — | — | — | account_move, account_payment |
| 228 | 2 | `partner_phone` | `@api.depends` | — | — | — | — | — | event_track, project_task |
| 229 | 2 | `partner_shipping_id` | `@api.depends` | — | — | — | — | — | account_move, sale_order |
| 230 | 2 | `pattern` | `@api.constrains` | — | — | — | ValidationError | — | barcode_nomenclature, barcode_rule |
| 231 | 2 | `payment_method_line_id` | `@api.depends` | — | — | — | — | — | account_payment, hr_expense |
| 232 | 2 | `payment_receipt_title` | `@api.depends` | filtered | — | filtered | — | — | account_payment |
| 233 | 2 | `peppol_move_state` | `@api.depends` | — | — | — | — | — | account_move |
| 234 | 2 | `phone` | `@api.depends` | — | — | — | — | — | crm_lead, event_registration |
| 235 | 2 | `phone_validation` | `@api.onchange` | _phone_format, phone | phone | _phone_format | — | — | crm_lead, res_partner |
| 236 | 2 | `picked` | `@api.depends` | — | — | — | — | — | stock_move, stock_move_line |
| 237 | 2 | `pos_pricelist_id` | `@api.depends` | — | — | — | — | ✓ | res_config_settings |
| 238 | 2 | `preferred_payment_method_line_id` | `@api.depends` | — | — | — | — | — | account_move, sale_order |
| 239 | 2 | `prefix_placeholder` | `@api.depends` | — | — | — | — | ✓ | account_analytic_distribution_model, account_analytic_plan |
| 240 | 2 | `prepayment_percent` | `@api.depends` | — | — | — | — | — | sale_order, sale_order_template |
| 241 | 2 | `presence_state` | `@api.depends` | filtered | — | filtered | — | — | hr_employee |
| 242 | 2 | `price_incl` | `@api.depends` | — | — | — | — | — | event_booth_category, event_event_ticket |
| 243 | 2 | `price_reduce` | `@api.depends_context` | — | — | — | — | — | event_booth_category, event_type_ticket |
| 244 | 2 | `priority` | `@api.depends` | — | — | — | — | — | stock_move |
| 245 | 2 | `private_state_id` | `@api.onchange` | private_country_id, private_state_id | private_country_id | — | — | — | hr_employee, res_users |
| 246 | 2 | `product_id` | `@api.onchange` | filtered | — | filtered | — | — | account_move_line, product_pricelist_item |
| 247 | 2 | `product_id` | `@api.depends` | — | — | — | — | — | mrp_production, mrp_unbuild |
| 248 | 2 | `product_tooltip` | `@api.depends` | — | — | — | — | — | product_template |
| 249 | 2 | `production_count` | `@api.depends` | — | — | — | — | — | analytic_account, repair |
| 250 | 2 | `purchase_line_warn_msg` | `@api.depends` | — | — | — | — | — | account_invoice, purchase_order_line |
| 251 | 2 | `purchase_warning_text` | `@api.depends` | purchase_warning_text | purchase_warning_text | — | — | — | account_invoice, purchase_order |
| 252 | 2 | `qty_invoiced` | `@api.depends` | _prepare_qty_invoiced | — | _prepare_qty_invoiced | — | — | purchase_order_line, sale_order_line |
| 253 | 2 | `qty_invoiced_at_date` | `@api.depends` | _date_in_the_past, _prepare_qty_invoiced | — | _date_in_the_past, _prepare_qty_invoiced | — | — | purchase_order_line, sale_order_line |
| 254 | 2 | `qty_to_order_computed` | `@api.depends` | — | — | — | — | — | stock, stock_orderpoint |
| 255 | 2 | `quantity` | `@api.depends` | — | — | — | — | — | account_move_line, stock_move_line |
| 256 | 2 | `reference` | `@api.depends` | — | — | — | — | — | snailmail_letter, stock_move |
| 257 | 2 | `reference` | `@api.depends` | — | — | — | — | ✓ | stock_move |
| 258 | 2 | `refund_sequence` | `@api.depends` | — | — | — | — | — | account_journal |
| 259 | 2 | `registration_status` | `@api.depends` | filtered | — | filtered | — | — | event_registration |
| 260 | 2 | `remaining_hours` | `@api.depends` | — | — | — | — | — | project_task, test_base_automation |
| 261 | 2 | `require_payment` | `@api.depends` | — | — | — | — | — | sale_order, sale_order_template |
| 262 | 2 | `require_signature` | `@api.depends` | — | — | — | — | — | sale_order, sale_order_template |
| 263 | 2 | `res_name` | `@api.depends` | — | — | — | — | ✓ | rating, rating_rating |
| 264 | 2 | `sale_line` | `@api.depends` | — | — | — | — | — | project_task |
| 265 | 2 | `sale_line_id` | `@api.depends` | filtered | — | filtered | — | — | project_project, project_sale_line_employee_map |
| 266 | 2 | `sale_line_warn_msg` | `@api.depends` | — | — | — | — | — | account_move_line, sale_order_line |
| 267 | 2 | `sale_order_id` | `@api.depends` | — | — | — | — | — | hr_expense_split, project_task |
| 268 | 2 | `sale_warning_text` | `@api.depends` | sale_warning_text | sale_warning_text | — | — | — | account_move, sale_order |
| 269 | 2 | `scheduled_date` | `@api.depends` | filtered | — | filtered | — | ✓ | event_mail, event_mail_slot |
| 270 | 2 | `scrap_qty` | `@api.depends` | scrap_qty | scrap_qty | — | — | — | stock_scrap |
| 271 | 2 | `seats_limited` | `@api.depends` | — | — | — | — | — | event_event, event_type_ticket |
| 272 | 2 | `seats_max` | `@api.depends` | — | — | — | — | — | event_event, event_type |
| 273 | 2 | `service_tracking` | `@api.onchange` | project_id, project_template_id, service_tracking | project_id, project_template_id | — | — | — | product_product, product_template |
| 274 | 2 | `service_tracking` | `@api.depends` | filtered | — | filtered | — | — | product_template |
| 275 | 2 | `service_type` | `@api.depends` | filtered | — | filtered | — | — | product_template |
| 276 | 2 | `shipping_weight` | `@api.depends` | — | — | — | — | — | sale_order, stock_picking |
| 277 | 2 | `should_withhold_tax` | `@api.depends` | — | — | — | — | — | account_payment, account_payment_register |
| 278 | 2 | `show_allocation` | `@api.depends` | show_allocation | show_allocation | — | — | — | stock_picking, stock_picking_batch |
| 279 | 2 | `show_line_subtotals_tax_selection` | `@api.depends` | — | — | — | — | — | website |
| 280 | 2 | `show_picking_type` | `@api.depends` | — | — | — | — | — | stock, stock_picking |
| 281 | 2 | `show_qty_status_button` | `@api.depends` | — | — | — | — | — | product |
| 282 | 2 | `show_qty_update_button` | `@api.depends` | — | — | — | — | — | product |
| 283 | 2 | `show_refresh_out_einvoices_status_button` | `@api.depends` | filtered | — | filtered | — | — | account_journal |
| 284 | 2 | `slide_type` | `@api.depends` | — | — | — | — | — | slide_slide |
| 285 | 2 | `sms_id` | `@api.depends` | filtered, sms_id | sms_id | filtered | — | ✓ | mail_notification, mailing_trace |
| 286 | 2 | `stage_id` | `@api.depends` | — | — | — | — | — | crm_lead, project_task |
| 287 | 2 | `standard_price` | `@api.onchange` | standard_price | — | — | ValidationError | — | product_product, product_template |
| 288 | 2 | `suitable_payment_token_ids` | `@api.depends` | — | — | — | — | ✓ | account_payment, account_payment_register |
| 289 | 2 | `tax_amount` | `@api.depends` | — | — | — | — | ✓ | account_move, hr_expense |
| 290 | 2 | `tax_country_id` | `@api.depends` | — | — | — | — | — | purchase_order, sale_order |
| 291 | 2 | `tax_string` | `@api.depends` | — | — | — | — | — | product |
| 292 | 2 | `type` | `@api.depends` | — | — | — | — | — | l10n_in_pan_entity, pos_payment_method |
| 293 | 2 | `type` | `@api.onchange` | _origin, sales_count | — | — | — | — | product_product, product_template |
| 294 | 2 | `type_event_booth` | `@api.onchange` | invoice_policy, service_tracking | invoice_policy | — | — | — | product_product, product_template |
| 295 | 2 | `type_tax_use` | `@api.depends` | — | — | — | — | — | account_payment_register_withholding_line, account_payment_withholding_line |
| 296 | 2 | `unreserve_visible` | `@api.depends` | — | — | — | — | — | mrp_production, repair |
| 297 | 2 | `untaxed_amount_invoiced` | `@api.depends` | — | — | — | — | — | sale_order, sale_order_line |
| 298 | 2 | `uom_id` | `@api.depends` | — | — | — | — | — | hr_expense, mrp_production |
| 299 | 2 | `use_create_lots` | `@api.depends` | — | — | — | — | — | stock_picking |
| 300 | 2 | `use_electronic_payment_method` | `@api.depends` | — | — | — | — | ✓ | account_payment, account_payment_register |
| 301 | 2 | `use_existing_lots` | `@api.depends` | — | — | — | — | — | stock_picking |
| 302 | 2 | `user_company_ids` | `@api.depends` | — | — | — | — | ✓ | crm_lead, crm_team_member |
| 303 | 2 | `user_has_debug` | `@api.depends_context` | user_has_debug | user_has_debug | — | — | — | loyalty_reward, loyalty_rule |
| 304 | 2 | `user_id` | `@api.depends` | — | — | — | — | — | maintenance, sale_order |
| 305 | 2 | `valid_values` | `@api.constrains` | — | — | — | ValidationError | — | product_template_attribute_line, product_template_attribute_value |
| 306 | 2 | `warehouse_id` | `@api.depends` | — | — | — | — | — | pos_config, stock |
| 307 | 2 | `website_image_url` | `@api.depends` | — | — | — | — | ✓ | event_sponsor, event_track |
| 308 | 2 | `website_menu` | `@api.depends` | — | — | — | — | — | event_event, website_page |
| 309 | 2 | `website_url` | `@api.depends` | — | — | — | — | — | ir_actions_server, website_page |
| 310 | 2 | `withholding_hide_tax_base_account` | `@api.depends` | — | — | — | — | — | account_payment, account_payment_register |


Singleton concerns (one-of, no synergy): 2655

## Concern count by primary decorator (synergistic only)

- `@api.depends` — 271 concerns
- `@api.constrains` — 17 concerns
- `@api.onchange` — 16 concerns
- `@api.depends_context` — 6 concerns
