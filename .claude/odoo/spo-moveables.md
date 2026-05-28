# SPO templates × moveable parts — second pass

From 3555 methods, field-name-abstracted templates yield **3072 unique templates**. **201 templates have ≥ 2 instances** (covering 546 methods); the rest are singletons.

**Moveable parts** = the slot list `[_0, _1, _2, …]` that varies between instances of the same template. **Method-emit template** = the body source with slots in place of field names + literals.

## Decomposition per opening

| opening | methods | unique templates | ≥2-instance templates | methods in those | singleton templates |
| --- | ---: | ---: | ---: | ---: | ---: |
| `iter_records_compute_from_related` | 2027 | 1778 | 111 | 264 | 1667 |
| `compute_scalar_other` | 260 | 201 | 14 | 67 | 187 |
| `iter_records_raise_on_violation` | 252 | 240 | 9 | 18 | 231 |
| `iter_records_aggregate_relation` | 211 | 187 | 16 | 40 | 171 |
| `onchange_other` | 206 | 178 | 17 | 44 | 161 |
| `super_extend` | 149 | 120 | 10 | 23 | 110 |
| `validator_other` | 138 | 121 | 8 | 22 | 113 |
| `iter_filtered_mutate` | 81 | 68 | 4 | 9 | 64 |
| `sudo_escalation_lookup` | 59 | 55 | 1 | 2 | 54 |
| `other` | 54 | 45 | 6 | 14 | 39 |
| `iter_filtered_raise_on_violation` | 37 | 36 | 0 | 0 | 36 |
| `super_delegation_pure` | 35 | 7 | 3 | 31 | 4 |
| `onchange_clear_dependent_cascade` | 25 | 17 | 1 | 9 | 16 |
| `with_context_query_shift` | 17 | 17 | 0 | 0 | 17 |
| `pass_override` | 3 | 1 | 1 | 3 | 0 |
| `state_transition_with_guard` | 1 | 1 | 0 | 0 | 1 |

## Top templates per opening (≥2 instances only)

### `iter_records_compute_from_related` (2027 methods, 111 reusable templates)

**Template #1** — 10 methods

```python
for line in self:
    line._0 = line._1._2
```

**Moveable slots** (3): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_analytic_line | `_compute_general_account_id` | `general_account_id`, `move_line_id`, `account_id` |
| account_payment_register_withholding_line | `_compute_comodel_currency_id` | `comodel_currency_id`, `payment_register_id`, `currency_id` |
| account_payment_register_withholding_line | `_compute_comodel_date` | `comodel_date`, `payment_register_id`, `payment_date` |
| account_payment_register_withholding_line | `_compute_comodel_payment_type` | `comodel_payment_type`, `payment_register_id`, `payment_type` |

**Template #2** — 5 methods

```python
for channel in self:
    channel._0 = channel._1._2
```

**Moveable slots** (3): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| discuss_channel | `_compute_channel_partner_ids` | `channel_partner_ids`, `channel_member_ids`, `partner_id` |
| discuss_channel | `_compute_livechat_agent_partner_ids` | `livechat_agent_partner_ids`, `livechat_agent_history_ids`, `partner_id` |
| discuss_channel | `_compute_livechat_bot_partner_ids` | `livechat_bot_partner_ids`, `livechat_bot_history_ids`, `partner_id` |
| discuss_channel | `_compute_livechat_customer_partner_ids` | `livechat_customer_partner_ids`, `livechat_customer_history_ids`, `partner_id` |

**Template #3** — 4 methods

```python
for move in self:
    move._0 = move._1()
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_move | `_compute_alerts` | `alerts`, `_get_alerts` |
| account_move | `_compute_edi_show_force_cancel_button` | `edi_show_force_cancel_button`, `_can_force_cancel` |
| account_move | `_compute_need_cancel_request` | `need_cancel_request`, `_need_cancel_request` |
| account_move | `_compute_quick_encoding_vals` | `quick_encoding_vals`, `_get_quick_edit_suggestions` |

**Template #4** — 4 methods

```python
for record in self:
    record._0 = record._1._0
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_move_line | `_compute_l10n_tr_ctsp_number` | `l10n_tr_ctsp_number`, `product_id` |
| res_config_settings | `_compute_account_peppol_contact_email` | `account_peppol_contact_email`, `company_id` |
| slide_slide | `_compute_can_publish` | `can_publish`, `channel_id` |
| stock_picking | `_compute_move_type` | `move_type`, `picking_type_id` |

**Template #5** — 4 methods

```python
ignore_exceptions = bool(self._0._1._2(_LIT0, False))
for company in self:
    company._3 = company._4(_LIT1, ignore_exceptions)
```

**Moveable slots** (5): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| company | `_compute_user_fiscalyear_lock_date` | `env`, `context`, `get`, `user_fiscalyear_lock_date`, `_get_user_lock_date` |
| company | `_compute_user_purchase_lock_date` | `env`, `context`, `get`, `user_purchase_lock_date`, `_get_user_lock_date` |
| company | `_compute_user_sale_lock_date` | `env`, `context`, `get`, `user_sale_lock_date`, `_get_user_lock_date` |
| company | `_compute_user_tax_lock_date` | `env`, `context`, `get`, `user_tax_lock_date`, `_get_user_lock_date` |

### `compute_scalar_other` (260 methods, 14 reusable templates)

**Template #1** — 16 methods

```python
self._0([_LIT0])
```

**Moveable slots** (1): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| fleet_vehicle | `_compute_category` | `_load_fields_from_model` |
| fleet_vehicle | `_compute_co2` | `_load_fields_from_model` |
| fleet_vehicle | `_compute_co2_standard` | `_load_fields_from_model` |
| fleet_vehicle | `_compute_color` | `_load_fields_from_model` |

**Template #2** — 11 methods

```python
self._0(_LIT0)
```

**Moveable slots** (1): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| event_sponsor | `_compute_email` | `_synchronize_with_partner` |
| event_sponsor | `_compute_image_512` | `_synchronize_with_partner` |
| event_sponsor | `_compute_name` | `_synchronize_with_partner` |
| event_sponsor | `_compute_phone` | `_synchronize_with_partner` |

**Template #3** — 8 methods

```python
self._0 = False
```

**Moveable slots** (1): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_journal | `_compute_show_fetch_in_einvoices_button` | `show_fetch_in_einvoices_button` |
| account_journal | `_compute_show_refresh_out_einvoices_status_button` | `show_refresh_out_einvoices_status_button` |
| account_move_line | `_compute_l10n_gr_edi_cls_vat` | `l10n_gr_edi_cls_vat` |
| account_move_line | `_compute_l10n_gr_edi_detail_type` | `l10n_gr_edi_detail_type` |

**Template #4** — 8 methods

```python
_LIT0
super()._0()
```

**Moveable slots** (1): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_payment | `_compute_outstanding_account_id` | `_compute_outstanding_account_id` |
| account_payment_register_withholding_line | `_compute_original_amounts` | `_compute_original_amounts` |
| account_payment_withholding_line | `_compute_original_amounts` | `_compute_original_amounts` |
| sale_order_line | `_compute_name` | `_compute_name` |

**Template #5** — 3 methods

```python
self._0 = _LIT0
```

**Moveable slots** (1): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| data_recycle_model | `_compute_domain` | `domain` |
| res_bank | `_compute_country_proxy_keys` | `country_proxy_keys` |
| stock_orderpoint | `_compute_days_to_order` | `days_to_order` |

### `iter_records_raise_on_violation` (252 methods, 9 reusable templates)

**Template #1** — 2 methods

```python
for record in self:
    if record._0 and len(record._0) > _LIT0:
        raise ValidationError(_(_LIT1))
```

**Moveable slots** (1): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_move_line | `_check_l10n_tr_ctsp_number` | `l10n_tr_ctsp_number` |
| product_product | `_check_l10n_tr_ctsp_number` | `l10n_tr_ctsp_number` |

**Template #2** — 2 methods

```python
for record in self:
    if record._0 == _LIT0 and (not record._1):
        raise ValidationError(_(_LIT1))
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_report | `_validate_availability_condition` | `availability_condition`, `country_id` |
| pos_printer | `_constrains_epson_printer_ip` | `printer_type`, `epson_printer_ip` |

**Template #3** — 2 methods

```python
for member in self:
    try:
        domain = literal_eval(member._0 or _LIT0)
        if domain:
            self._1[_LIT1]._2(domain, limit=_LIT2)
    except Exception:
        raise exceptions._3(_(_LIT3, user=member._4._5, team=member._6._5))
```

**Moveable slots** (7): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| crm_team_member | `_constrains_assignment_domain` | `assignment_domain`, `env`, `search`, `ValidationError`, `user_id`, `name`, `crm_team_id` |
| crm_team_member | `_constrains_assignment_domain_preferred` | `assignment_domain_preferred`, `env`, `search`, `ValidationError`, `user_id`, `name`, `crm_team_id` |

**Template #4** — 2 methods

```python
for server in self:
    if server._0 == _LIT0 and (not server._1):
        raise UserError(_(_LIT1, server._2))
```

**Moveable slots** (3): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| fetchmail_server | `_check_use_google_gmail_service` | `server_type`, `is_ssl`, `name` |
| fetchmail_server | `_check_use_microsoft_outlook_service` | `server_type`, `is_ssl`, `name` |

**Template #5** — 2 methods

```python
_LIT0
for provider in self:
    if provider._0 == _LIT1 and provider._1():
        raise ValidationError(_(_LIT2))
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| payment_provider | `_check_onboarding_of_enabled_provider_is_completed` | `state`, `_stripe_onboarding_is_ongoing` |
| payment_provider | `_check_state_of_connected_account_is_never_test` | `state`, `_stripe_has_connected_account` |

### `iter_records_aggregate_relation` (211 methods, 16 reusable templates)

**Template #1** — 4 methods

```python
for account in self:
    account._0 = len(account._1)
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_move | `_compute_wip_production_count` | `wip_production_count`, `wip_production_ids` |
| analytic_account | `_compute_bom_count` | `bom_count`, `bom_ids` |
| analytic_account | `_compute_production_count` | `production_count`, `production_ids` |
| mrp_production | `_compute_wip_move_count` | `wip_move_count`, `wip_move_ids` |

**Template #2** — 4 methods

```python
for record in self:
    record._0 = len(record._1)
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| crm_lead | `_compute_registration_count` | `registration_count`, `registration_ids` |
| event_registration | `_compute_lead_count` | `lead_count`, `lead_ids` |
| product_template_attribute_line | `_compute_value_count` | `value_count`, `value_ids` |
| website_blog | `_compute_blog_post_count` | `blog_post_count`, `blog_post_ids` |

**Template #3** — 3 methods

```python
for move in self:
    move._0 = len(move._1)
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_move | `_compute_adjusting_entries_count` | `adjusting_entries_count`, `adjusting_entries_move_ids` |
| account_move | `_compute_adjusting_entry_origin_moves_count` | `adjusting_entry_origin_moves_count`, `adjusting_entry_origin_move_ids` |
| stock_move | `_compute_move_lines_count` | `move_lines_count`, `move_line_ids` |

**Template #4** — 3 methods

```python
for plan in self:
    plan._0 = len(plan._1)
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| analytic_plan | `_compute_analytic_account_count` | `account_count`, `account_ids` |
| analytic_plan | `_compute_children_count` | `children_count`, `children_ids` |
| mail_activity_plan | `_compute_steps_count` | `steps_count`, `template_ids` |

**Template #5** — 3 methods

```python
for production in self:
    production._0 = len(production._1())
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| mrp_production | `_compute_mrp_production_child_count` | `mrp_production_child_count`, `_get_children` |
| mrp_production | `_compute_mrp_production_source_count` | `mrp_production_source_count`, `_get_sources` |
| mrp_production | `_compute_purchase_order_count` | `purchase_order_count`, `_get_purchase_orders` |

### `onchange_other` (206 methods, 17 reusable templates)

**Template #1** — 5 methods

```python
if self._0 == _LIT0:
    self._1 = _LIT1
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| delivery_carrier | `_onchange_integration_level` | `integration_level`, `invoice_policy` |
| product_product | `_onchange_type_event_booth` | `service_tracking`, `invoice_policy` |
| product_template | `_onchange_type_event` | `service_tracking`, `invoice_policy` |
| product_template | `_onchange_type_event_booth` | `service_tracking`, `invoice_policy` |

**Template #2** — 4 methods

```python
if self._0:
    self._1 = self._0._2
```

**Moveable slots** (3): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| fleet_vehicle_odometer | `_onchange_vehicle` | `vehicle_id`, `unit`, `odometer_unit` |
| hr_employee | `_onchange_private_state_id` | `private_state_id`, `private_country_id`, `country_id` |
| res_config_settings | `_onchange_timesheet_task_id` | `leave_timesheet_task_id`, `internal_project_id`, `project_id` |
| res_users | `_onchange_private_state_id` | `private_state_id`, `private_country_id`, `country_id` |

**Template #3** — 4 methods

```python
self._0()
```

**Moveable slots** (1): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| mrp_production | `_onchange_qty_producing` | `_change_producing` |
| pos_order | `_onchange_amount_all` | `_compute_prices` |
| product_pricelist_item | `_onchange_validity_period` | `_check_date_range` |
| spreadsheet_mixin | `_onchange_data_` | `_check_spreadsheet_data` |

**Template #4** — 4 methods

```python
for employee in self._0:
    if employee._1._2(_LIT0):
        self._0 -= employee
    elif employee in self._3:
        self._3 -= employee
    elif employee in self._4:
        self._4 -= employee
```

**Moveable slots** (5): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| pos_config | `_onchange_basic_employee_ids` | `basic_employee_ids`, `user_id`, `_has_group`, `advanced_employee_ids`, `minimal_employee_ids` |
| pos_config | `_onchange_minimal_employee_ids` | `minimal_employee_ids`, `user_id`, `_has_group`, `basic_employee_ids`, `advanced_employee_ids` |
| res_config_settings | `_onchange_basic_employee_ids` | `pos_basic_employee_ids`, `user_id`, `_has_group`, `pos_advanced_employee_ids`, `pos_minimal_employee_ids` |
| res_config_settings | `_onchange_minimal_employee_ids` | `pos_minimal_employee_ids`, `user_id`, `_has_group`, `pos_basic_employee_ids`, `pos_advanced_employee_ids` |

**Template #5** — 3 methods

```python
if self._0:
    self._1 = self._0._2._3
```

**Moveable slots** (4): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| pos_order | `_onchange_partner_id` | `partner_id`, `pricelist_id`, `property_product_pricelist`, `id` |
| resource_resource | `_onchange_company_id` | `company_id`, `calendar_id`, `resource_calendar_id`, `id` |
| stock_orderpoint | `_onchange_product_id` | `product_id`, `product_uom`, `uom_id`, `id` |

### `super_extend` (149 methods, 10 reusable templates)

**Template #1** — 3 methods

```python
super()._0()
for move in self:
    if move._1 == _LIT0:
        move._2 = move._3()
```

**Moveable slots** (4): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_move | `_compute_show_delivery_date` | `_compute_show_delivery_date`, `country_code`, `show_delivery_date`, `is_sale_document` |
| account_move | `_compute_show_delivery_date` | `_compute_show_delivery_date`, `country_code`, `show_delivery_date`, `is_sale_document` |
| account_move | `_compute_show_delivery_date` | `_compute_show_delivery_date`, `country_code`, `show_delivery_date`, `is_sale_document` |

**Template #2** — 3 methods

```python
super()._0()
for order in self._1(_LIT0):
    order._2 = order._3._2
```

**Moveable slots** (4): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| sale_order | `_compute_journal_id` | `_compute_journal_id`, `filtered`, `journal_id`, `sale_order_template_id` |
| sale_order | `_compute_require_payment` | `_compute_require_payment`, `filtered`, `require_payment`, `sale_order_template_id` |
| sale_order | `_compute_require_signature` | `_compute_require_signature`, `filtered`, `require_signature`, `sale_order_template_id` |

**Template #3** — 3 methods

```python
super()._0()
for move in self:
    if move._1:
        move._2 = move._1._3
```

**Moveable slots** (4): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| stock | `_compute_packaging_uom_id` | `_compute_packaging_uom_id`, `sale_line_id`, `packaging_uom_id`, `product_uom_id` |
| stock_move | `_compute_packaging_uom_id` | `_compute_packaging_uom_id`, `production_id`, `packaging_uom_id`, `product_uom_id` |
| stock_move | `_compute_packaging_uom_id` | `_compute_packaging_uom_id`, `purchase_line_id`, `packaging_uom_id`, `product_uom_id` |

**Template #4** — 2 methods

```python
super()._0()
for journal in self:
    if journal._1:
        journal._2 = False
```

**Moveable slots** (3): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_journal | `_compute_debit_sequence` | `_compute_debit_sequence`, `l10n_latam_use_documents`, `debit_sequence` |
| account_journal | `_compute_refund_sequence` | `_compute_refund_sequence`, `l10n_latam_use_documents`, `refund_sequence` |

**Template #5** — 2 methods

```python
super()._0()
for move in self:
    if move._1:
        move._2 = False
```

**Moveable slots** (3): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_move | `_compute_show_reset_to_draft_button` | `_compute_show_reset_to_draft_button`, `l10n_es_tbai_chain_index`, `show_reset_to_draft_button` |
| account_move | `_compute_show_reset_to_draft_button` | `_compute_show_reset_to_draft_button`, `l10n_hr_fiscalization_status`, `show_reset_to_draft_button` |

### `validator_other` (138 methods, 8 reusable templates)

**Template #1** — 6 methods

```python
if self._0():
    raise ValidationError(_(_LIT0))
```

**Moveable slots** (1): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_account | `_check_parent_not_circular` | `_has_cycle` |
| forum_post | `_check_parent_id` | `_has_cycle` |
| hr_department | `_check_parent_id` | `_has_cycle` |
| pos_category | `_check_category_recursion` | `_has_cycle` |

**Template #2** — 3 methods

```python
if self._0(_LIT0):
    raise ValidationError(_(_LIT1))
```

**Moveable slots** (1): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| mrp_routing | `_check_no_cyclic_dependencies` | `_has_cycle` |
| mrp_workorder | `_check_no_cyclic_dependencies` | `_has_cycle` |
| project_task | `_check_no_cyclic_dependencies` | `_has_cycle` |

**Template #3** — 3 methods

```python
if any((record._0 == _LIT0 and record._1._2._3 != _LIT1 for record in self)):
    raise UserError(_(_LIT2))
```

**Moveable slots** (4): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| pos_payment_method | `_check_pine_labs_terminal` | `use_payment_terminal`, `company_id`, `currency_id`, `name` |
| pos_payment_method | `_check_qfpay_terminal` | `use_payment_terminal`, `company_id`, `currency_id`, `name` |
| pos_payment_method | `_check_razorpay_terminal` | `use_payment_terminal`, `company_id`, `currency_id`, `name` |

**Template #4** — 2 methods

```python
invalid_products = self._0(lambda product: product._1 and (not product._2._1))
if invalid_products:
    raise UserError(_(_LIT0, _LIT1._3(invalid_products._2._4(_LIT2))))
```

**Moveable slots** (5): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| lunch_product | `_check_active_categories` | `filtered`, `active`, `category_id`, `join`, `mapped` |
| lunch_product | `_check_active_suppliers` | `filtered`, `active`, `supplier_id`, `join`, `mapped` |

**Template #5** — 2 methods

```python
if any((group._0 and (not group._1) for group in self)):
    raise ValidationError(_(_LIT0))
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| mail_group | `_check_moderation_guidelines` | `moderation_guidelines`, `moderation_guidelines_msg` |
| mail_group | `_check_moderation_notify` | `moderation_notify`, `moderation_notify_msg` |

### `iter_filtered_mutate` (81 methods, 4 reusable templates)

**Template #1** — 3 methods

```python
for source in self._0(lambda r: r._1 and (not r._2)):
    source._2 = source._1._3
```

**Moveable slots** (4): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| test_mail_feature_models | `_compute_customer_email` | `filtered`, `customer_id`, `customer_email`, `email_formatted` |
| test_mail_feature_models | `_compute_customer_phone` | `filtered`, `customer_id`, `customer_phone`, `phone` |
| test_mail_models | `_compute_email_from` | `filtered`, `customer_id`, `email_from`, `email_formatted` |

**Template #2** — 2 methods

```python
for move in self._0(lambda move: move._1 == _LIT0):
    move._2 = move._2 or _LIT1
```

**Moveable slots** (3): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_move | `_compute_l10n_es_edi_facturae_reason_code` | `filtered`, `country_code`, `l10n_es_edi_facturae_reason_code` |
| account_move | `_compute_l10n_es_payment_means` | `filtered`, `country_code`, `l10n_es_payment_means` |

**Template #3** — 2 methods

```python
for rec in self._0(lambda x: x._1 and x._2._3._4):
    stdnum_vat = stdnum._5._6(rec._2._3._4, _LIT0)
    if hasattr(stdnum_vat, _LIT1):
        rec._1 = stdnum_vat._7(rec._1)
```

**Moveable slots** (8): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| l10n_latam_check | `_clean_issuer_vat` | `filtered`, `issuer_vat`, `company_id`, `country_id`, `code`, `util`, `get_cc_module`, `compact` |
| l10n_latam_payment_register_check | `_clean_issuer_vat` | `filtered`, `issuer_vat`, `company_id`, `country_id`, `code`, `util`, `get_cc_module`, `compact` |

**Template #4** — 2 methods

```python
_LIT0
for calendar in self._0(lambda c: not c._1):
    calendar._2 = float_round(calendar._3(), precision_digits=_LIT1)
```

**Moveable slots** (4): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| resource_calendar | `_compute_hours_per_day` | `filtered`, `flexible_hours`, `hours_per_day`, `_get_hours_per_day` |
| resource_calendar | `_compute_hours_per_week` | `filtered`, `flexible_hours`, `hours_per_week`, `_get_hours_per_week` |

### `sudo_escalation_lookup` (59 methods, 1 reusable templates)

**Template #1** — 2 methods

```python
multi_enabled = self._0[_LIT0]._1()._2(_LIT1, False)
self._3 = multi_enabled
```

**Moveable slots** (4): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| crm_team | `_compute_is_membership_multi` | `env`, `sudo`, `get_param`, `is_membership_multi` |
| crm_team_member | `_compute_is_membership_multi` | `env`, `sudo`, `get_param`, `is_membership_multi` |

### `other` (54 methods, 6 reusable templates)

**Template #1** — 3 methods

```python
self._0._1()
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_payment_method | `_ensure_unique_name_for_journal` | `journal_id`, `_check_payment_method_line_ids_multiplicity` |
| account_report | `_validate_groupby` | `expression_ids`, `_validate_engine` |
| company | `onchange_vat` | `partner_id`, `onchange_vat` |

**Template #2** — 3 methods

```python
if self._0:
    self._1 = True
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_tax | `onchange_price_include` | `price_include`, `include_base_amount` |
| res_config_settings | `onchange_analytic_accounting` | `group_analytic_accounting`, `module_account_accountant` |
| res_config_settings | `onchange_module_account_budget` | `module_account_budget`, `group_analytic_accounting` |

**Template #3** — 2 methods

```python
_LIT0
if self._0 == _LIT1:
    self._1 = _LIT2
    self._2 = True
    self._3 = _LIT3
else:
    self._4 = False
    self._5 = False
    self._6 = False
    super()._7()
```

**Moveable slots** (8): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| fetchmail_server | `onchange_server_type` | `server_type`, `server`, `is_ssl`, `port`, `google_gmail_refresh_token`, `google_gmail_access_token`, `google_gmail_access_token_expiration`, `onchange_server_type` |
| fetchmail_server | `onchange_server_type` | `server_type`, `server`, `is_ssl`, `port`, `microsoft_outlook_refresh_token`, `microsoft_outlook_access_token`, `microsoft_outlook_access_token_expiration`, `onchange_server_type` |

**Template #4** — 2 methods

```python
_LIT0
if self._0 == _LIT1:
    self._1 = self._2
```

**Moveable slots** (3): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| ir_mail_server | `_on_change_smtp_user_gmail` | `smtp_authentication`, `from_filter`, `smtp_user` |
| ir_mail_server | `_on_change_smtp_user_outlook` | `smtp_authentication`, `from_filter`, `smtp_user` |

**Template #5** — 2 methods

```python
if not self._0:
    return
self._1 = self._0 - timedelta(minutes=self._2)
self._3()
```

**Moveable slots** (4): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| mrp_workcenter | `_date_end_changed` | `date_end`, `date_start`, `duration`, `_loss_type_change` |
| mrp_workcenter | `_duration_changed` | `date_end`, `date_start`, `duration`, `_loss_type_change` |

### `super_delegation_pure` (35 methods, 3 reusable templates)

**Template #1** — 24 methods

```python
super()._0()
```

**Moveable slots** (1): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_move | `_compute_expected_currency_rate` | `_compute_expected_currency_rate` |
| account_move | `_compute_need_cancel_request` | `_compute_need_cancel_request` |
| account_move | `_compute_need_cancel_request` | `_compute_need_cancel_request` |
| event_sponsor | `_compute_website_absolute_url` | `_compute_website_absolute_url` |

**Template #2** — 5 methods

```python
return super()._0()
```

**Moveable slots** (1): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_move | `_compute_need_cancel_request` | `_compute_need_cancel_request` |
| account_move | `_compute_need_cancel_request` | `_compute_need_cancel_request` |
| res_config_settings | `_compute_active_provider_id` | `_compute_active_provider_id` |
| res_config_settings | `_compute_has_enabled_provider` | `_compute_has_enabled_provider` |

**Template #3** — 2 methods

```python
super(AccountAnalyticLine, self._0(lambda t: t._1()))._2()
```

**Moveable slots** (3): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| hr_timesheet | `_compute_partner_id` | `filtered`, `_is_not_billed`, `_compute_partner_id` |
| hr_timesheet | `_compute_project_id` | `filtered`, `_is_not_billed`, `_compute_project_id` |

### `onchange_clear_dependent_cascade` (25 methods, 1 reusable templates)

**Template #1** — 9 methods

```python
if not self._0:
    self._1 = False
```

**Moveable slots** (2): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| delivery_carrier | `_onchange_can_generate_return` | `can_generate_return`, `return_label_on_delivery` |
| delivery_carrier | `_onchange_return_label_on_delivery` | `return_label_on_delivery`, `get_return_label_from_portal` |
| hr_employee | `_onchange_contract_date_start` | `contract_date_start`, `contract_date_end` |
| product_template | `_onchange_sale_ok` | `sale_ok`, `available_in_pos` |

### `pass_override` (3 methods, 1 reusable templates)

**Template #1** — 3 methods

```python
pass
```

**Moveable slots** (0): per-instance parameter tuples →

| family | fn | slot values |
| --- | --- | --- |
| account_journal | `_check_incoming_einvoice_notification_email` |  |
| hr_version | `_check_ssnid` |  |
| project_project | `_check_account_id` |  |

