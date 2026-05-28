# Duplicate-route detection via codegen round-trip

From **3555** ruff-py-dto bundles, **3555** canonicalised via `ast.unparse` (the `ruff_python_codegen::Generator.unparse_suite` equivalent in Python).

- **3516** unique canonical bodies
- **36** duplicate clusters (n ≥ 2)
- **75** methods inside a duplicate cluster
- **3480** singletons

Decomposition ceiling: if every duplicate cluster collapses to one shared implementation, the implementation count drops from 3555 to 3516 — a **1.1%** reduction.

## Cluster size distribution

| cluster size | num clusters | methods covered |
| ---: | ---: | ---: |
| 2 | 33 | 66 |
| 3 | 3 | 9 |

## Top 30 duplicate clusters (by member count)

| n | function names (deduped) | families | canonical body (first 200 chars) |
| ---: | --- | --- | --- |
| 3 | `_check_account_id, _check_incoming_einvoice_notification_email, _check_ssnid` | account_journal, hr_version, project_project | `pass` |
| 3 | `_compute_website_absolute_url` | event_sponsor, slide_channel, slide_slide | `super()._compute_website_absolute_url()` |
| 3 | `_compute_qty_delivered` | sale_order, sale_order_line | `super()._compute_qty_delivered()` |
| 2 | `_compute_need_cancel_request` | account_move | `return super()._compute_need_cancel_request()` |
| 2 | `_compute_need_cancel_request` | account_move | `super()._compute_need_cancel_request()` |
| 2 | `_check_l10n_tr_ctsp_number` | account_move_line, product_product | `for record in self: ⏎     if record.l10n_tr_ctsp_number and len(record.l10n_tr_ctsp_number) > 12: ⏎         raise ValidationError(_('CTSP Number must be 12 digits or fewer.'))` |
| 2 | `_compute_allowed_uom_ids` | account_move_line, sale_order_line | `for line in self: ⏎     line.allowed_uom_ids = line.product_id.uom_id ¦ line.product_id.uom_ids` |
| 2 | `_compute_sale_line_warn_msg` | account_move_line, sale_order_line | `has_warning_group = self.env.user.has_group('sale.group_warning_sale') ⏎ for line in self: ⏎     line.sale_line_warn_msg = line.product_id.sale_line_warn_msg if has_warning_group else ''` |
| 2 | `_compute_translated_product_name` | account_move_line, purchase_order_line | `for line in self: ⏎     line.translated_product_name = line.product_id.with_context(lang=line.partner_id.lang).display_name` |
| 2 | `_compute_original_amounts` | account_payment_register_withholding_line, account_payment_withholding_line | `""" Adds a dependency to the payment amount to ensure recomputation when necessary. """ ⏎ super()._compute_original_amounts()` |
| 2 | `_compute_fiscal_country_codes` | account_payment_term, product | `for record in self: ⏎     allowed_companies = record.company_id or self.env.companies ⏎     record.fiscal_country_codes = ','.join(allowed_companies.mapped('account_fiscal_country_id.code'))` |
| 2 | `_onchange_phone_validation` | crm_lead, res_partner | `if self.phone: ⏎     self.phone = self._phone_format(fname='phone', force_format='INTERNATIONAL') or self.phone` |
| 2 | `_compute_is_membership_multi` | crm_team, crm_team_member | `multi_enabled = self.env['ir.config_parameter'].sudo().get_param('sales_team.membership_multi', False) ⏎ self.is_membership_multi = multi_enabled` |
| 2 | `_compute_notification_type` | event_mail, event_type_mail | `"""Assigns the type of template in use, if any is set.""" ⏎ self.notification_type = 'mail'` |
| 2 | `_compute_co2_emission_unit` | fleet_vehicle, fleet_vehicle_model | `for record in self: ⏎     if record.range_unit == 'km': ⏎         record.co2_emission_unit = 'g/km' ⏎     else: ⏎         record.co2_emission_unit = 'g/mi'` |
| 2 | `_compute_last_activity` | hr_employee, hr_employee_public | `for employee in self: ⏎     tz = employee.tz ⏎     if (last_presence := employee.user_id.sudo().presence_ids.last_presence): ⏎         last_activity_datetime = last_presence.replace(tzinfo=UTC).astimezone(t` |
| 2 | `_onchange_private_state_id` | hr_employee, res_users | `if self.private_state_id: ⏎     self.private_country_id = self.private_state_id.country_id` |
| 2 | `_clean_issuer_vat` | l10n_latam_check, l10n_latam_payment_register_check | `for rec in self.filtered(lambda x: x.issuer_vat and x.company_id.country_id.code): ⏎     stdnum_vat = stdnum.util.get_cc_module(rec.company_id.country_id.code, 'vat') ⏎     if hasattr(stdnum_vat, 'compact` |
| 2 | `_onchange_name` | l10n_latam_check, l10n_latam_payment_register_check | `if self.name: ⏎     self.name = self.name.zfill(8)` |
| 2 | `_compute_user_has_debug` | loyalty_reward, loyalty_rule | `self.user_has_debug = self.env.user.has_group('base.group_no_one')` |
| 2 | `_compute_render_model` | mail_template, sms_template | `for template in self: ⏎     template.render_model = template.model` |
| 2 | `_date_end_changed, _duration_changed` | mrp_workcenter | `if not self.date_end: ⏎     return ⏎ self.date_start = self.date_end - timedelta(minutes=self.duration) ⏎ self._loss_type_change()` |
| 2 | `_onchange_epson_printer_ip` | pos_config, pos_printer | `for rec in self: ⏎     if rec.epson_printer_ip: ⏎         rec.epson_printer_ip = format_epson_certified_domain(rec.epson_printer_ip)` |
| 2 | `_compute_cashier` | pos_order, pos_payment | `for order in self: ⏎     if order.employee_id: ⏎         order.cashier = order.employee_id.name ⏎     else: ⏎         order.cashier = order.user_id.name` |
| 2 | `_onchange_service_tracking` | product_product, product_template | `if self.service_tracking == 'no': ⏎     self.project_id = False ⏎     self.project_template_id = False ⏎ elif self.service_tracking == 'task_global_project': ⏎     self.project_template_id = False ⏎ elif self.s` |
| 2 | `_onchange_standard_price` | product_product, product_template | `if self.standard_price < 0: ⏎     raise ValidationError(_("The cost of a product can't be negative."))` |
| 2 | `_onchange_type_event_booth` | product_product, product_template | `if self.service_tracking == 'event_booth': ⏎     self.invoice_policy = 'order'` |
| 2 | `_compute_product_tooltip` | product_template | `super()._compute_product_tooltip()` |
| 2 | `_check_order_line_company_id` | purchase_order, sale_order | `for order in self: ⏎     invalid_companies = order.order_line.product_id.company_id.filtered(lambda c: order.company_id not in c._accessible_branches()) ⏎     if invalid_companies: ⏎         bad_products = ` |
| 2 | `_compute_currency_rate` | purchase_order, sale_order | `for order in self: ⏎     order.currency_rate = self.env['res.currency']._get_conversion_rate(from_currency=order.company_id.currency_id, to_currency=order.currency_id, company=order.company_id, date=(or` |

## Worked examples (top 5 clusters, full canonical body)

### Cluster #1 — 3 methods

**Function-name variants:** ['_check_account_id', '_check_incoming_einvoice_notification_email', '_check_ssnid']

**Family-of-occurrence:** ['account_journal', 'hr_version', 'project_project']

**Canonical body (post-`ast.unparse`):**

```python
pass
```

**One representative site:** `account/models/account_journal.py:L701`

### Cluster #2 — 3 methods

**Function-name variants:** ['_compute_website_absolute_url']

**Family-of-occurrence:** ['event_sponsor', 'slide_channel', 'slide_slide']

**Canonical body (post-`ast.unparse`):**

```python
super()._compute_website_absolute_url()
```

**One representative site:** `website_event_exhibitor/models/event_sponsor.py:L172`

### Cluster #3 — 3 methods

**Function-name variants:** ['_compute_qty_delivered']

**Family-of-occurrence:** ['sale_order', 'sale_order_line']

**Canonical body (post-`ast.unparse`):**

```python
super()._compute_qty_delivered()
```

**One representative site:** `pos_sale/models/sale_order.py:L133`

### Cluster #4 — 2 methods

**Function-name variants:** ['_compute_need_cancel_request']

**Family-of-occurrence:** ['account_move']

**Canonical body (post-`ast.unparse`):**

```python
return super()._compute_need_cancel_request()
```

**One representative site:** `l10n_vn_edi_viettel/models/account_move.py:L169`

### Cluster #5 — 2 methods

**Function-name variants:** ['_compute_need_cancel_request']

**Family-of-occurrence:** ['account_move']

**Canonical body (post-`ast.unparse`):**

```python
super()._compute_need_cancel_request()
```

**One representative site:** `l10n_tw_edi_ecpay/models/account_move.py:L169`

