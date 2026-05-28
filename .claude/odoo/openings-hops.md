# Odoo openings × hop chains — the Elixir shape

**From** 3555 method bodies (priority-classified, first-match wins). **Hop chains** are dotted-attribute paths rooted at recordset names (`self`, `record`, `line`, `move`, …).

## Opening distribution

| opening | methods | most common 1st hop | depth-1 | depth-2 | depth-3 | depth-≥4 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `iter_records_compute_from_related` | 2027 | `env` (594) | 3044 | 939 | 261 | 21 |
| `compute_scalar_other` | 260 | `filtered` (143) | 460 | 74 | 45 | 2 |
| `iter_records_raise_on_violation` | 252 | `env` (65) | 257 | 119 | 14 | 3 |
| `iter_records_aggregate_relation` | 211 | `env` (38) | 333 | 108 | 20 | 4 |
| `onchange_other` | 206 | `env` (36) | 433 | 142 | 26 | 1 |
| `super_extend` | 149 | `filtered` (49) | 265 | 79 | 21 | 4 |
| `validator_other` | 138 | `env` (52) | 183 | 39 | 31 | 0 |
| `iter_filtered_mutate` | 81 | `filtered` (78) | 178 | 36 | 9 | 0 |
| `sudo_escalation_lookup` | 59 | `env` (40) | 135 | 44 | 11 | 2 |
| `other` | 54 | `product_id` (13) | 128 | 27 | 10 | 0 |
| `iter_filtered_raise_on_violation` | 37 | `filtered` (37) | 51 | 12 | 1 | 0 |
| `super_delegation_pure` | 35 | `filtered` (4) | 4 | 1 | 0 | 0 |
| `onchange_clear_dependent_cascade` | 25 | `env` (7) | 60 | 7 | 7 | 0 |
| `with_context_query_shift` | 17 | `env` (9) | 38 | 19 | 10 | 0 |
| `pass_override` | 3 | — | 0 | 0 | 0 | 0 |
| `state_transition_with_guard` | 1 | `parent_id` (1) | 4 | 1 | 1 | 0 |

## Per-opening: top hop chains (correlated 1st → 2nd → 3rd → 4th)

### `iter_records_compute_from_related` (2027 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 314 | `env` |
| 115 | `ids` |
| 61 | `id` |
| 55 | `state` |
| 49 | `env.context.get` |
| 46 | `company_id` |
| 40 | `name` |
| 40 | `country_code` |
| 36 | `currency_id` |
| 31 | `env.user.has_group` |

**Sample Elixir emission for top chain:**

```elixir
def _compute_account_root(record) do
  record |> :env
end
```

### `compute_scalar_other` (260 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 143 | `filtered` |
| 46 | `env` |
| 16 | `_load_fields_from_model` |
| 9 | `grouped` |
| 9 | `ids` |
| 7 | `env.context.get` |
| 7 | `id` |
| 7 | `_compute_template_field_from_variant_field` |
| 5 | `env.cr.execute` |
| 4 | `env.ref` |

**Sample Elixir emission for top chain:**

```elixir
def _compute_account_group(record) do
  record |> :filtered
end
```

### `iter_records_raise_on_violation` (252 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 34 | `env` |
| 26 | `env._` |
| 11 | `name` |
| 10 | `company_id` |
| 6 | `id` |
| 5 | `amount` |
| 4 | `amount_type` |
| 4 | `search` |
| 3 | `code` |
| 3 | `type` |

**Sample Elixir emission for top chain:**

```elixir
def _check_account_code(record) do
  record |> :env
end
```

### `iter_records_aggregate_relation` (211 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 22 | `env` |
| 15 | `ids` |
| 8 | `state` |
| 7 | `mapped` |
| 4 | `with_context` |
| 4 | `picking_ids` |
| 4 | `product_uom_qty` |
| 4 | `env.companies` |
| 4 | `picking_ids.filtered` |
| 4 | `order_line.mapped` |

**Sample Elixir emission for top chain:**

```elixir
def _compute_tds_tcs_features(record) do
  record |> :env
end
```

### `onchange_other` (206 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 18 | `env` |
| 11 | `name` |
| 8 | `product_id` |
| 7 | `filtered` |
| 7 | `service_tracking` |
| 6 | `state` |
| 5 | `company_id.id` |
| 4 | `env.context.get` |
| 4 | `trigger` |
| 4 | `_phone_format` |

**Sample Elixir emission for top chain:**

```elixir
def _onchange_account_type(record) do
  record |> :env
end
```

### `super_extend` (149 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 49 | `filtered` |
| 13 | `env` |
| 8 | `env.context.get` |
| 8 | `display_name` |
| 5 | `id` |
| 5 | `show_delivery_date` |
| 5 | `country_code` |
| 5 | `show_reset_to_draft_button` |
| 5 | `name` |
| 4 | `is_sale_document` |

**Sample Elixir emission for top chain:**

```elixir
def _compute_incoterm_location(record) do
  record |> :filtered
end
```

### `validator_other` (138 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 39 | `filtered` |
| 20 | `env` |
| 12 | `ids` |
| 10 | `_has_cycle` |
| 8 | `env.cr.execute` |
| 8 | `env._` |
| 4 | `company_id` |
| 4 | `env.context.get` |
| 3 | `env.cr.fetchone` |
| 3 | `flush_model` |

**Sample Elixir emission for top chain:**

```elixir
def _check_account_is_bank_journal_bank_account(record) do
  record |> :filtered
end
```

### `iter_filtered_mutate` (81 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 78 | `filtered` |
| 14 | `env` |
| 4 | `name` |
| 3 | `partner_id` |
| 3 | `issuer_vat` |
| 2 | `country_code` |
| 2 | `l10n_latam_available_document_type_ids._origin` |
| 2 | `l10n_latam_document_type_id` |
| 2 | `company_id.id` |
| 2 | `email_normalized` |

**Sample Elixir emission for top chain:**

```elixir
def _compute_internal_index(record) do
  record |> :filtered
end
```

### `sudo_escalation_lookup` (59 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 25 | `env` |
| 19 | `sudo` |
| 15 | `filtered` |
| 8 | `ids` |
| 7 | `company_id` |
| 5 | `env._` |
| 4 | `location_id` |
| 4 | `product_id.tracking` |
| 3 | `id` |
| 3 | `lot_id` |

**Sample Elixir emission for top chain:**

```elixir
def _check_company_consistency(record) do
  record |> :env
end
```

### `other` (54 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 5 | `filtered` |
| 4 | `display_type` |
| 3 | `_conditional_add_to_compute` |
| 3 | `product_id` |
| 3 | `port` |
| 3 | `is_ssl` |
| 3 | `date_start` |
| 3 | `date_end` |
| 3 | `_loss_type_change` |
| 3 | `duration` |

**Sample Elixir emission for top chain:**

```elixir
def on_change_unit_amount(record) do
  record |> :filtered
end
```

### `iter_filtered_raise_on_violation` (37 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 37 | `filtered` |
| 4 | `env._` |
| 3 | `env` |
| 3 | `env.ref` |
| 2 | `company_id` |
| 1 | `id` |
| 1 | `l10n_latam_document_type_id.internal_type` |
| 1 | `move_type` |
| 1 | `partner_id.l10n_ar_afip_responsibility_type_id.code` |
| 1 | `journal_id.l10n_ar_afip_pos_system` |

**Sample Elixir emission for top chain:**

```elixir
def _check_auto_post_draft_entries(record) do
  record |> :filtered
end
```

### `super_delegation_pure` (35 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 4 | `filtered` |
| 1 | `expense_id.payment_mode` |

**Sample Elixir emission for top chain:**

```elixir
def _check_payable_receivable(record) do
  record |> :filtered
end
```

### `onchange_clear_dependent_cascade` (25 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 4 | `ids` |
| 3 | `name` |
| 2 | `env.cr.execute` |
| 2 | `return_label_on_delivery` |
| 2 | `image_1920` |
| 2 | `_get_projects_for_invoice_status` |
| 2 | `env` |
| 2 | `group_lot_on_delivery_slip` |
| 2 | `group_expiry_date_on_delivery_slip` |
| 2 | `module_product_expiry` |

**Sample Elixir emission for top chain:**

```elixir
def _onchange_journal_id(record) do
  record |> :ids
end
```

### `with_context_query_shift` (17 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 6 | `filtered` |
| 5 | `env` |
| 3 | `with_context` |
| 3 | `ids` |
| 2 | `env.context.get` |
| 1 | `invoice_line_ids.filtered` |
| 1 | `l10n_in_tcs_feature` |
| 1 | `_get_l10n_in_invalid_tax_lines` |
| 1 | `_get_l10n_in_tds_tcs_applicable_sections` |
| 1 | `tax_ids` |

**Sample Elixir emission for top chain:**

```elixir
def _compute_l10n_in_warning(record) do
  record |> :filtered
end
```

### `pass_override` (3 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |

**Sample Elixir emission for top chain:**

### `state_transition_with_guard` (1 methods)

**Top hop chains by frequency:**

| n | chain |
| ---: | --- |
| 1 | `parent_id.project_id.id` |
| 1 | `env.user` |
| 1 | `display_in_project` |
| 1 | `project_id` |
| 1 | `user_ids` |
| 1 | `state` |

**Sample Elixir emission for top chain:**

```elixir
def _onchange_project_id(record) do
  record |> :parent_id |> :project_id |> :id
end
```

