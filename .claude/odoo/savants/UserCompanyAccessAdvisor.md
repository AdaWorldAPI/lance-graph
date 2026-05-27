# Savant: UserCompanyAccessAdvisor  (id 10 · family 0x80 · lane L12)

**Tuple:** kind=CustomerCategory · inference=Induction · semiring=NarsTruth · style=Empathic
**Feeds Reasoner impl:** `CustomerCategoryReasoner`   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::CustomerCategory` -> "classify against the family codebook (deductive
> lookup)" (`examples/savant_dispatch.rs:29`); the *inductive* facet ("users-like-this access this
> subset") rides the same impl, selecting `QueryStrategy::CamWide` via
> `InferenceType::Induction::default_strategy()`. Style Empathic inherited from 0x80
> SmbFoundryCustomer (relationship/role-shaped access).

## What it decides (AXIS-B core)
In a multi-branch company tree, decide **which subset of branches a user should have active** in their
allowed-companies context -- `_accessible_branches` returns the branches accessible to the current
user (L12 R10, `res_company.py:L429-450`). The hard constraint (a user may only access companies
within their granted `company_ids`, scoped over the tree's `parent_ids`/`root_id`) is deterministic;
the AXIS-B core is the *recommendation* residual: given the user's role, recent activity, and the
branch topology, which subset of the granted branches should be **active by default** (the working
set) -- induce from users-like-this-role. Output is a suggested active-branch subset with NARS
`(frequency, confidence)`; woa-rs sets the default allowed-companies selection behind its guard, never
granting access the security model has not already permitted.

## Deterministic guard (AXIS-A -- stays in woa-rs)
`_accessible_branches` (L12 R10, `res_company.py:L429-450`) computes the accessible subset; the
company-tree mechanics are deterministic: `_parent_store` / `root_id` / `parent_id` immutable after
create, and `_get_company_root_delegated_field_names() = ['currency_id']` so branches inherit root
currency (L12 R9, `res_company.py:L96-104, L341-418`). The `check_company` /
`check_company_domain_parent_of` rule -- a related record's company must be in
`company_id.parent_ids` (ancestor-or-equal, ltree `parent_path` subtree) -- is a deterministic
security guard (L12 R11, `account_move.py:L78, L877-881`). The set of branches a user is *permitted*
to access (their granted `company_ids`) is fixed by the security model in woa-rs; the savant only
ranks/selects the **default active subset** within that permitted set.

## Slot 1 -- Evidence (Arrow EvidenceRef)
Two correlated tables. The *user access context*
`EvidenceRef { table: "res_users.company_access_context", schema_fingerprint, rows }`
(one row = the user whose active-branch subset is being recommended):

| column | dtype | signal |
|---|---|---|
| `user_id` | `Int64` | the user the recommendation is scoped to |
| `company_id` | `Int64` | the user's current/default company (the working root) |
| `allowed_company_ids` | `List<Int64>` | the **permitted** set (security-granted; the savant selects a subset of this, never beyond it) |
| `root_id` | `Int64` | the tree root the user's companies hang under (L12 R9) |
| `role_group_ids` | `List<Int64>` | the user's security groups / role (induction axis: users-like-this-role) |
| `recent_company_ids` | `List<Int64>` | branches the user recently transacted in (activity-weighted working set) |

The *branch topology corpus* `EvidenceRef { table: "res_company.branch_tree", ... }`
(`res.company` -> `fibo:LegalEntity`, family 0x80, `odoo_alignment.rs:214-219`):

| column | dtype | signal |
|---|---|---|
| `branch_id` | `Int64` | candidate branch identity |
| `parent_id` | `Int64`/nullable | tree parent (NULL = root); immutable after create (R9) |
| `parent_path` | `Utf8` | ltree materialised path -- `check_company_domain_parent_of` subtree test (R11) |
| `root_id` | `Int64` | the branch's root (branches share root currency/rates, R9/R3) |
| `currency_id` | `Int64` | inherited from root (delegated field, R9) -- context for cross-branch work |

## Slot 2 -- Odoo field -> signal map                 (cite L-doc file:lines)
- `_accessible_branches` (subset of branches accessible to the current user in multi-branch context) + delegation tuple <- `L12-MULTICOMPANY-CURRENCY.md:54-56` (R10; `res_company.py:L429-450`).
- company tree + root delegation (`_parent_store`, `root_id`, `parent_id` immutable, `currency_id` delegated to root, branch currency must equal parent) <- `L12-MULTICOMPANY-CURRENCY.md:51-52` (R9; `res_company.py:L96-104, L341-418`).
- `check_company` / `check_company_domain_parent_of` (related record company must be in `company_id.parent_ids`, ltree subtree) <- `L12-MULTICOMPANY-CURRENCY.md:58-59` (R11; `account_move.py:L78, L877-881`).
- rate lookup uses `company.root_id` (branches share root rates) -- context for cross-branch currency <- `L12-MULTICOMPANY-CURRENCY.md:32` (R3; `res_currency.py:L120-139`).
- delegation tuple `(CustomerCategory, Induction, NarsTruth, Empathic)` <- `L12-MULTICOMPANY-CURRENCY.md:56` (R10 savant seed).
- NEEDS-INPUT: `role_group_ids` / `recent_company_ids`. L12 establishes the tree + accessible-subset mechanics (R9/R10/R11) but does not enumerate the user/role fields (`res.users` security groups) or an activity log; the role evidence lives in the RBAC/tenancy surface, not L12. Source candidates: a `res.users` / `res.groups` lane and a transaction-activity aggregate -- confirm the exact role/activity fields with the RBAC worker (cf. `super-domain-rbac-tenancy-v1` referenced in `odoo_alignment.rs:9`) before the impl binds these columns.

## Slot 3 -- Property-level alignment
N/A -- class-level pivots only; no `owl:equivalentProperty` defined. (Confirmed: `odoo_alignment.rs`
holds only class-level `owl:equivalentClass` rows; zero property IRIs in the repo.) `res.company` ->
`fibo:LegalEntity` (class-level, `odoo_alignment.rs:214-219`; same row as `res.partner`'s company
facet) is the only pivot touched; `res.users` is unmapped and used only as an identity key. The
parent/child branch relation is read as the ltree `parent_path` scalar (R11), not as a traversed OWL
property. No FIBO/SKR/ZUGFeRD seam is crossed at decision time. **N/A -- stays within 0x80 reading the
company tree's own `parent_path`/`root_id` scalars.**

## Slot 4 -- AXIS-B decision in evidence terms
Let E = the user access context (slot 1) + the branch topology corpus, with `allowed_company_ids` as
the hard permitted ceiling (the savant selects a subset of it, never beyond).

-> Conclusion C = `RecommendActiveBranches(user_id, subset-of allowed_company_ids)` emitted with NARS
`(frequency, confidence)` where:
- **frequency** of including a branch in the default active subset rises with: recent activity in that
  branch (`recent_company_ids`), the branch being on the user's working `root_id` subtree (close in
  `parent_path` to `company_id`), and the user's role matching roles that typically work that branch
  (the inductive "users-like-this-role" signal).
- **frequency** falls for branches the user is permitted but never touches, or branches on a distant
  root the user's role does not typically span.
- **confidence** is the NARS evidence weight from how many comparable role/activity observations
  support the subset; a user with little activity history yields a low-confidence recommendation
  (default to the full permitted set is the safe fallback). Capped by phi-1.

Discriminating features (ranked): `recent_company_ids` activity (NEEDS-INPUT) >> role-group match
(NEEDS-INPUT) > `parent_path` proximity to the working company > `root_id` alignment. Induction is
"users with this role and activity pattern work in this subset of branches" -- a working-set
recommendation, strictly inside the security-granted ceiling.

## Parity / GoBD notes
This is a UX/working-set convenience over the multi-company security model, **not** a permission grant:
the savant can only ever *narrow* (or re-order) the already-permitted `allowed_company_ids` -- it must
never propose a branch outside the security-granted set, because `check_company` /
`check_company_domain_parent_of` (R11) is the real, deterministic enforcement and stays authoritative.
Suggestion-only per Iron Rule 7: woa-rs may pre-select the default active-companies subset, but the
user can switch to any branch they are permitted, and no posting is gated by this advisory. No GoBD
ledger interaction (this is access-context selection, not a journal entry). Branch currency is
root-delegated (R9), so a recommended cross-branch working set stays currency-consistent within a root
by construction.
