# D-METHOD-COORDINATE-1: XVERWEIS as a method pointer over routed coordinates

**Status:** PROPOSED, architecture/admission plan only. **Date:** 2026-11-05. **Owners:** DT64 binds the behavior; DR routes coordinates; Quack/relalg/mask-RISC execute admitted parts; ndarray provides positional mechanics; OGAR supplies source-qualified identity laws.

## 1. Name the three different things correctly

1. **Operation pointer:** a DT64 `MethodRef { class, method, sig_rev, guard }` and, where used, an `AlgebraRef { lens, word }`. It names an immutable, typed behavioral contract. **XVERWEIS is one possible presentation/name for a lookup operation; the name is not its identifier.**
2. **Execution result:** a particular method may produce a **witnessed coordinate reference**, selection/mask or scalar. The returned coordinate is not the operation pointer, not a copied value and not a permanent source identity.
3. **Presentation result:** a caller may dereference a coordinate and request a property, a PSCustomObject-like projected object, or an Excel-compatible XVERWEIS return value. **Excel's return-value, first/last and spill rules belong to that consumer, not the definition of the pointer.**

A method can resolve a cross-tile coordinate or perform a fold. **It is not a law that every method, or every XVERWEIS-shaped operation, is an ordered Bag fold.**

```text
a call site / function spelling / class behavior
    → DT64 MethodRef (what to do, signature/guard/basis)
    → DR route + scoped source-population binding (where)
    → admitted Quack / mask-RISC operation (how)
    → coordinate descriptor or terminal result (execution result)
    → ClassView field ordinal → lazy property read (if demanded)
    → external object / Excel cell / report (only if demanded)
```

This is the **DR teleportation** and **DT parity law** model. The behavior pointer is stable in meaning under class/schema/signature revision; its execution may require new coordinate binding when data, scope or frame changes.

## 2. Canonical existing owners

- [DT64 `MethodId`, `MethodRef`, `SigRev`, `AlgebraRef`, `WideFieldMask`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dt64/src/lib.rs): method names are non-identifying labels; class behavior is immutable once registered, overrides require derived class and reroute. F### behavior **lens** convention exists; do not mint a duplicate GridCraft F-number or pretend the implementation registry has every Excel function.
- [DR route and frame](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dt64/src/dr.rs): source position + classid route, Identity × Property × Predicate address space, `RouteVersion`, mapping/layout witnesses. DR stores neither business values nor method semantics.
- [Quack](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-quack/src/lib.rs): lowers lawful predicates/folds to one mask-RISC program; no second evaluator.
- [OGAR `BACKSYNC_TO_ENTRA_ID`](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-dir-core/src/identity.rs) plus [hybrid correspondence](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-dir-core/src/correspond.rs): the AD `msDS-ExternalDirectoryObjectId = User_{cloudGUID}` backsync, Entra `id = cloudGUID`, and EXO `ExternalDirectoryObjectId = cloudGUID` are a practical **one-key coordinate chain** after decoding at ingress. AD's `objectGUID` and `sourceAnchor` are distinct identity/evidence fields, not automatically equal to Entra `id`.

## 3. Abstract method contract for coordinate lookup

```text
Input:
  MethodRef, source population + scope + snapshot, query key,
  field basis and typed equality, match/multiplicity policy,
  target population and layout/version witnesses

Bound:
  normalized key/reference → source row ordinal → target row ordinal
  (where a lawful, source-qualified correspondence exists)

Output:
  CoordinateRef { population, row_ordinal, scope, data_version,
                  frame/layout witness, class/basis binding }
  OR typed not-found / ambiguous / stale / unauthorized / unsupported refusal
```

This is **conceptual**, not a new deployed Rust ABI. A method that chooses a single coordinate must declare duplicate-match policy; do not quietly take the first. A method that counts occurrences preserves Bag multiplicity. An Excel XVERWEIS adapter may separately specify first/last match, lookup array order, blank/error propagation, approximate/wildcard and result shape. None of those Excel rules should be imposed on unrelated SAP/PowerShell methods.

**No cross-source materialization:** once correspondence is verified, a cached coordinate descriptor can serve multiple fields using ClassView field ordinals and borrowed SoA lanes. Its lifetime is bounded by scope, data and layout revisions, not by a naked `u16`.

## 4. Admission boundary

The existing hybrid identity fold prepares sorted GUID indexes and resolves correspondence by scope. `ndarray/src/simd_stride.rs` is a **positional access pattern**, not a shipped full GUID join. Do not conflate "chained strides" with an unrestricted value-derived Gather: a foreign-lane address is still subject to ADMISSION B and the Gather ban. XLOG NotASide, computed-mask reuse and OP-14 composition remain blocked where applicable.

A **prebound co-addressed view** can be used where already lawful; otherwise the method refuses pending owner admission. DR ISA/layout decisions M1/M2 and D4/S-INGRESS remain open. No invented `HashMap`, local F### opcode map or sheet-row cache may become an alternate semantic authority.

## 5. Minimal independent proof

- Synthetic AD / Entra / EXO `Guid128` correspondence: one unique backsync, one missing, two duplicate targets, one mismatched source-anchor witness, and cross-tenant ID collision. Decode `User_{GUID}` at boundary, not per lookup.
- One DT64-bound method pointer, method/guard/signature/basis witness, one coordinate lookup, then **two distinct property dereferences through the same coordinate** without reconstructing joined rows.
- One SAP-like numeric fold on the selected, lawful population; compare with an independently authored scalar oracle.
- Change the input row permutation and route version. The stale descriptor refuses or rebinds. A changed `MethodBinding` under the same classid is refused; derived class + reroute works as allowed.
- Negative controls: dropped scope, order/multiplicity policy, layout witness, value revision, guard and source family; count cold lookup, warm reuse, per-row objects and terminal egress separately.

**STOP** if a test relies on an unadmitted cross-table hop; show explicit `NOT EXPRESSIBLE` rather than secretly implementing another kernel.

## 6. Program role

This PR is an **optional operation-pointer/BAGOP probe** for the SAP/PSCustomObject tile POC. It does **not** require GridCraft. P3/P5/P6/P7 cover optional consumers, and [lance-graph-java P4](https://github.com/AdaWorldAPI/lance-graph-java/pull/93) is non-blocking evidence of no-materialization only. No runtime changes, merge or method admission are authorized here.
