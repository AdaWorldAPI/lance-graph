# D-LANESOCK — DR × DT64 property-to-lane socket and joint version witness

**Status:** NORMATIVE SPEC (D-LANESOCK-0). Implementation D-LANESOCK-1 is a separate draft PR.
**Date:** 2026-11-06.
**Follows:** #1964 / #1965 (hybrid ordinal chain; batch selection over OGAR Ruler planes), AdaWorldAPI/OGAR#371.

## 0. Ownership (unchanged)

| Layer | Owns | Never owns |
|---|---|---|
| SoA64K | the canonical lanes and their values | a second lane universe |
| DR (`lance_graph_dt64::dr`) | positions, `RouteLog`, `FrameStore`/`Frame`, `MapTable`, `Guards` | values, method meaning |
| DT64 (`BindStore`) | `ClassSchema`/`SchemaRev`, method bindings, `MethodRef` admission, the route log it keeps | values, physical lane addresses |
| `ClassView` | the field basis (`FieldRef` order) and labels | a physical address |
| `WideFieldMask` | which properties are requested | a row-survivor bit |
| Quack → mask-RISC → ndarray | admitted execution over borrowed lanes | identity, routing, schema |
| Source owner (dir-sim, SAP adapter, ...) | which of its lanes holds which property, in whose row order | DT64 schema |

The socket adds no store, no clock, no index and no writer.

## 1. Gap this closes

1. Nothing maps `(classid, SchemaRev, field position)` to a source-owned lane and the coordinate space that lane is in.
2. `BindStore` owns a private `RouteLog` with its own clock; `FrameStore` has another. Nothing relates a class route to a source frame, and nothing may assume they are one clock.

## 2. The socket (implemented by the source owner)

```rust
/// Whose row order a lane is in.
pub struct CoordSpace { pub population: u64, pub coordinates: u64 }

/// A source-owned lane: address, row order, and the reader scope it serves.
/// No value.
pub struct LaneAddr { pub space: CoordSpace, pub lane: u16, pub scope: Scope }

pub trait LaneSource {
    type Lane<'a> where Self: 'a;            // the borrowed lane (e.g. mask-RISC LaneRef)
    fn lane_of(&self, class: u32, rev: SchemaRev, field: u8, basis: &FieldRef, frame: &Frame)
        -> Option<LaneAddr>;
    fn borrow(&self, addr: &LaneAddr) -> Option<Self::Lane<'_>>;
}
```

`basis` is the field DT64 resolved at `field` through the `ClassView` it validated (amended by D-LANESOCK-3: a source that read a ClassView of its own could disagree with DT64 on field order or identity at equal class, revision and count, and returned another lane). A source maps `basis`, never a position. `None` from `lane_of` is UNBOUND and is refused, never read as empty. `borrow` hands out the source's own lane; it never copies, reorders or allocates a lane.

## 3. The joint witness (DT64)

```rust
pub struct RouteAt { pub position: u16, pub version: RouteVersion }

BindStore::bind_lanes(view, frames, source, LaneRequest {
    route: RouteAt, rev: SchemaRev, fields: &WideFieldMask,
    frame: &Frame, scope: Scope, selection: CoordSpace,
}) -> Result<LaneBinding, LaneRefusal>

LaneBinding::check(&self, store, view, frames, source) -> Result<(), LaneRefusal>   // hot, no allocation
LaneBinding::lane(&self, store, view, frames, source, field) -> Result<S::Lane<'_>, LaneRefusal>
```

`LaneBinding` carries the class, `SchemaRev`, `RouteAt`, the sealed `Frame`, the reader `Scope`, the selection `CoordSpace` and one `LaneAddr` per requested field. It holds no value.

**Versions are a pair, never a merged clock.** The route is checked against `BindStore`'s own log (`routed_at(current_version, position)` must still be the bound class). The frame is checked against the `FrameStore` that issued it. Each refuses on its own. Sharing a `VersionClock` remains an owner's construction choice and is not required.

## 4. Refusals (exhaustive)

| Condition | Refusal |
|---|---|
| unknown class, unrouted position, future version, schema revision or field basis changed | `LaneRefusal::Bind(Refusal::…)` (existing) |
| class re-routed since the bind | `RouteChanged { position, bound, now }` |
| foreign, stale (source / map / coordinates) or re-sealed frame | `Frame(FrameRefusal::…)` (existing) |
| requested field outside the class basis | `FieldNotInBasis { class, field }` |
| source declares no lane for the field | `Unbound { class, field }` |
| lane row order ≠ selection row order | `CoordinateSpace { field, selection, lane }` |
| lane scope ≠ reader scope | `Scope { field, reader, lane }` |
| source's mapping changed since the bind | `MappingChanged { field }` |
| source no longer holds the lane | `Vanished { field }` |

`CoordinateSpace` is the guard that stops an AD-slot row mask reading an Entra-order lane. No mapping is admitted today to lift it; when one is, it is named here and owned by DR, not by the socket.

## 5. Consumers

- **PSCustomObject-shaped projection.** Row mask (Quack `Agg::Rows`) × `WideFieldMask` × `LaneBinding::lane` per field; labels from `ClassView`. No row object.
- **XVERWEIS.** A DT64 `MethodRef`, admitted by `BindStore::admit`; its `reads` must be inside the binding's fields. The method pointer is not the coordinate; the coordinate is not the value. Across coordinate spaces it refuses (`CoordinateSpace`) until a gather is admitted.
- **Quack masked selection and typed folds.** `QueryPlanes.lanes` filled from `LaneBinding::lane`; `Filter`/`Agg` as today.

## 6. Admissions unchanged

Gather ban, ADMISSION B, NotASide, OP-14, DRISA M1/M2, D4/S-INGRESS. The socket is read-side only.

## 7. Falsifiers (D-LANESOCK-1)

1. Two properties resolved through one `ClassView` basis; one `WideFieldMask` selects both.
2. DR routes the class under a witnessed frame.
3. A DT64-admitted method drives one Quack fold over a borrowed lane; the same binding drives a second operation on the other lane.
4. Independent scalar oracle parity; zero hot allocations.
5. Each refusal in §4 fires on a realistic input, and a disabled guard turns its test red.

## 8. Downstream consumer contract (handoff, not wiring)

The report-rs Cockpit (`/view/mobile`), Hub (`/view`) and Matrix (`/`) are three lenses over one scoped projection (report-rs #43). When they move onto source-backed lanes, they consume one answer through this chain and add nothing upstream:

```text
authoritative role/customer scope
  → admitted source population and frame          (source owner, DR FrameStore)
  → DT64 property binding                         (BindStore::bind_lanes, one field basis)
  → borrowed source-owned lanes                   (LaneSource::borrow, e.g. AbiLanes)
  → Quack                                          (admitted MethodRef, one evaluator)
  → authorized projection                          (field mask ∩ row scope)
  → Cockpit / Hub / Matrix                         (presentation only)
```

No lens gets its own lane binding, KPI evaluator, SoA or storage in lance-graph. A KPI is a Quack fold over the same binding; the three surfaces differ only in what they render.

### Reader-scope witness: inventory (2026-11-06)

`dr::Scope` is a bare `{ id: u64 }`; nothing issues it with authority, so no production `LaneSource` may be constructed yet (`AbiLanes` has only a `#[cfg(test)]` constructor). Existing authority pieces, read in code:

| Piece | Where | What it yields |
|---|---|---|
| `RbacAuthority::bind(&RbacPlug) -> RbacBinding` | `lance-graph-contract/src/rbac_plug.rs` | grants for exactly the plugged classids and roles; private fields, no `Default` (fails closed) |
| `authorize_scoped` → `ScopedDecision { decision, scope: Option<ScopeSpec>, field_mask }` | `lance-graph-rbac/src/authorize.rs` | stage-1 verdict, restrictive-AND row scope, union field mask |
| `authorize_memberships` → `MembershipDecision { decision, scope: Option<ScopeSet>, field_mask }` | `lance-graph-rbac/src/authorize.rs` | the same over per-membership scopes |
| `ogar_rbac::authorize_identity(&RbacBinding, &AuthenticatedUser, ..)` | OGAR `crates/ogar-rbac/src/lib.rs` | the membership decision for an `ogar-auth` user |
| `ActorContext { actor_id, tenant_id, roles }` | `lance-graph-contract/src/auth.rs` | JWT-derived actor; not a decision |

The missing seam is a witness that seals an authorized decision (decision + row scope + field mask, for one actor, one class, one frame) into what `LaneRequest.scope` / `LaneAddr.scope` compare. Rules it must keep:

- A numeric `Scope` alone authorizes nothing; equality of two ids is not a grant.
- `scope: None` ("unrestricted") from a decision is never widened automatically into customer-wide access; it must be an explicit grant.
- report-rs's ZITADEL login and its `RoleScopeAuthority` are a downstream integration point that would supply the actor to this chain, not a second upstream identity authority.
- Source status (available, stale, unknown) is reported only from observed evidence.

### Reader-scope seam: where it can live (2026-11-06, D-LANESOCK-4)

Read in code at lance-graph `fb27c46`, OGAR main and report-rs `83093de`. Nothing here is built.

**Authentication.** OGAR `ogar_auth::federation::VerifiedLogin` is the proof of a login: private fields, no public constructor, `From` or `Default`; `establish` is its only producer (`ogar-auth/src/federation.rs:199-213`). `AuthenticatedUser` is not a proof: its fields are public and its own docs call it "descriptive, not a proof" (`ogar-auth/src/user.rs:296-308`).

**Authorization.** `ogar_rbac::authorize_identity(&RbacBinding, &AuthenticatedUser, class, op) -> MembershipDecision` (`ogar-rbac/src/lib.rs:327`) and `OgarRbac::authorize_user -> ScopedDecision` (`:155`) run the `lance-graph-rbac` kernels. A decision carries allow/deny, a row scope and a field mask, but no actor, class, frame or issuer identity: it is a value, not a witness. `scope: None` means at least one granting membership is unscoped.

**DR.** `dr::Scope { pub id: u64 }` is compared by equality only (`dr.rs:503`, `lane.rs:209`). Every literal construction in lance-graph and OGAR is in a test, example or probe crate (about 50 sites). The sealing precedent is `FrameStore` / `Frame`: issuer identity, no public constructor, two stores never equal. `GuardWitness` is the other precedent and is value-derived (equal guards, equal witnesses), so it cannot carry authority.

**Dependency direction.** `lance-graph-dt64` and `lance-graph-rbac` each depend only on the contract; OGAR depends on lance-graph (`ogar-rbac` → `lance-graph-rbac`), never the reverse. An issuer that starts from `VerifiedLogin` therefore lives in OGAR or a downstream consumer, not in lance-graph.

**Shape the code admits.**

1. dt64 gains a sealed reader scope next to `Scope`, issued by its own store like `FrameStore` issues `Frame`. A new type, so the test literals of `Scope` stay valid; the lane socket would compare the sealed scope instead of the bare id.
2. The issuer holds that store, lives in OGAR or downstream, and issues only from a `VerifiedLogin` through `authorize_identity`, binding actor, class, row scope, field mask and policy version, checked against the frame it was issued for.
3. It keeps the rules above, plus: nothing keys on a hand-built `AuthenticatedUser`.

**Downstream observations (report-rs `83093de`, reported to its owner, not changed here).**

- `LanceGraphScope::decide_in` maps `scope: None` to all customers (`src/iam.rs:231`). Safe today only because report-rs's actor source emits `None` solely for an explicit all-customers grant.
- `RoleScopeAuthority::scope_for(&AuthenticatedUser)` (`src/flow_scratch.rs`, around 288-306) binds the user id from a `VerifiedLogin` at construction but accepts any `AuthenticatedUser` with that id. The trait is OGAR's (`ReportAuthority`, `ogar-flow-m365/src/host.rs:88`).

### Still closed

Value ingress and the durable workbook writer; D4/S-INGRESS and layout admission; text/codebook authority; exact decimal; production CRM_COMPANY XVERWEIS; cross-coordinate Gather, ADMISSION B, NotASide, OP-14; the `RegistryClassView` DOLCE `entity_type_id` defect (tracked separately).
