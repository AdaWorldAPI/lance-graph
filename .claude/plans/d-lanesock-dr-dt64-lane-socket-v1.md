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
    fn lane_of(&self, class: u32, rev: SchemaRev, field: u8, frame: &Frame) -> Option<LaneAddr>;
    fn borrow(&self, addr: &LaneAddr) -> Option<Self::Lane<'_>>;
}
```

`None` from `lane_of` is UNBOUND and is refused, never read as empty. `borrow` hands out the source's own lane; it never copies, reorders or allocates a lane.

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
