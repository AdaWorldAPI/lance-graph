# Gate D: `.mermaid/03`, `/05` and `/06` checked against source

Source: lance-graph `317c4ffc0`. The OGAR citations in `06` were read in an OGAR checkout at `8fab880`.
A worker read each diagram in full and each cited source range, and wrote the lists below. The
orchestrator re-read four of its items at the stated lines (PR #1996 merged; `on` read only by
`methods_on`, `footprint`, the digest and the registration check; `NativeReceipt` has no `method`;
`MapTable::map` checks only the frame's issuer). The other items are the worker's reading and are
not re-checked. The diagrams are references; nothing here is a machine fact. No cargo was run for
this check.

---------------------------------------------------------------------
## FILE 03-dr-dt64-binding.md
---------------------------------------------------------------------

### 1. CONFIRMED

C1. "`Width` is DR64 (`W64`) or DR256 (`W256`). A DR64 tile has 64 positions, a DR256 tile has 256." (l.89) -> dr.rs:32-44.
C2. Class "reserves an `Extent`: one `Axis` each for identity, property and predicate (methods)" (l.90) -> dr.rs:96-102 (`predicate` doc: "its methods").
C3. "A reservation costs nothing. Only identity tiles a route touches count as claimed." (l.90) -> dr.rs:91-95, 140-148, 319-335.
C4. "`RouteLog` is append-only. Only routing is versioned" (l.91) -> dr.rs:206-208, 285-292 (`push` only appends); class diagram methods route/check/retract/routed_at/claimed at dr.rs:255, 268, 281, 296, 319.
C5. `RouteRefusal` has OutsideReservation and FutureVersion (l.54-58) -> dr.rs:191-204.
C6. `VersionClock` current/tick/fork (l.43-46) -> dr.rs:175, 180, 185.
C7. "`MapTable` changes only by appending. An old frame replays the layout in force when it was sealed." (l.92) -> dr.rs:703-713, 859-869, 1053-1056.
C8. MapTable members advance/advance_at/seal/map(frame,tile)/diagnose_at (l.59-65) -> dr.rs:970, 997, 1045, 1057, 1075.
C9. "`diagnose_at` takes a bare version number and returns a read-only diagnostic, not an executable map. Execution goes through `MapTable::map` with a frame." (l.93) -> dr.rs:825-831, 1062-1088.
C10. `LocalMap` prepare/local/global and `MapRefusal` TooMany/Duplicate/Unmapped/OutOfTile (l.66-77) -> dr.rs:555-566, 586, 622, 631.
C11. "A frame is a seal over three revisions. Only a `FrameStore` can issue one. A frame compares by the identity of its issue, never by its numbers." (l.109) -> dr.rs:356-361, 375-377, 482-491, 505-507.
C12. `FrameRefusal` five arms (l.130-137) -> dr.rs:364-373.
C13. "A frame from another store is `ForeignFrame` even at equal numbers. A stale frame names the first revision that moved, or `FrameResealed` if none did." (l.169) -> dr.rs:459-479.
C14. "`Focus` is motion, not reference"; "`Scope` here is an opaque `u64` id" (l.170-171) -> dr.rs:510-522. `Guards::check(store, now)` checks frame, then focus, then scope -> dr.rs:543-551.
C15. `ClassView` has fields / fields_at / lens_fields (l.188-192) -> class_view.rs:1061, 1105, 1084.
C16. `ClassSchema` = schema_rev, field_count, extent (l.194-198) -> lib.rs:253-258.
C17. `MethodBinding` has id, sig_rev, on, when, reads, effects (l.199-206); "`on` is the field mask that triggers it, `when` is an optional guard, `effects` is the ordered DO list" (l.258) -> lib.rs:214-231.
C18. `BindStore` register / route / resolve / method_ref / admit / bind_lanes (l.207-214) -> lib.rs:814, 1018, 1073, 1136, 1180; lane.rs:171.
C19. "`Resolved` is an answer, not a witness"; accessors field_view / field_meaning / mask_where / action_ordinals (l.260) -> lib.rs:638-645, 683, 700, 719, 760.
C20. "A field with no spec is never selected by `mask_where`." (l.262) -> lib.rs:715-717, 724.
C21. `PropertySpec.marking` is one of Public, Internal, Pii, Financial, Restricted (l.261) -> property.rs:41-58, 769-783.
C22. "`BindStore::method_ref` issues a `MethodRef` stamped with the store identity. `BindStore::admit` ... refuses with `SignatureRevisionMismatch`, `GuardDropped`, `GuardAdded`, `GuardMismatch`, `BindingMismatch` or `ForeignStore`." (l.259) -> lib.rs:1136-1157, 1192-1210 (each of those six is returned there; the list is not complete, see S2).
C23. `LaneSource` lane_of / borrow / frames (l.221-226) -> lane.rs:57, 66, 79. `LaneAddr` space/lane/scope -> lane.rs:36-40. `LaneRequest` six fields -> lane.rs:91-100.
C24. "`LaneBinding` holds addresses only. It never holds a value." (l.263) -> lane.rs:150-165.
C25. 3.4 sequence order of checks in `bind_lanes`: source_owns, frame check, routed_at, current route, resolve, per-field FieldNotInBasis / Unbound / CoordinateSpace / Scope (l.292-328) -> lane.rs:178-215, 220-263.
C26. `LaneRefusal` arms named in the 3.4 alt branches (Frame, Bind, RouteChanged, FieldNotInBasis, Unbound, CoordinateSpace, Scope, StoreBasisMismatch, MappingChanged, Vanished, NotBound) -> lane.rs:104-141.
C27. "`borrow_all` runs the one check first, so N lanes cost N source lookups and not N squared." (l.349) -> lane.rs:365-391.
C28. "`LaneBinding::lane(field)` re-runs the whole check on each call and returns `NotBound` for a field outside the binding." (l.350) -> lane.rs:352-363.
C29. "`check` does not hash the basis again. It compares a `u128` witness and the revision and field count. A same-count reorder ... is caught only if the source maps lanes by predicate." (l.351) -> lane.rs:296-326 (esp. 303-306, 317-326).
C30. "`Vanished` is yielded per lane and the iterator carries on." (l.352) -> lane.rs:374-375, 404-413.
C31. "The returned lanes are as current as the check. No lock is held on the source." (l.353) -> lane.rs:377-378.
C32. 3.5: no business value, no evaluator, no persistence path (l.368-372) -> lib.rs:24-30.
C33. Basis check in `bind_lanes` goes through `resolve` (witnessed) (l.309-311) -> lane.rs:187; lib.rs:1073-1091, 1220-1246 (revision, count, then predicate-witness).

### 2. STALE or WRONG

S1. "Both the route log and the map table tick one shared version clock." (l.18) -> only when built from clones of one `VersionClock`. `RouteLog::with_clock` (dr.rs:235) and `MapTable::with_clock` (dr.rs:885) share; `MapTable::new` (dr.rs:879) and `RouteLog::default` make their own clocks; `BindStore` holds `routes: RouteLog` built by `Default` (lib.rs:777). Doc: dr.rs:158-165. The statement is true as a capability, not as a property of every store.
S2. admit refusal list (l.259) is incomplete: `admit` also returns `UnknownClass` (lib.rs:1184) and `UnknownMethod` (lib.rs:1190).
S3. 3.4 "class unknown or revision or basis differs: Bind(UnknownClass, SchemaRevisionMismatch, PredicateBasisChanged)" (l.311) -> `check_basis` can also return `FieldBasisChanged` (lib.rs:1235, 1291-1303) and `basis_at` can return `UnknownLensBasis` (lib.rs:1277-1279).
S4. Line drift in the Sources lists (symbol is present, range moved): `ClassSchema` cited lib.rs:253-262, actual 253-258; `MethodRef` cited 352-366, actual 352-359 (`Stamp` 370-374); `Resolved` cited 640-770, actual 638-763; `BindStore` cited 772-795, actual 771-791; `admit` cited 1180-1215, actual 1180-1212; `resolve` cited 1073-1095, actual 1073-1091; `LaneBinding` methods cited lane.rs:264-396, actual 265-392; dr.rs ranges are within about five lines (e.g. `LocalMap` cited 568-660, actual 572-655; `MapTable` cited 860-1100, actual 859-1134).

### 3. MISSING RELATIONSHIPS (each checked)

M1. `binding_digest` does NOT cover the guard. Doc: "neither is the guard: `BindStore::admit` compares the guard witness structurally" (lib.rs:407-410); body lib.rs:411-468 hashes class, basis digest, id, sig_rev, on, reads, effects (target, lens, word, lens rev, arity, args), population, truth, demand; no `when`, no `name`. Diagram 3.3 shows `MethodBinding` but not what the stamp digest covers.
M2. `admit` does NOT check the field basis. It has no `ClassView` parameter (lib.rs:1180). The basis enters only as `BasisDigest` stored in `ClassEntry.basis` and folded into `entry.bindings[i]` at registration (lib.rs:277-284, 879, 893-896); `admit` compares the stamp digest to that stored one (lib.rs:1207). The basis against the view is checked in `resolve`/`check_basis` (lib.rs:1073-1091) and `bind_lanes`, not in `admit`. Diagram edge "BindStore ..> ClassView : reads basis through" is true of resolve/bind_lanes only.
M3. Registration `validate` does NOT check the guard against the basis. Position check covers `on`, `reads`, effect targets and `Arg::Field` args only (lib.rs:1353-1371); `m.when` is not visited. Other registration checks omitted from the diagram: BasisTooWide, ReservationTooWide, FieldsExceedExtent, MethodsExceedExtent, DuplicateMethodId, UndeclaredRead (read-before-write) (lib.rs:1310-1394); BasisConflict (lib.rs:881-892); UnknownAlgebra/ArityMismatch (lib.rs:852-870).
M4. The polyfill lens is absent from 3.3: `AlgebraRef{lens, word}` (lib.rs:128-132), `declare_algebra` (lib.rs:917-939), `arity`, `resolve_algebra` (lib.rs:954-973), `words_used`, `users_of` (lib.rs:977-1004). `resolve_algebra` calls `check_basis(..., None)`: revision and field count only, NO predicate witness (lib.rs:960-966, with the in-source comment "a reorder of two words with the same arity is NOT detected here"). `AlgebraLens` stores `schema` and `arity` only, no `BasisDigest` (lib.rs:136-140). `Resolved` doc states the difference (lib.rs:641-645).
M5. `MethodBinding` fields omitted in 3.3: `name`, `population`, `truth`, `demand` (lib.rs:220, 228-230). The last three are inside the binding digest (lib.rs:455-467).
M6. `BindStore` also has `register_derived`, `resolve_at`, `check_route`, `check_issuer`, `claimed`, `retract`, `declare_algebra` (lib.rs:826, 1119, 1034, 1161, 1052, 1062, 917). `ClassView::lens_fields_at` (class_view.rs:1115) is not in the interface box. `MapTable::claim` and `locate`, `MapRead`, `TileMap`, `MapDiagnostic` (dr.rs:985, 1113, 689, 763, 831) are not drawn.
M7. `MapTable::map` checks only the frame's issuer (`frames.issued`, dr.rs:1057-1060), never currency: "A frame of this table is never stale" (dr.rs:690-693, 1053-1056). 3.2 says `FrameStore::check` accepts only the current seal; both statements are right, but the diagram does not say `map` is the history path that does not use `check`.

---------------------------------------------------------------------
## FILE 05-methods.md
---------------------------------------------------------------------

### 1. CONFIRMED

C1. MethodBinding "id, sig_rev, ON mask, WHEN Guard, reads, DO effects" (l.22) -> lib.rs:214-231.
C2. "`register` is the only way a binding enters a store." (l.56) -> `register`/`register_derived` both call `insert` (lib.rs:814-822, 826-841, 843-912); `classes.insert` only at lib.rs:908. Node text "checks basis and unique MethodIds; a bound class never changes" (l.23) -> lib.rs:871-880, 1345-1351, 904-910 (BehaviourImmutable).
C3. "`method_ref` is the only way a `MethodRef` is issued. Its `Stamp` has private fields" (l.57-59) -> lib.rs:1136-1157, 370-374 (compile_fail doctest at 367-369).
C4. "`admit` returns the stored binding only after all checks pass. The order is fixed: identity, signature revision, guard witness, binding digest, issuer." (l.60-61) -> lib.rs:1180-1211; order doc lib.rs:1174-1179. Exact order in code: class lookup (1182), method lookup (1186-1190), sig_rev (1192), guard (1200-1206), binding digest (1207), issuer (1210).
C5. `Refusal` arms at "554, 555, 562, 564, 566, 571, 574" (l.84) -> exactly UnknownMethod 554, SignatureRevisionMismatch 555, GuardDropped 562, GuardAdded 564, GuardMismatch 566, BindingMismatch 571, ForeignStore 574 (lib.rs).
C6. `check_issuer` lib.rs:1161-1167 -> confirmed.
C7. `Guard` / `witness` (l.81) -> lib.rs:169-195.
C8. `MethodRef` + `Stamp` private fields (l.83) -> lib.rs:352-374.
C9. "DT64 keeps the binding and executes nothing." (l.5) -> lib.rs:1-27.
C10. "`run_with_program` ... `symbolic` answer returns no program" (l.89) -> lower.rs:70-83 (symbolic returns `(run, None)`).
C11. `Measure`, `Query` (l.90) -> ir.rs:98-119 (Measure at ir.rs:98-110, Query to 119).
C12. `to_set` normalize.rs:82 (l.91) -> `pub(crate) fn to_set` at normalize.rs:82.
C13. method.rs line cites for the lab are exact: `can_cook_guard` 47-61, `can_cook_method` 77-94 (with `when: Some(Guard::new(...))` at 87 and `reads` at 88), `MethodRefusal` 105-135 incl. `Folds` 118-120, `prepare` 211-292 with admit :217, borrow_all :229, rename :251, scope :252-257, run_with_program :264-272, one-fold check :276-280, `run` 295-320 with execute_into :302-309.
C14. Lab sequence (l.103-128): admit -> bind_lanes(reads + identity lane) + borrow_all -> rename -> optional conjoin "id lane != denied store" -> `run_with_program(Query{rel_from_set(guard), Keep})` -> refuse unless exactly one program -> per question `execute_into(Out::Mask)` -> read identity lane at set bits -> method.rs:211-320.
C15. "It refuses with `MethodRefusal::Folds` if the guard lowered into more than one fold" (l.137-138) -> method.rs:274-280; doc method.rs:114-118.
C16. "The reader scope in the lab is a stand-in written into the program. It is not the RBAC decision." (l.139-140) -> method.rs:14-16, 252-257; bind.rs:21-23 (READER "NOT an authorization").
C17. `execute_into` at mask-risc exec.rs:1343 (l.154) -> confirmed (`pub fn execute_into(` at exec.rs:1343).
C18. "Arm C ... `CostRefusal::NotBound` ... merged (#1995)" (l.142-144) -> arm_c.rs:66 (enum), :82 (NotBound), :182 (returned); commit 15d4b4cb4 "(#1995)" is in the checkout history.
C19. r2il module doc (binding.rs:1-73) says prepare/run, what is checked when -> confirmed; `prepare` order: guards (329), route (330), resolve (331-333), admit (334), WHEN refused (335-337), three-valued refused (338-340), lens check (341-343), per effect opcode (344-355).
C20. "r2il refuses a binding that has a WHEN guard" (l.72) and node NOWHEN "GuardNotExecutable" (l.43) -> binding.rs:251-252, 335-337.
C21. "`run` ... check the frame, focus, scope, route and an explicit `ExecPermit`" (l.70-71), sequence 3 steps "check_issuer; check; check_route; permit must list this exact MethodRef, else NotPermitted" -> binding.rs:386-395; same order in `run_native` binding_native.rs:361-370.
C22. "`ExecPermit` is documented as a lab permission. Who issues it is not decided." (l.75-76) -> binding.rs:124-130.
C23. `Receipt` fields plan_id, method, frame, steps_executed, programs_run, output; "bounded by the number of steps ... not a row trace" (l.187, 197-198) -> binding.rs:219-231 (for `run`).
C24. `Prepared` carries plan_id (l.174) -> binding.rs:164-175; plan_id hashes the stamp's binding digest -> binding.rs:522.
C25. `prepare_native` keeps compiled programs, refuses fold results that feed folds (l.179) -> binding_native.rs:203-340 (243-245, 259-262); `run_native` refuses `PlanesChanged` (l.199-200) -> binding_native.rs:372-377.
C26. "per effect: lens, label alias, EXECUTABLE opcode, arity" (l.173) -> binding.rs:541-629 (`opcode`, `frozen_opcode`).
C27. xlog `bag_count_admitted` needs `Population::Bag` and steps GroupCount, GroupCount, MulI64 (l.45, 201-203) -> xlog lib.rs:384-403; `BoundRefusal` lib.rs:363-376.
C28. `run` "never reads the `ClassView` or the vocabulary again" (l.192) -> binding.rs:13-16 and the body of `run` (377-400) calls none.
C29. r2il `prepare` also consumes `resolve`: "P->>S: check_route, resolve(view, class, schema_rev)" (l.170) -> binding.rs:330-333.
C30. Plan wording: "natural candidate is the `Operation::Act` decision in `lance-graph-rbac` ... candidate only" (l.224-226) -> .claude/plans/foundry-parity-v1.md:295 ("the natural issuer is an `lance-graph-rbac` `Operation::Act` decision").

### 2. STALE or WRONG

S1. PR #1996 described as open, lab code described as not on main. Statements: legend "Blue = open PR" (l.10); nodes `RFS` and `LAB` classed `open` (l.31, 35); "It lives on the open PR #1996 branch, not on main. `rel_from_set` is also new in that PR." (l.64-67); sequence note "G6 lab, crates/foundry-parity-lab (PR #1996, open)" (l.111); "(PR #1996)" tags on method.rs sources (l.93, 148-151); normalize.rs:124-130 "(`rel_from_set`, PR #1996 branch only)" (l.92); "The lab (`PreparedMethod`, `rel_from_set`) is on PR #1996 and is not merged." (l.228). Source: crates/foundry-parity-lab/src/method.rs exists on this checkout and contains `PreparedMethod` (method.rs:200-321); crates/lance-graph-relalg/src/normalize.rs:124-130 holds `pub fn rel_from_set`; `git merge-base --is-ancestor fcee68c90 HEAD` succeeded (fcee68c90 is the method.rs commit); the history contains "83d532759 Merge pull request #1996". So RFS and LAB are in code on main; after the correction no node is blue.
S2. "The intake re-checks its `MapTable` frame on every query." (l.204) -> `Intake::bag_count` calls `self.map(table)` (xlog lib.rs:345-347) which is `table.map(&self.frame, self.tile)` (xlog lib.rs:334-336). `MapTable::map` checks only that the frame was issued by this table and replays the layout at the frame's version; "A frame of THIS table is never stale" (dr.rs:690-693, 1053-1060). So a stale-but-same-table frame is not refused. xlog's own doc (lib.rs:378-381, "a stale frame, and nothing runs") overstates the same way. Only a foreign frame or missing tile refuses.
S3. "`Receipt` plan_id, method, frame, output" attached to the node `RUN` "run_native, hot" (l.41-42) and sequence 3 line 187 -> `run_native` returns `NativeReceipt` with plan_id, frame, programs_run, len, values and NO `method` (binding_native.rs:177-187, 410-416). Only `binding::run` returns `Receipt` with `method` (binding.rs:448-455).
S4. Line drift: `ExecPermit` cited binding.rs:125-131, actual 124-130; `Prepared` 165-176, actual 164-175; `ExecRefusal` 236-317, actual 235-312; `prepare` 321-355, actual 321-365; `run` 377-404, actual 377-400; `GuardNotExecutable` ":336-338", actual 335-337; `Guard` lib.rs:170-196, actual 169-195; `run_with_program` lower.rs:70-100, actual 70-92 (`run_as_written` starts at 96); xlog 363-405, actual 363-403.
S5. Node `REFUSE` (l.26) lists UnknownMethod but not UnknownClass, which `admit` also returns first (lib.rs:1182-1184).

### 3. MISSING RELATIONSHIPS (each checked)

M1. Guard vs digest: the guard is not in the binding digest (lib.rs:407-410, 411-468). Guard identity is carried separately as `MethodRef.guard: Option<GuardWitness>` (lib.rs:357, 1151) and compared in `admit` by recomputing `g.witness()` (normal form under the guard's logic; lib.rs:185-194, 1200-1206). Node `ADM` says "guard witness, binding digest" as two checks; it does not say they cover different things. Also `name` is in neither (lib.rs:407-408).
M2. `admit` does not check the field basis and takes no `ClassView` (lib.rs:1180-1212). Order differs by caller: r2il `prepare` runs `resolve` (basis witness) then `admit` (binding.rs:331-334); the lab `prepare` runs `admit` first (method.rs:217) and the basis check happens afterwards inside `bind_lanes` -> `resolve` (lane.rs:187). The diagram's "ADM" box suggests admit is the gate for the whole binding.
M3. Registration does not check WHEN guard positions against the basis (lib.rs:1353-1371 visits `on`, `reads`, effect targets, `Arg::Field`; never `m.when`). Nothing requires guard fields to be inside `reads`. In the lab, `reads` is derived from the guard (method.rs:76-81), and a guard field outside the bound reads is refused only at prepare as `MethodRefusal::NotBound` (method.rs:164, 169, 178).
M4. ON is never evaluated: in dt64, r2il, xlog and the lab, `on` is read only by `methods_on` (lib.rs:677-678), `footprint` (lib.rs:244-245), the binding digest (lib.rs:424), and registration position check (lib.rs:1355). No code in `prepare`, `run`, `run_native` or `PreparedMethod` consults it. The title "ON x WHEN x DO" in the diagram is the data shape; no trigger path exists in the code read.
M5. The polyfill lens check in r2il: `opcode()` calls `store.resolve_algebra(view, lens, rev)` (binding.rs:555-557), which checks revision and field count only, no predicate witness (lib.rs:954-966). Under `POLYFILL_LENS` the position binds and the view's label at that position must be an alias (binding.rs:567-568, 606-629; polyfill.rs:7-12, 116-170); under the R2IL fold reading the label binds and must name exactly one opcode (binding.rs:570-581). The binding digest includes each effect's lens `schema_rev` and word arity (lib.rs:436-443) but not the lens basis. binding.rs:35-36 says "its basis matches its `ClassView` at the stated schema revision (`resolve_algebra`)"; the call checks revision and count only.
M6. Registration refusals for effects not drawn: UnknownAlgebra, ArityMismatch (lib.rs:852-870), UndeclaredRead (lib.rs:1376-1391), AlgebraImmutable (lib.rs:931-936). `MethodBinding.truth`, `population`, `demand` are in the digest (lib.rs:455-467); r2il refuses `Truth::ThreeValued` (binding.rs:338-340); xlog requires `Population::Bag` (xlog lib.rs:391).
M7. Authorization relative to admission: `admit` has no actor, operation or decision parameter (lib.rs:1180); neither `prepare` (binding.rs:321-328), `run` (binding.rs:377-385), `run_native` (binding_native.rs:351-359) nor the lab `PreparedMethod::prepare` (method.rs:211-216) takes a `lance-graph-rbac` type. The only gate in the hot path is `ExecPermit.methods.contains(&plan.method)` (binding.rs:393). `MethodRef` equality includes the stamp (lib.rs:352-358, 385-389), so a permit matches the exact issued reference. I found no code mapping a `MethodId` to an `Operation::Act { action }` name.
M8. `Resolved::action_ordinals` returns `MethodId`s in registration order (lib.rs:760-762), the only "action address" the store exposes.

---------------------------------------------------------------------
## FILE 06-authorization.md
---------------------------------------------------------------------

### 1. CONFIRMED

C1. "`establish` runs ... verifier first ... issuer trusted ... `(issuer, subject)` resolve ... no fallback to email" (l.39-43) -> OGAR ogar-auth/src/federation.rs:244-263 (doc 244-251, code 257-270) at OGAR 8fab880.
C2. "`VerifiedLogin` has private fields and no public constructor." (l.44) -> federation.rs:199-210 (doc says only producer is `establish`).
C3. "`AuthenticatedUser` has public fields. It is descriptive" (l.45-47) -> ogar-auth/src/user.rs:292-308.
C4. `acting_party()` on `AuthenticatedUser` (l.24, 63) -> user.rs:325-330.
C5. "`OgarRbac::authorize_user` takes an `AuthenticatedUser` ... calls the generic kernel `authorize_scoped` and adds only the identity." (l.48-50) -> ogar-rbac/src/lib.rs:135-162. Kiosk/unauthenticated not refused (l.46-47) -> lib.rs:146-153.
C6. "`authorize_identity` ... binds every membership of the user to the user's tenant and calls `authorize_memberships`. It returns a `MembershipDecision`" (l.51-54) -> ogar-rbac lib.rs:303-340 (membership scope is `ScopeSpec{tenant}` at 308-319).
C7. "A delegated login authorizes as the user whose authority the token carries. The acting party is recorded and holds no role." (l.55-56) -> ogar-rbac lib.rs:249-254.
C8. `ScopedDecision` decision/scope/field_mask and `visible_fields()` (l.29-30, 124-128) -> authorize.rs:165-200.
C9. "Stage 1 ... not Allow -> Deny or Escalate, scope None, field_mask FULL" (l.78) -> authorize.rs:216-224; `AccessDecision` Allow/Deny/Escalate -> access.rs:5-13.
C10. "Stage 2: fold the granting roles only; scope: ScopeSpec intersect, None is global and does not narrow; field_mask: union of role masks" (l.79-81) -> authorize.rs:225-250 (comment 230-235).
C11. "On a refused decision `field_mask` holds the open default (`FULL`). ... `visible_fields()` returns `EMPTY` unless the decision is `Allow`." (l.96-98) -> authorize.rs:184-199, 219-223.
C12. "`authorize_scoped` intersects the scopes of the granting roles ... `authorize_memberships` ... each granting membership adds its own scope, and the scopes are unioned" (l.99-102); "any unscoped grant means unrestricted" (l.83) -> authorize.rs:284-297, 325-340.
C13. `Operation` Read{depth}/Write{predicate}/Act{action} (l.76, 103-105) -> rbac.rs:346-363.
C14. `OpMask` READ, WRITE, CREATE, DELETE, ACT (l.106-107) -> rbac.rs:453-466; `OpMask::permits` is verb-only (rbac.rs:485-497); `grants_permit` rbac.rs:548-550.
C15. `ScopeSpec` fields tenant/predicate_key/path/deny and `admits`, `intersect` (l.136-142) -> rbac.rs:114-132, 144-151, 166-194.
C16. "`ScopeSpec::admits` ... true when the scope is not `deny` and the row's path lies at or below the scope's path. The tenant is matched by the store that owns the rows." (l.168-171) -> rbac.rs:144-151.
C17. "`intersect` never widens. `deny` absorbs. Two paths on different branches give `DENY`. Two different tenants give `DENY`. A `None` tenant (global) takes the other side's tenant." (l.172-176) -> rbac.rs:166-193.
C18. "`predicate_key` is reserved. It is OR-combined and not interpreted." (l.177-178) -> rbac.rs:118-120, 177.
C19. `ScopeSet` members/admits; `Membership` (l.144-147) -> rbac.rs:235-267, 302-308.
C20. `PropertySpec.marking`, `Marking` five variants, "The default is `Internal`" (l.148-158, 179-180) -> property.rs:41-58, 769-783.
C21. "DT64's `Resolved::mask_where` builds a field mask from meaning ... not an authorization decision." (l.181-184) -> lib.rs:715-725.
C22. `walk_rails_authorized` "applies `visible_fields()`, decides nothing" (l.197, 210) -> selection.rs:308-362.
C23. "No DT64, r2il, xlog, Quack or mask-risc code uses `lance-graph-rbac`" (l.236-237, 263) -> `lance-graph-rbac/Cargo.toml:1-9` depends on contract only; grep of all crates/**/Cargo.toml for "rbac" finds only lance-graph-dir-sim, lance-graph-consumer-conformance, lance-graph-callcenter (plus rbac itself); grep for rbac/ScopedDecision/lance_graph_rbac in the dt64, r2il, xlog, quack, mask-risc, foundry-parity-lab crates returned no files. (Search plus manifest read.)
C24. "The lab uses a stand-in: one reader denied one store id, written into the program." (l.239-241) -> method.rs:14-16, 252-257; fixture DENIED_STORE imported at method.rs:34.
C25. "`Binder::live` is named ... as the existing seam where a scope plane could enter" -> quack/src/bind.rs:92-96 (`fn live(&self, table) -> Mask`, "the plane every row of the table must pass"). The plan-side wording is outside this check.
C26. "`ExecPermit` ... who issues it is not decided" (l.244-247, 256) -> binding.rs:124-130.
C27. "`ResolvedIdentity::auth_classid` is documented as the identity for the audit." (l.249) -> lance-graph-rbac/src/auth.rs:256 (`ResolvedIdentity`), 275, 376-381.
C28. Gap statement G2 "per-row scope lane ... `visible_fields()` limits the bound fields ... not built" (l.234-238) -> .claude/plans/foundry-parity-v1.md:34 ("Apply it to a DT64 binding: per-row scope lane -> resident plane; `visible_fields()` limits bound fields"); DT64 scope is only the equality check in lane.rs:36-40, 96-97, 255-261.

### 2. STALE or WRONG

S1. Source list item "`.claude/plans/foundry-parity-v1.md` ... on branch `ccr-ca0562e7-hudvlz`" (l.262) -> the file is present in this main checkout (.claude/plans/foundry-parity-v1.md); PR #1996 from that branch was merged (83d532759). Also "(PR #1996)" on method.rs sources (l.261) is stale: method.rs is on main.
S2. Section 2 diagram edge `S1 -. "memberships route" .-> MS` (l.91) -> `authorize_memberships` does not run stage 1 or call `authorize`. It reads `rbac.memberships(actor, class)`, returns `Deny{"unknown role"}` on an empty list, and calls `grant_permits` per membership (authorize.rs:310-335). The doc says the verdict "matches `authorize` whenever `memberships` lists the roles of `actor_roles`" (authorize.rs:296-297), which is a consistency property, not a call.
S3. "The grant gate is a verb bit per class ... A finer check on depth, predicate or action name layers above it." (l.106-108) -> true of `OpMask::permits` (rbac.rs:485-497), but the reference `ClassGrants::grant_permits` already applies `can_read_at(depth)`, `can_write(predicate)`, `can_act(action)` inside the stage-1 gate (authorize.rs:153-161). The finer check is not a separate later stage in that implementation.
S4. Line drift: `ScopedDecision`/`visible_fields` cited authorize.rs:164-208, actual 165-200; `authorize_scoped` 210-253, actual 210-251; `MembershipDecision` 255-278 / 255-345, actual 256-282 and 299-341; `intersect` rbac.rs:153-191, actual 153-194 (fn body 166-194); `mask_where` lib.rs:714-723, actual 719-725 (doc starts 715). OGAR cites, checked at OGAR 8fab880 (diagram author's OGAR revision unknown): `VerifiedLogin` cited federation.rs:206-216, actual 205-210; `establish` 244-283, actual 244-279; `AuthenticatedUser` user.rs:290-335, actual 292-331; `authorize_user` lib.rs:122-162, actual 129-162; `IdentityActors`/`authorize_identity` 225-345, actual 238-340.

### 3. MISSING RELATIONSHIPS (each checked)

M1. Where authorization sits relative to method admission: nowhere in the code read. `BindStore::admit` takes only a `MethodRef` (lib.rs:1180). `ExecPermit` is a `Vec<MethodRef>` (binding.rs:127-130). I found no function that turns a `ScopedDecision`/`MembershipDecision` or an `Operation::Act { action }` into an `ExecPermit` or a `MethodRef`, and `Resolved::action_ordinals` yields `MethodId`s, not action names (lib.rs:760-762). Section 4's "ACT -. candidate only .-> ISS" matches this.
M2. Type gap between decision scope and DT64 scope: decision scope is `ScopeSpec{tenant: Option<u64>, predicate_key, path: ScopePath, deny}` or `ScopeSet` (rbac.rs:115-132, 236-238); DT64's `Scope` is `{id: u64}` (dr.rs:519-522) compared by equality against `LaneAddr.scope` (lane.rs:255-261). No conversion exists between them in the code read. `ScopeSpec::admits` takes a `ScopePath` row address (rbac.rs:149-151), which DT64 lanes do not carry.
M3. Other crates that do depend on `lance-graph-rbac`, absent from the diagram: lance-graph-dir-sim (feature `rbac`, Cargo.toml:55, 78), lance-graph-consumer-conformance (Cargo.toml:23), lance-graph-callcenter (Cargo.toml:21).
M4. The membership route's scope source: `IdentityActors::memberships_of` gives every role the same tenant-only `ScopeSpec` (ogar-rbac lib.rs:308-319), so on that route the decision scope is tenant-only unless the binding overrides it.
M5. `ScopedDecision::visible_fields` is positional over `WideFieldMask`; `LaneRequest.fields` is a `WideFieldMask` over the same positions (lane.rs:94, 192-195). The two masks are the same type, so "limit bound fields to `visible_fields()`" is an intersection of two masks in principle; no code in dt64 does it (checked lane.rs:171-216: `req.fields` is used unmodified).

---------------------------------------------------------------------
## UNVERIFIED (not checked, and why)

U1. 03 l.370 and 06: reference to `04-read-path.md` and its "authorization gap". Not in the three files assigned.
U2. 06 OGAR crates: only the cited ranges and the functions named were read, at OGAR 8fab880. The OGAR revision the diagram author used is unknown, so the line drift in S4 may be OGAR movement, not an error.
U3. 06 l.12 "The OGAR crates are in the OGAR repository, not in lance-graph": true for the paths read (no ogar-auth or ogar-rbac under the checkout's crates/ list), but I did not read OGAR's whole tree.
U4. 05 l.80 and others cite "`lib.rs:1-27` (what DT64 stores)": confirmed 1-30 (C9); the exact end line is a judgment call, not marked.
U5. 05 `xlog bag_count_admitted` "(`Population::Bag`)" runtime behaviour: read the function (xlog lib.rs:384-403) but not `bag_count_of` or the `Intake` constructor.
U6. 05 `normalize`/`lower` internals ("normalize, lower" box): only `run_with_program` (lower.rs:62-92) and `rel_from_set`/`to_set` (normalize.rs:82-130) were read; the claim "every `SetExpr` lowers to one fold under `Keep`" (method.rs:117-118) rests on a test `every_guard_shape_lowers_to_one_reusable_fold` that I did not open.
U7. 05 l.209-218 diagram text "The intake re-checks its `MapTable` frame": S2 is based on `MapTable::map`; I did not run anything, so no behavioural confirmation.
U8. Nothing here claims the code compiles or tests pass; no cargo was run.

---------------------------------------------------------------------
## Counts (statements checked = CONFIRMED + STALE items listed above)

- 03-dr-dt64-binding.md: 33 confirmed, 4 stale/wrong, 7 missing-relationship items; 37 statements checked.
- 05-methods.md: 30 confirmed, 5 stale/wrong, 8 missing-relationship items; 35 statements checked.
- 06-authorization.md: 28 confirmed, 4 stale/wrong, 5 missing-relationship items; 32 statements checked.
