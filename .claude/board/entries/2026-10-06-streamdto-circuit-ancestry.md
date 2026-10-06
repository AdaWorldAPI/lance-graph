# 2026-10-06 — D-ANCESTRY-K1-0 + D-STREAMDTO-0: the k → k+1 circuit was intended and left open at one seam

## BASIS

`.claude/v3/` first (`COMPONENT-MAP.md` :55-61, :80, :106-110;
`knowledge/v3-substrate-primer.md` §3; `MODULE-TABLE.md` 2026-07-10(2)
ancestry census; `soa_layout/tenants.md`, `routing.md`), then the code.
Git archaeology is not available: the clone is shallow and every symbol's
`git log -S` stops at the #1327 merge. Code and board only.

## VERIFIED-IN-CODE — ancestry map (old → current)

| v3 name | current carrier | status |
|---|---|---|
| Φ `StreamDto` (perturbation ingress) | `thinking_engine::dto::StreamDto`; its fields reach `engine_bridge::ingest_codebook_indices` from the lab serve/grpc wire DTOs | RETAIN. The struct has no production constructor; ingest writes the BindSpace singleton only |
| Ψ `PerturbationDto` | `dispatch_from_top_k` consumes `top_k` only | REUSE semantics; `energy` never reaches the driver (E-THE-PERTURBATION-FIELD-NEVER-REACHED-THE-MASK-ALU-1) |
| B `BusDto` (commit, `converged`/`cycle_count` = D-MBX-A6 outcome) | `dispatch_busdto` / `unbind_busdto` (singleton); `MailboxSoA::cast_on_behalf` (W4a, `with-planner`) | RETAIN |
| Γ `ThoughtStruct` | no consumer on the circuit | not evaluated |
| shader cycle | `ShaderDriver::run` (`&self`): reads each row's `CausalEdge64`, `s_idx` as cascade query; emits up to 8 CE64 in `ShaderBus` | RETAIN |
| commit row | `EmitMode::Persist` → `persisted_row = top_k[0].row`; contract doc still says "Commit to BindSpace via CollapseGate" | RETAIN; stale prose |
| write-back | `engine_bridge::persist_cycle` writes `emitted_edges[0]` into the singleton | SUPERSEDED for the mailbox arm (BindSpace = RETIRE W7) |
| substrate | `MailboxSoA<1024>` via the `mailbox-thoughtspace` read shim (`backing.rs`) | RETAIN |
| commit boundary | `MailboxSoA::write_row` (cycle-gated) + `tick` | RETAIN |
| emitted edge → mailbox row | none: the driver owned the mailbox read-only (`mailbox(&self)`), `run` is `&self` | MISSING → completed |
| `StreamDto` → SoA ingress | none (ingest wrote the BindSpace singleton, superseded by the SoA) | MISSING → completed: `ingest_codebook_indices_soa` (cycle-gated `write_row`, shared encoding) |

## DECISION — Path B

The circuit was intended (`EmitMode::Persist`, `persist_cycle`, the read
shim, W4a cast pairing) and never closed. Smallest completion, both
ends at the SoA: `ingest_codebook_indices_soa` (ingress) and
`ShaderDriver::mailbox_mut(&mut self, id)` (write-back). No new DTO, scheduler, or
temporal type; `StreamDto` untouched; #1370 law untouched.
REVISIT WHEN: W5 multi-mailbox routing or W7 BindSpace retirement moves
the persist call site.

## MEASURED — `streamdto_circuit_probe.rs`

6 tests, 9 disable runs red (no write, no tick, wrong row, BE restore,
temporal from cycle, eligibility from the uncommitted word, SoA encoding
drift, SoA ingest bypassing `write_row`, SoA ingest not growing
`populated`); plus a default-feature lib test of the SoA arm.

- Cycle k: 8 hits, 8 emitted, row 4 persisted (`s_idx` 4). Cycle k+1 reads
  it and its output changes; without the write k+1 repeats k.
- The edge's only read in `run` is `s_idx`, keyed on `s_idx / 4`. F/C are
  revised and discarded (TD-INT-10, "observed only"); bits 59..63 unread.
  A write-back to block 0, or to a block whose distance-0 self-hit carries
  the same predicates, is invisible to k+1 (both hit while building the
  probe).
- #1370 eligibility at k+1 reads the committed row and is unchanged by the
  circuit: the shader stamps code 0, which under Pearl S grants `0x20`, the
  same as an unwritten word. Code 7 in the same row gives `0x24`.
- `StreamDto::timestamp` lands in `temporal` as the caller's number; `tick`
  never touches it; the dispatch's `cycle_index` is passed as `pack`'s v1
  `temporal`, which v2 drops.
- Restart: the committed word as LE bytes, restored into a fresh mailbox 7
  cycles later, gives the same k+1.
- Visibility does not wait for `tick` (no double buffer, see
  D-CE64-NEXTSTATE-0); k/k+1 separation is the owner's `&mut` plus the
  cycle gate.

## OPEN

- No production writer of bits 59..63 (`ISS-NO-EVIDENCE-WRITER-FOR-EPISTEMIC-STATE`), so the circuit cannot move eligibility.
- `CognitiveShaderBuilder` still requires a BindSpace on the mailbox arm (W7).
- `run` reads only `s_idx` from the edge; whether F/C/state should steer the cascade is undecided.
