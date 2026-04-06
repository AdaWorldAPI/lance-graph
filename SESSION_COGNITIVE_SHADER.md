# SESSION: Cognitive Shader — Wave/Particle Duality on Shared Memory GPU

## THE ARCHITECTURE

```
Centroid = WAVE   (L1-L3, immutable distance tables, interference, superposition)
VSA      = PARTICLE (L4, mutable i8 accumulators, bundles, measurement/collapse)

commit() = MEASUREMENT:
  Wave (4096 continuous amplitudes) → collapses to → Particle (16384 discrete bits)
  Particle biases next Wave (L4 recognize → perturbation bias)
  = quantum measurement process implemented as threshold crossing

Temperature = UNCERTAINTY PRINCIPLE:
  Low T → sharp position (which centroid), blurry momentum (how it propagates)
  High T → sharp momentum (convergence path), blurry position (many candidates)
  T=1.0 → complementarity (balanced wave/particle behavior)
```

## GPU MAPPING

### Distance Table = Texture

```
BF16 table 256×256 = 128 KB → fits ENTIRELY in GPU shared memory
  Intel XMX SLM: 64 KB per sub-slice (2 sub-slices = 128 KB ✓)
  AMD LDS: 64 KB per workgroup (need 2 workgroups or use L1)
  NVIDIA: 48-164 KB per SM (fits in one SM ✓)

The table is READ-ONLY during thinking.
Load once into shared memory. Sample forever.
= Flash Attention principle: "data never leaves SRAM"
= Doctrine #15: "Distance tables = textures"
```

### Energy Vector = Framebuffer

```
energy[4096] × f32 = 16 KB → fits in shared memory
  Or: energy[256] × f32 = 1 KB (for 256-centroid lenses)
  
READ-WRITE each cycle.
Normalized after each cycle (parallel reduction in shared memory).
= Double-buffered framebuffer (read current, write next, swap)
```

### MatVec = Shader Dispatch

```
One MatVec cycle = one compute shader dispatch:

kernel matvec_cycle(
    table: texture<bf16, 2d>,     // 256×256 BF16 distance table
    energy_in: buffer<f32>,        // current energy (read)
    energy_out: buffer<f32>,       // next energy (write)
    temperature: uniform<f32>,     // per-role temperature
    bundle: shared<u64>,           // prime fingerprint (broadcast)
) {
    let j = global_id.x;          // which atom am I?
    let n = 256;                   // table dimension
    
    var acc = 0.0f;
    for (var i = 0u; i < n; i++) {
        let e_i = energy_in[i];
        if (e_i < 1e-10) { continue; }
        let dist = bf16_to_f32(table[i * n + j]);  // BF16 → f32 (bit shift)
        acc += dist * e_i;                          // signed: positive excites, negative inhibits
    }
    
    // Clamp negative (inhibited atoms die)
    acc = max(acc, 0.0);
    
    // Temperature excitation (softmax/T)
    acc = exp(acc / temperature);
    
    energy_out[j] = acc;
    
    // Barrier → normalize (parallel reduction)
    workgroup_barrier();
    // ... parallel sum reduction ...
    energy_out[j] /= total;
}

Dispatch: 256 threads (one per atom), 10 dispatches (10 cycles)
Total: 2,560 thread invocations × 256 table lookups = 655,360 operations
At 1 GHz GPU clock: < 1μs per thought
```

### L4 = Storage Buffer (persistent)

```
L4 accum[16384] × i8 = 16 KB → storage buffer
  Persists across shader dispatches (across thoughts)
  Updated AFTER commit (not during thinking)

kernel l4_learn(
    accum: buffer<i8>,           // L4 personality (read-write, persistent)
    bundle: buffer<u8>,          // committed bundle (2048 bytes = 16384 bits)
    reward: uniform<i8>,         // +1 good, -1 bad
) {
    let k = global_id.x;        // which accumulator?
    let byte_idx = k / 8;
    let bit_idx = 7 - (k % 8);
    let is_set = (bundle[byte_idx] >> bit_idx) & 1;
    
    if (is_set == 1) {
        accum[k] = clamp(accum[k] + reward, -128, 127);  // saturating add
    } else {
        accum[k] = clamp(accum[k] - reward, -128, 127);  // saturating sub
    }
}

Dispatch: 16384 threads, 1 dispatch after commit
= Hebb learning in parallel, < 1μs
```

### Bundle Perturbation = Shared Memory Broadcast

```
kernel bundle_perturb(
    energy: buffer<f32>,
    fingerprints: buffer<u64>,     // one per centroid
    source_fps: shared<u64>[8],    // up to 8 sources, broadcast via shared memory
    n_sources: uniform<u32>,
) {
    let j = global_id.x;
    
    // Compute bundle (majority vote) — ONCE per workgroup
    if (local_id.x == 0) {
        var bundle = 0u64;
        for (var bit = 0u; bit < 64u; bit++) {
            var count = 0u;
            for (var s = 0u; s < n_sources; s++) {
                if ((source_fps[s] >> bit) & 1u64 != 0) { count++; }
            }
            if (count > n_sources / 2) { bundle |= 1u64 << bit; }
        }
        shared_bundle = bundle;
    }
    workgroup_barrier();
    
    // Hamming distance from bundle to this centroid's fingerprint
    let hamming = popcount(shared_bundle ^ fingerprints[j]);
    let similarity = 1.0 - 2.0 * f32(hamming) / 64.0;
    
    energy[j] += max(similarity, 0.0);
}
```

## COMPLETE PIPELINE ON GPU

```
Frame 0 (initialization):
  Upload BF16 distance table → texture (128 KB, once)
  Upload centroid fingerprints → buffer (256 × 8 = 2 KB, once)
  Upload L4 accum → storage buffer (16 KB, once)

Frame N (one thought):
  1. Bundle perturbation → energy       (1 dispatch, 256 threads)
  2. MatVec cycle × 10 → converged energy (10 dispatches, 256 threads each)
  3. Commit: find top-K peaks           (1 dispatch, parallel reduction)
  4. L4 learn: Hebb update              (1 dispatch, 16384 threads)
  5. Read back: BusDto                  (16 bytes, negligible transfer)

Total: 13 dispatches per thought
Data movement: 0 (everything stays in shared memory / storage buffers)
Latency: < 50μs per thought at 1 GHz GPU clock

vs CPU: ~7ms per thought (F32x16 SIMD)
= 140× speedup from GPU shared memory
```

## INTEL XMX SPECIFICS

```
Intel Arc / Battlemage / Celestial:

XMX (Xe Matrix Extensions):
  8×8 systolic array
  i8 × i8 → i32 accumulation
  = EXACTLY our L4 accum pattern (i8 weights × i8 bundle → i32 score)

For MatVec on XMX:
  Reshape energy[256] as 16×16 matrix
  Reshape table row as 16×16 matrix
  XMX: 16×16 × 16×16 → 16×16 (one XMX instruction)
  = 256 multiply-accumulates in ONE instruction
  = entire cycle in 16 XMX instructions (256×256 table)

SLM (Shared Local Memory):
  64 KB per sub-slice
  BF16 table 256×256 = 128 KB → 2 sub-slices or halve table
  Or: use 128×128 table (L2 lens) = 32 KB → fits ONE sub-slice

EU (Execution Units):
  Gen3 Arc A770: 32 Xe-Cores, 512 EU
  Each EU can run the MatVec kernel independently
  = 512 parallel thoughts (different inputs, same table)
  = batch inference: 512 texts × 50μs = 25ms for 512 texts
```

## AMD RDNA3/4 SPECIFICS

```
WMMA (Wave Matrix Multiply Accumulate):
  Wavefront = 32 or 64 threads
  WMMA: 16×16 × 16×16 → 16×16 (bf16 or i8)
  Same principle as XMX but different wave size

LDS (Local Data Share):
  64 KB per workgroup
  BF16 table 128 KB → need LDSS (cross-workgroup) or split

CU (Compute Units):
  RX 7900 XTX: 96 CU × 64 threads = 6144 parallel
  = 6144 parallel thoughts if table fits in L1
```

## WAVE/PARTICLE ON GPU

```
The GPU naturally implements wave/particle duality:

WAVE phase (MatVec, parallel):
  All 256 atoms updated SIMULTANEOUSLY (SIMT execution)
  Each thread sees ALL other atoms' energy (shared memory)
  = wave propagation: all paths computed at once
  = no sequential bottleneck

PARTICLE phase (commit, sequential):
  Reduce to top-K (parallel reduction → sequential decision)
  One winner selected (measurement / collapse)
  L4 updated (Hebb, parallel but applied to ONE bundle)
  = particle measurement: superposition → definite state

The GPU dispatch model IS wave/particle:
  Dispatch = wave (all threads compute)
  Barrier = measurement (synchronize, reduce)
  Next dispatch = new wave (informed by measurement)
```

## CONNECTION TO ADA-CONSCIOUSNESS

```
ada-consciousness (Python, Upstash):
  VSA.bundle_all(*states)              → superposition
  VSA.collapse_triangle(q, states)     → measurement
  gate → winner (aware/unaware/uncertain)

thinking-engine (Rust, CPU/GPU):
  bundle_perturb(sources, fingerprints) → superposition (wave)
  commit() → BusDto                     → measurement (particle)
  Temperature → CollapseGate            → uncertainty principle

SAME SEMANTICS. Different runtime:
  Python + Upstash: ~100ms per thought (network + Redis)
  Rust CPU:         ~7ms per thought (F32x16 SIMD)
  Rust GPU:         ~50μs per thought (shared memory, XMX)
  
  2000× speedup from Python → GPU
  While keeping the SAME wave/particle/measurement model
```

## WHAT NEEDS TO HAPPEN

```
1. Write WGSL compute shader (WebGPU) or GLSL compute (Vulkan)
   → the MatVec kernel is ~30 lines
   → bundle perturbation is ~20 lines
   → L4 learn is ~15 lines

2. Use wgpu crate (Rust, cross-platform: Vulkan/Metal/DX12/WebGPU)
   → same binary runs on Intel/AMD/NVIDIA/Apple
   → WebGPU in browser = cognitive shader in the browser

3. Upload BF16 table + fingerprints + L4 once
   → think() = 13 dispatches, read back 16 bytes
   → batch: 512 texts in 25ms

4. For Intel XMX specifically:
   → oneAPI Level Zero (low-level, maximum control)
   → or wgpu with Vulkan backend (portable)
```
