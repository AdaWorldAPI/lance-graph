# RESEARCH: Primzahl-Encoding vs 2ⁿ Bitpacking vs Zeckendorf

## FOR: Opus 4.6 Zeckendorf Proof Session

---

## THREE NUMBER SYSTEMS FOR DISTANCE ENCODING

```
System          Basis           Uniqueness        Operation
──────          ─────           ──────────        ─────────
2ⁿ bitpacking   powers of 2     trivial (binary)  addition
Zeckendorf      Fibonacci nums  Zeckendorf thm    addition (non-consecutive)
Primzahl        prime numbers   Fundamental thm   multiplication
```

## COMPARISON MATRIX

### 2ⁿ Bitpacking (current: BF16 = 16 bits)

```
Encoding:  value → binary → bits
  0.5 → 0x3F00 (BF16) → 16 bits fixed

Overhead:     0 (native hardware format)
Precision:    7-bit mantissa = 1/128 ≈ 0.0078
Compression:  none (every value = 16 bits, no correlation exploited)
Re-encode:    idempotent after 1 iteration (proven)
Hardware:     native (VDPBF16PS, every CPU)

For 17-dim vector (Base17):
  17 × 16 bits = 272 bits = 34 bytes
  No cross-dimension correlation used.
  Each dimension encoded independently.
```

### Zeckendorf (ZeckF64 in bgz-tensor)

```
Encoding:  value → sum of non-consecutive Fibonacci numbers
  30 = F(8) + F(4) + F(2) = 21 + 8 + 2   (no consecutive Fibs)
  → bits: ...00010101 (positions 8,4,2 set)

Overhead:     ~1 bit (no-consecutive constraint = ~1.44 bits per Fib vs 1.0 for binary)
              → 44% overhead per value for the constraint
Precision:    depends on largest Fib used. F(64) ≈ 1.06e13
              → 64 Fibs cover [0, 1.06e13] with ~44 significant bits
Compression:  ADDITIVE correlation between dimensions
              → shared Fibonacci components = shared bits
              → 17 correlated dimensions: ~60-70% of independent size
Re-encode:    idempotent (unique representation)
Hardware:     scalar (no native Fibonacci instruction)

For 17-dim vector:
  IF dimensions share Fibonacci components (correlated):
    ~190-210 bits ≈ 24-26 bytes (vs 34 bytes BF16 = 1.3-1.4×)
  IF independent:
    17 × 44 × 1.44 = ~1078 bits ≈ 135 bytes (WORSE than BF16)
  
Sweet spot: CORRELATED vectors (most transformer weights are)
```

### Primzahl (proposed)

```
Encoding:  value → product of primes → bit per prime
  30 = 2 × 3 × 5 → bits 0,1,2 set → 0b00000111

  With K primes and 1 bit per prime:
    only squarefree numbers representable (no exponents)
    range: product of first K primes

  With K primes and E bits per prime (exponents 0..2^E-1):
    any integer up to (p_K)^(2^E-1) representable
    K×E bits total

Overhead analysis for distance values (cosine in [-1, +1]):

  Step 1: quantize cos to integer
    cos × 32768 → integer [0, 32767] (15-bit unsigned)
    or: cos × 127 → integer [-128, 127] (i8, current signed path)

  Step 2: factorize
    For i8 values [-128, 127]:
      127 = prime (the 31st prime) → needs 31 prime-bits minimum
      128 = 2⁷ → needs 7 bits for exponent of prime 2

    For BF16 cosine values (quantized to 15-bit):
      32767 = 7 × 31 × 151 → needs primes up to 151 (36th prime)
      → 36 bits minimum (1 per prime, squarefree only)
      → 36 + sign = 37 bits vs 16 bits BF16 = 2.3× LARGER

  For EXPONENT encoding (2 bits per prime):
    36 primes × 2 bits = 72 bits = 9 bytes vs 2 bytes BF16 = 4.5× LARGER

SINGLE VALUE: Primzahl is 2-5× WORSE than BF16. No advantage.

Step 3: BUT for VECTORS (cross-dimension shared factors):
  cos[0] = 30 = 2×3×5           → primes {0,1,2}
  cos[1] = 42 = 2×3×7           → primes {0,1,3}
  cos[2] = 70 = 2×5×7           → primes {0,2,3}
  
  Shared: prime 2 appears in ALL THREE → encode ONCE
  Matrix:
    prime 2: dims {0,1,2}    = 0b111 (3 bits for which dims)
    prime 3: dims {0,1}      = 0b011
    prime 5: dims {0,2}      = 0b101
    prime 7: dims {1,2}      = 0b110
  
  Total: 4 primes × (1 + 3 bits) = 16 bits for 3 values
  vs BF16: 3 × 16 = 48 bits
  = 3× compression IF primes are shared

For 17-dim Base17 vector:
  Each prime needs: 1 bit (present) + 17 bits (which dims) = 18 bits
  IF 10 primes cover all 17 dims: 10 × 18 = 180 bits = 22.5 bytes
  vs BF16: 34 bytes = 1.5× compression

  IF 30 primes needed: 30 × 18 = 540 bits = 67.5 bytes
  vs BF16: 34 bytes = 0.5× (WORSE)
```

## SUMMARY TABLE

```
                    Single Value    17-dim Vector       17-dim Vector
                    (bits)          (independent)       (correlated)
                    ──────          ─────────────       ────────────
2ⁿ BF16            16              272 (34 bytes)      272 (34 bytes)
Zeckendorf          ~23 (+44%)      ~1078 (135 B) ✗    ~200 (25 B) ✓
Primzahl (1-bit)    37 (+131%)      ~629 (79 B) ✗      ~180 (22 B) ✓
Primzahl (2-bit)    72 (+350%)      ~1224 (153 B) ✗    ~360 (45 B) ✗

Winner per case:
  Independent dims:   BF16 (no overhead, hardware native)
  Correlated dims:    Zeckendorf ≈ Primzahl (both exploit correlation)
  Single value:       BF16 (smallest, native)
  Hardware:           BF16 (VDPBF16PS) > Zeckendorf (scalar) > Primzahl (factorize = slow)
```

## THE REAL QUESTION

```
Transformer weight vectors: HOW correlated are the 17 Base17 dimensions?

IF highly correlated (same primes/Fibs across dims):
  → Zeckendorf or Primzahl compress 1.3-1.5× vs BF16
  → worth the overhead for storage/transmission
  → NOT for runtime (decode too slow for MatVec hot path)

IF weakly correlated:
  → BF16 wins everywhere
  → no benefit from shared factors

EMPIRICALLY (from StackedN SPD=32, Pearson 0.996):
  The 17 dims ARE correlated (golden-step maps correlated weights to same bin).
  BUT: the correlation is captured by the AVERAGING in StackedN.
  After averaging: the 17 dim means are LESS correlated (averaging decorrelates).
  
  → Primzahl encoding on RAW octave data: potentially 1.5× compression
  → Primzahl encoding on AVERAGED Base17: probably no benefit (already decorrelated)
```

## HYBRID: Zeckendorf POSITIONS × Primzahl VALUES

```
Zeckendorf:  WHERE on the φ-spiral (positions, additive)
Primzahl:    WHAT the value is (magnitude, multiplicative)

Combined encoding per Base17 entry:
  position = Zeckendorf decomposition of bin index (which bin on the spiral)
  value = Primzahl decomposition of quantized cosine (which primes active)

The Zeckendorf position IS the address (highheelbgz SpiralAddress).
The Primzahl value IS the data at that address.

This is dual: position space (additive/Fibonacci) × value space (multiplicative/prime).
Like: complex numbers have real (additive) + imaginary (multiplicative) part.
```

## WHAT THE OPUS 4.6 SESSION SHOULD PROVE

```
1. Zeckendorf reconstruction bound:
   ε ≤ C × (k/n)^(log φ/log 2) × ‖f''‖
   (POSITION space, additive, proven structure exists)

2. Primzahl encoding efficiency:
   For a 17-dim vector with inter-dim correlation ρ:
   bits(Primzahl) ≈ bits(BF16) × (1 - ρ) + overhead × ρ
   (VALUE space, multiplicative, needs proof)

3. Combined bound:
   The product Zeckendorf × Primzahl encoding has total error:
   ε_total ≤ ε_position + ε_value + ε_position × ε_value
   where the cross-term captures position-value interaction.
   
4. Re-encode safety of the combined system:
   IF Zeckendorf is idempotent (proven: unique representation)
   AND Primzahl is idempotent (proven: unique factorization)
   THEN the product is idempotent (both components stable)
   → x256 re-encode safety for the combined system

5. Optimal bit allocation:
   Given B total bits for a 17-dim vector:
   How many bits for Zeckendorf (positions) vs Primzahl (values)?
   → probably: BF16 for values (hardware) + Zeckendorf for positions (optimal coverage)
   → Primzahl only if inter-dim correlation > threshold (empirical: check on real weights)
```
