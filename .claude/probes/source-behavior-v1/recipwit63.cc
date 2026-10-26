// Amendment 6.3 addendum: reciprocal observability gate for TFLM int8 SOFTMAX.
//
// Searches reachable rows for an output byte that changes when ONLY the reciprocal is replaced.
// Everything else stays the source's: prepared weights come from TFLM's own exp_on_negative_values and
// Rescale<12> (read from `trace63 softexp`), the exponent from the source num_bits_over_unit, the output
// from SaturatingRoundingDoublingHighMul + RoundingDivideByPOT, -128, clamp (reference/softmax.h:115-139).
//
// Row family (reachable int8 rows): k copies of the row maximum and m copies of max+d, k = 1..511,
// d = -255..-1 (admitted by diff_min), m = 0..M_MAX. The sum S = k*B[0] + m*B[d] depends only on (k, m, d);
// only two distinct output bytes exist per row (the max and the d element). Admission: S <= INT32_MAX and
// 0 <= exponent <= 31 (the source abort guard), i.e. a real sum below 512.
//
// Arms per row: source reciprocal (TFLM GetReciprocal), exact reciprocal (round(2^62 / (S << headroom)),
// clamped to INT32_MAX), and, as the negative control, the source reciprocal + DELTA for each DELTA given.
//
// stdin:  "C idx" then 256 lines "E d admitted rescaled exp_raw acc_raw" (trace63 softexp output body)
// argv:   M_MAX DELTA1 [DELTA2 ...]
// stdout: per calibration "S idx rows witnesses nearest_margin" and up to 5 "W idx k m d elem src exact"
//         lines, then per delta "N idx delta rows_detected".
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/fixedpoint.h"

using namespace tflite::micro;

static inline int out_byte(int32_t sh, int32_t a, int ex) {
  int32_t v = RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(sh, a), ex) - 128;
  return v < -128 ? -128 : (v > 127 ? 127 : v);
}
// distance of the pre-division product from the nearest rounding half, in raw units
static inline int64_t margin(int32_t sh, int32_t a, int ex) {
  int64_t p = SaturatingRoundingDoublingHighMul(sh, a);
  if (ex == 0) return INT64_MAX;
  int64_t step = int64_t{1} << ex, r = ((p % step) + step) % step, half = step >> 1;
  int64_t d = r - half; return d < 0 ? -d : d;
}

int main(int argc, char** argv) {
  const long M_MAX = std::atol(argv[1]);
  std::vector<int32_t> deltas;
  for (int i = 2; i < argc; ++i) deltas.push_back(std::atoi(argv[i]));
  char tag[8]; int idx;
  while (std::scanf("%7s %d", tag, &idx) == 2) {
    int adm[256]; int32_t A[256], B[256];
    for (int i = 0; i < 256; ++i) {
      char e[4]; int d, ad, resc; long long ex_raw, acc;
      if (std::scanf("%3s %d %d %d %lld %lld", e, &d, &ad, &resc, &ex_raw, &acc) != 6) return 2;
      adm[d + 255] = ad; A[d + 255] = (int32_t)ex_raw; B[d + 255] = (int32_t)acc;
    }
    const int64_t b0 = B[255]; const int32_t a0 = A[255];
    long long rows = 0, wit = 0; int64_t best = INT64_MAX;
    std::vector<long long> det(deltas.size(), 0);
    for (int d = -255; d <= -1; ++d) {
      if (!adm[d + 255]) continue;
      const int64_t bd = B[d + 255]; const int32_t ad = A[d + 255];
      for (int k = 1; k <= 511; ++k) {
        for (long m = 0; m <= M_MAX; ++m) {
          if (m > 0 && bd == 0 && m > 1) break;             // S constant in m: one row suffices
          const int64_t S = k * b0 + m * bd;
          if (S > INT32_MAX) break;
          int nb; const int32_t sh = GetReciprocal((int32_t)S, 12, &nb);
          const int ex = nb + 23;
          if (ex < 0 || ex > 31) break;                      // source abort guard; S grows with m
          ++rows;
          const int hpo = CountLeadingZeros(static_cast<uint32_t>(S));
          const __int128 den = (__int128)S << hpo;
          __int128 q = ((((__int128)1) << 62) * 2 + den) / (2 * den);   // round half up, positive
          const int32_t shx = q > INT32_MAX ? INT32_MAX : (int32_t)q;
          const int32_t elems[2] = {a0, ad};
          for (int e = 0; e < (m > 0 ? 2 : 1); ++e) {
            const int32_t a = elems[e];
            const int bs = out_byte(sh, a, ex), bx = out_byte(shx, a, ex);
            const int64_t mg = margin(sh, a, ex); if (mg < best) best = mg;
            if (bs != bx) {
              if (wit < 5) std::printf("W %d %d %ld %d %d %d %d\n", idx, k, m, d, e, bs, bx);
              ++wit;
            }
            for (size_t t = 0; t < deltas.size(); ++t) {
              int64_t s2 = (int64_t)sh + deltas[t]; if (s2 > INT32_MAX) s2 = INT32_MAX;
              if (out_byte((int32_t)s2, a, ex) != bs) { ++det[t]; }
            }
          }
        }
      }
    }
    std::printf("S %d %lld %lld %lld\n", idx, rows, wit, (long long)best);
    for (size_t t = 0; t < deltas.size(); ++t) std::printf("N %d %d %lld\n", idx, deltas[t], det[t]);
    std::fflush(stdout);
  }
  return 0;
}
