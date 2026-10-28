// Amendment 6.3 addendum: exhaustive SUPERSET check of the reciprocal stage of TFLM int8 SOFTMAX.
//
// For a fixed calibration, an admitted output byte depends only on the row sum S (Q12.19 raw) and the
// element's prepared weight A[d] (reference/softmax.h:115-139). Every admitted row has
//   B[0] <= S < 2^28   (the max element contributes B[0]; exponent <= 31 <=> num_bits_over_unit <= 8)
// and every admitted element has a weight A[d] with d in [diff_min, 0]. This program checks EVERY S in
// that interval against EVERY admitted weight: a set that contains all reachable (S, d) pairs. If no byte
// changes when the source reciprocal is replaced by the correctly rounded one, the substitution is
// output-equivalent on the whole admitted domain of that calibration. A pair found here is only a
// CANDIDATE: it must still be shown reachable by a real int8 row.
//
// stdin: as recipwit63 ("C idx" + 256 "E ..." lines). stdout: "A idx s_lo s_hi s_recip_differs candidates"
// and one "X idx S d src exact" line per candidate per calibration.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/fixedpoint.h"

using namespace tflite::micro;

static inline int out_byte(int32_t sh, int32_t a, int ex) {
  int32_t v = RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(sh, a), ex) - 128;
  return v < -128 ? -128 : (v > 127 ? 127 : v);
}

int main(int argc, char** argv) {
  // negative control: `recipall63 DELTA` replaces the exact reciprocal by source + DELTA
  const long long delta = argc > 1 ? std::atoll(argv[1]) : 0;
  char tag[8]; int idx;
  while (std::scanf("%7s %d", tag, &idx) == 2) {
    int adm[256], dval[256]; int32_t A[256], B[256]; int na = 0; int32_t W[256];
    for (int i = 0; i < 256; ++i) {
      char e[4]; int d, ad, resc; long long ex_raw, acc;
      if (std::scanf("%3s %d %d %d %lld %lld", e, &d, &ad, &resc, &ex_raw, &acc) != 6) return 2;
      adm[d + 255] = ad; A[d + 255] = (int32_t)ex_raw; B[d + 255] = (int32_t)acc; dval[d + 255] = d;
      if (ad) { W[na] = (int32_t)ex_raw; ++na; }
    }
    std::sort(W, W + na); na = (int)(std::unique(W, W + na) - W);   // distinct admitted weights
    const int64_t lo = B[255], hi = (int64_t{1} << 28) - 1;
    long long differs = 0, cand = 0;
    for (int64_t S = lo; S <= hi; ++S) {
      int nb; const int32_t sh = GetReciprocal((int32_t)S, 12, &nb);
      const int ex = nb + 23;
      if (ex < 0 || ex > 31) continue;
      const int hpo = CountLeadingZeros(static_cast<uint32_t>(S));
      const __int128 den = (__int128)S << hpo;
      const __int128 q = ((((__int128)1) << 62) * 2 + den) / (2 * den);
      int32_t shx = q > INT32_MAX ? INT32_MAX : (int32_t)q;
      if (delta) { const long long p = (long long)sh + delta; shx = p > INT32_MAX ? INT32_MAX : (int32_t)p; }
      if (shx == sh) continue;
      ++differs;
      for (int j = 0; j < na; ++j) {
        const int bs = out_byte(sh, W[j], ex), bx = out_byte(shx, W[j], ex);
        if (bs != bx) {
          {  // every candidate is printed; reach63 decides reachability
            int d = 0; for (int i = 0; i < 256; ++i) if (adm[i] && A[i] == W[j]) { d = dval[i]; break; }
            std::printf("X %d %lld %d %d %d\n", idx, (long long)S, d, bs, bx);
          }
          ++cand;
        }
      }
    }
    std::printf("A %d %lld %lld %lld %lld\n", idx, (long long)lo, (long long)hi, differs, cand);
    std::fflush(stdout);
  }
  return 0;
}
