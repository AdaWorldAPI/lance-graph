// D-TFLP Amendment 6.4 follow-up: the reciprocal superset search, pruned around the final quantization
// boundaries, as an independent re-derivation of recipall63's exhaustive result.
//
// Output computation (reference/softmax.h:131-143, identical to recipall63's out_byte): for row sum S in
// [B[0], 2^28) and admitted weight w > 0,
//     h = clz(S) in [4, 12], ex = 35 - h in [23, 31] for S in [2^19, 2^28),   sh = reciprocal(S),
//     x = SaturatingRoundingDoublingHighMul(sh, w) = floor((sh*w + 2^30) / 2^31)       (sh, w >= 0),
//     unsat = RoundingDivideByPOT(x, ex) = floor(x / 2^ex) + [x mod 2^ex >= 2^(ex-1)],
//     byte = min(unsat - 128, 127).
// For fixed (w, ex) the byte is a non-decreasing step function of sh: unsat >= m + 1 iff
//     x >= beta_m = m*2^ex + 2^(ex-1)  iff  sh >= T_m = ceil((beta_m*2^31 - 2^30) / w),
// and only the steps m = 0..254 are visible (unsat >= 255 clamps to 127).
// So the source and exact arms give different bytes iff some T_m lies in (min(sh_src, sh_ex), max(...)].
//
// Pruning. Let E_h = max over the band h of |sh_src(S) - sh_ex(S)| (measured below by a scan of the
// reciprocal alone, which does not depend on w). Then a differing byte requires
//     sh_ex(S) in [T_m - E_h, T_m + E_h]   for some m.
// sh_ex(S) = round(2^62 / (S << h)) is non-increasing in S inside a band (asserted during the scan), so
// that condition selects one interval of S per (w, h, m), found by binary search. Only S in those
// intervals are evaluated, with the same exact byte computation as recipall63. The method is sound iff
// E_h bounds the reciprocal difference and sh_ex is monotone; both are checked, and the result is
// compared set-for-set with the exhaustive run by recipprune64.py.
//
// Arguments: [DELTA [SLACK_OVERRIDE]]. DELTA: the negative control of recipall63 (second arm = source +
// DELTA instead of the exact reciprocal); the window then widens to E_h + |DELTA| (both arms lie within
// that distance of sh_ex). SLACK_OVERRIDE >= 0 replaces the window (disable runs).
// stdin: as recipall63. stdout: "X idx S d src alt" per differing pair, then
// "P idx intervals s_evaluated distinct_pairs_evaluated" per calibration; first, "E h E_h" per band.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <utility>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/fixedpoint.h"

using namespace tflite::micro;

static inline int out_byte(int32_t sh, int32_t a, int ex) {  // verbatim recipall63
  int32_t v = RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(sh, a), ex) - 128;
  return v < -128 ? -128 : (v > 127 ? 127 : v);
}
static inline int32_t exact_recip(int64_t S) {  // verbatim recipall63
  const int hpo = CountLeadingZeros(static_cast<uint32_t>(S));
  const __int128 den = (__int128)S << hpo;
  const __int128 q = ((((__int128)1) << 62) * 2 + den) / (2 * den);
  return q > INT32_MAX ? INT32_MAX : (int32_t)q;
}
static inline int32_t src_recip(int64_t S, int* ex) {
  int nb; const int32_t sh = GetReciprocal((int32_t)S, 12, &nb); *ex = nb + 23; return sh;
}

int main(int argc, char** argv) {
  const long long delta = argc > 1 ? std::atoll(argv[1]) : 0;
  const long long slack_override = argc > 2 ? std::atoll(argv[2]) : -1;
  const int64_t LO_ALL = int64_t{1} << 19, HI = (int64_t{1} << 28) - 1;
  // band h holds S in [2^(31-h), 2^(32-h)); h = 4 .. 12 covers [2^19, 2^28)
  long long E[33] = {0};
  for (int h = 4; h <= 12; ++h) {
    const int64_t a = std::max(LO_ALL, int64_t{1} << (31 - h)), b = std::min(HI, (int64_t{1} << (32 - h)) - 1);
    int32_t prev = INT32_MAX;
    for (int64_t S = a; S <= b; ++S) {
      int ex; const int32_t s = src_recip(S, &ex); const int32_t x = exact_recip(S);
      if (ex != 35 - h) return 10;
      if (x > prev) return 11;  // monotonicity of the exact reciprocal inside the band
      prev = x;
      const long long dd = std::llabs((long long)s - x); if (dd > E[h]) E[h] = dd;
    }
    std::printf("E %d %lld\n", h, E[h]);
  }
  char tag[8]; int idx;
  while (std::scanf("%7s %d", tag, &idx) == 2) {
    int adm[256], dval[256]; int32_t A[256], B[256]; int na = 0; int32_t W[256];
    for (int i = 0; i < 256; ++i) {
      char e[4]; int d, ad, resc; long long ex_raw, acc;
      if (std::scanf("%3s %d %d %d %lld %lld", e, &d, &ad, &resc, &ex_raw, &acc) != 6) return 2;
      adm[d + 255] = ad; A[d + 255] = (int32_t)ex_raw; B[d + 255] = (int32_t)acc; dval[d + 255] = d;
      if (ad) { W[na] = (int32_t)ex_raw; ++na; }
    }
    std::sort(W, W + na); na = (int)(std::unique(W, W + na) - W);
    const int64_t lo = B[255];
    if (lo < LO_ALL) return 12;
    std::set<std::pair<int64_t, int>> seen;  // (S, weight index)
    long long intervals = 0, s_eval = 0, pair_eval = 0;
    for (int j = 0; j < na; ++j) {
      const int64_t w = W[j];
      if (w <= 0) continue;  // weight 0: x = 0 for both arms, no byte can change
      int d = 0; for (int i = 0; i < 256; ++i) if (adm[i] && A[i] == W[j]) { d = dval[i]; break; }
      for (int h = 4; h <= 12; ++h) {
        const int ex = 35 - h;
        const int64_t a = std::max(lo, int64_t{1} << (31 - h)), b = std::min(HI, (int64_t{1} << (32 - h)) - 1);
        if (a > b) continue;
        const long long win = slack_override >= 0 ? slack_override : E[h] + std::llabs(delta);
        for (int m = 0; m <= 254; ++m) {
          const __int128 beta = ex >= 1 ? ((__int128)m << ex) + ((__int128)1 << (ex - 1)) : (__int128)(m + 1);
          const __int128 num = beta * ((__int128)1 << 31) - ((__int128)1 << 30);
          const __int128 T = num <= 0 ? 0 : (num + w - 1) / w;
          const __int128 tlo = T - win, thi = T + win;
          // first S in [a, b] with exact_recip(S) <= thi; last S with exact_recip(S) >= tlo
          int64_t L = a, Rr = b + 1;
          while (L < Rr) { const int64_t mid = L + (Rr - L) / 2; if ((__int128)exact_recip(mid) <= thi) Rr = mid; else L = mid + 1; }
          const int64_t s0 = L;
          L = a; Rr = b + 1;
          while (L < Rr) { const int64_t mid = L + (Rr - L) / 2; if ((__int128)exact_recip(mid) < tlo) Rr = mid; else L = mid + 1; }
          const int64_t s1 = L - 1;
          if (s0 > s1) continue;
          ++intervals;
          for (int64_t S = s0; S <= s1; ++S) {
            ++s_eval;
            if (!seen.insert({S, j}).second) continue;
            ++pair_eval;
            int exs; const int32_t sh = src_recip(S, &exs);
            if (exs != ex) return 13;
            int32_t alt = exact_recip(S);
            if (delta) { const long long p = (long long)sh + delta; alt = p > INT32_MAX ? INT32_MAX : (int32_t)p; }
            if (alt == sh) continue;
            const int bs = out_byte(sh, (int32_t)w, ex), bx = out_byte(alt, (int32_t)w, ex);
            if (bs != bx) std::printf("X %d %lld %d %d %d\n", idx, (long long)S, d, bs, bx);
          }
        }
      }
    }
    std::printf("P %d %lld %lld %lld\n", idx, intervals, s_eval, pair_eval);
    std::fflush(stdout);
  }
  return 0;
}
