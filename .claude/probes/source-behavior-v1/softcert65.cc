// D-TFLP Amendment 6.5 P1: prepared-softmax reciprocal admission gate (lab).
//
// Two stages, kept apart because they depend on different inputs.
//
//   global   Source-global certificate for the pinned TFLM int8 SOFTMAX reciprocal. It depends ONLY on
//            the source reciprocal algorithm (GetReciprocal, common.h:968-982, with
//            one_over_one_plus_x_for_x_in_0_1), x_integer_digits = 12, int32 arithmetic and the supported
//            sum domain [2^19, 2^28) (6.4 follow-up N11). It does not depend on any calibration: the
//            reciprocal is a function of the row sum S alone. Contents: the domain, the correctly rounded
//            reference round(2^62 / (S << clz(S))), its monotonicity inside each exponent band (checked on
//            every S), the exponent of every S (= 35 - clz(S), checked), and E_h = max |source - exact| per
//            band h = clz(S) in 4..12 (measured on every S). The source revision and a digest of the source
//            text are supplied by the driver and copied into the certificate; the gate refuses a mismatch.
//
//   gate     Per-calibration decision from a certificate and one calibration's prepared source weights
//            (trace63 `softexp`: A[d] = exp_on_negative_values, B[d] = Rescale<12>, admitted flag).
//            1. Boundary-pruned search (as recipprune64, window E_h): every (S, w) in the supported domain
//               whose output byte differs between the source reciprocal and the exact one.
//            2. Exact reachability (as reach64): a candidate is a reachable pair iff an admitted row with the
//               mandatory maximum, an element at its difference and sum exactly S exists. Witness rows are
//               built and re-checked (positive counts, admitted codes, recomputed sum = S).
//            Decision:
//              PROVED_OUTPUT_EQUIVALENT    no reachable differing pair: the exact reciprocal gives the source
//                                          bytes on every supported row of this calibration;
//              REQUIRES_SOURCE_RECIPROCAL  at least one reachable witness row;
//              REFUSE_UNKNOWN              a certificate, identity, domain or table assumption fails.
//
// Mutations (negative controls; argv after the gate arguments): e_minus1, e_minus2, one_sided_window,
// exclude_lower_endpoint, wrong_band, wrong_weight_threshold, no_mandatory_max, no_reachability,
// allow_negative_multiplicity, witness_sum_off_by_max. A mutated run prints "MUTATION name".
//
// usage:  softcert65 global REVISION DIGEST > cert.txt
//         softcert65 gate CERT EXPECTED_REVISION EXPECTED_DIGEST [MUTATION] < calibration
// calibration stdin: "K scale_bits zp beta_bits", then trace63 `softexp` output ("P ..." + 256 "E d adm A B").
// gate stdout: "T stage seconds", one "C S d w sh_src sh_exact byte_src byte_exact status len" per candidate
// (followed by "M e count" lines for a reachable one), and "D decision reason candidates reachable unreachable".
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/fixedpoint.h"

using namespace tflite::micro;

static const int64_t DOMAIN_LO = int64_t{1} << 19, DOMAIN_HI = (int64_t{1} << 28) - 1;
static const int H_LO = 4, H_HI = 12;

static inline int out_byte(int32_t sh, int32_t a, int ex) {  // softmax.h:131-142, verbatim recipall63
  int32_t v = RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(sh, a), ex) - 128;
  return v < -128 ? -128 : (v > 127 ? 127 : v);
}
static inline int32_t exact_recip(int64_t S) {
  const int hpo = CountLeadingZeros(static_cast<uint32_t>(S));
  const __int128 den = (__int128)S << hpo;
  const __int128 q = ((((__int128)1) << 62) * 2 + den) / (2 * den);
  return q > INT32_MAX ? INT32_MAX : (int32_t)q;
}
static inline int32_t src_recip(int64_t S, int* ex) {
  int nb; const int32_t sh = GetReciprocal((int32_t)S, 12, &nb); *ex = nb + 23; return sh;
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
[[noreturn]] static void refuse(const char* why) { std::printf("\nD REFUSE_UNKNOWN %s 0 0 0\n", why); std::exit(0); }  // own line, even mid-record

static int global_mode(const char* rev, const char* digest) {
  const double t0 = now();
  long long E[33] = {0};
  for (int h = H_LO; h <= H_HI; ++h) {
    const int64_t a = std::max(DOMAIN_LO, int64_t{1} << (31 - h)), b = std::min(DOMAIN_HI, (int64_t{1} << (32 - h)) - 1);
    int32_t prev = INT32_MAX;
    for (int64_t S = a; S <= b; ++S) {
      int ex; const int32_t s = src_recip(S, &ex); const int32_t x = exact_recip(S);
      if (CountLeadingZeros(static_cast<uint32_t>(S)) != h || ex != 35 - h) { std::fprintf(stderr, "band/exponent at S=%lld\n", (long long)S); return 10; }
      if (x > prev) { std::fprintf(stderr, "exact reciprocal not monotone at S=%lld\n", (long long)S); return 11; }
      if (s <= 0 || x <= 0) { std::fprintf(stderr, "non-positive reciprocal at S=%lld\n", (long long)S); return 12; }
      prev = x;
      const long long dd = std::llabs((long long)s - x); if (dd > E[h]) E[h] = dd;
    }
  }
  std::printf("SOFTCERT 1\nsource_revision %s\nsource_digest %s\nreciprocal GetReciprocal(x,12)\n", rev, digest);
  std::printf("domain %lld %lld\nmonotone_exact 1\n", (long long)DOMAIN_LO, (long long)DOMAIN_HI);
  for (int h = H_LO; h <= H_HI; ++h) std::printf("E %d %lld\n", h, E[h]);
  std::printf("scan_seconds %.3f\n", now() - t0);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 4 && !std::strcmp(argv[1], "global")) return global_mode(argv[2], argv[3]);
  if (argc < 5 || std::strcmp(argv[1], "gate")) { std::fprintf(stderr, "usage\n"); return 2; }
  const std::string mut = argc > 5 ? argv[5] : "";
  if (!mut.empty()) std::printf("MUTATION %s\n", mut.c_str());
  double t = now();
  // ---- certificate ------------------------------------------------------------------------------
  FILE* cf = std::fopen(argv[2], "r");
  if (!cf) refuse("certificate_missing");
  char key[64], v1[256]; long long E[33]; for (auto& e : E) e = -1;
  std::string rev, dig; long long dlo = -1, dhi = -1; int mono = 0, magic = 0;
  while (std::fscanf(cf, "%63s", key) == 1) {
    if (!std::strcmp(key, "SOFTCERT")) { if (std::fscanf(cf, "%d", &magic) != 1) refuse("certificate_malformed"); }
    else if (!std::strcmp(key, "source_revision")) { if (std::fscanf(cf, "%255s", v1) != 1) refuse("certificate_malformed"); rev = v1; }
    else if (!std::strcmp(key, "source_digest")) { if (std::fscanf(cf, "%255s", v1) != 1) refuse("certificate_malformed"); dig = v1; }
    else if (!std::strcmp(key, "domain")) { if (std::fscanf(cf, "%lld %lld", &dlo, &dhi) != 2) refuse("certificate_malformed"); }
    else if (!std::strcmp(key, "monotone_exact")) { if (std::fscanf(cf, "%d", &mono) != 1) refuse("certificate_malformed"); }
    else if (!std::strcmp(key, "E")) { int h; long long e; if (std::fscanf(cf, "%d %lld", &h, &e) != 2 || h < 0 || h > 32) refuse("certificate_malformed"); E[h] = e; }
    else { if (std::fscanf(cf, "%*[^\n]") < 0) break; }
  }
  std::fclose(cf);
  if (magic != 1) refuse("certificate_version");
  if (rev != argv[3]) refuse("source_revision_mismatch");
  if (dig != argv[4]) refuse("source_digest_mismatch");
  if (dlo != DOMAIN_LO || dhi != DOMAIN_HI) refuse("certificate_domain_mismatch");
  if (mono != 1) refuse("monotonicity_not_certified");
  for (int h = H_LO; h <= H_HI; ++h) if (E[h] < 0 || E[h] > 1000) refuse("error_bound_missing_or_implausible");
  if (mut == "e_minus1") for (int h = H_LO; h <= H_HI; ++h) E[h] -= 1;
  if (mut == "e_minus2") for (int h = H_LO; h <= H_HI; ++h) E[h] -= 2;
  std::printf("T certificate_load_and_check %.6f\n", now() - t); t = now();
  // ---- calibration table ------------------------------------------------------------------------
  char tag[8]; unsigned sbits, bbits; int zp;
  if (std::scanf("%7s %x %d %x", tag, &sbits, &zp, &bbits) != 4 || std::strcmp(tag, "K")) refuse("calibration_identity_missing");
  long long pm, psh, pdmin;
  if (std::scanf("%7s %lld %lld %lld", tag, &pm, &psh, &pdmin) != 4 || std::strcmp(tag, "P")) refuse("calibration_table_malformed");
  int adm[256]; int64_t A[256], B[256];
  for (int i = 0; i < 256; ++i) {
    int d, ad; long long resc, a, b;  // trace63: "E d admitted input_diff_rescaled A B"
    if (std::scanf("%7s %d %d %lld %lld %lld", tag, &d, &ad, &resc, &a, &b) != 6 || std::strcmp(tag, "E") || d != i - 255) refuse("calibration_table_malformed");
    adm[i] = ad; A[i] = a; B[i] = b;
  }
  // assumptions the certificate relies on
  if (!adm[255]) refuse("row_maximum_not_admitted");
  if (B[255] != DOMAIN_LO) refuse("maximum_weight_not_domain_lower_bound");  // S >= B[0] = 2^19
  for (int i = 0; i < 256; ++i) if (adm[i] && (A[i] < 0 || A[i] > INT32_MAX || B[i] < 0 || B[i] > DOMAIN_LO)) refuse("prepared_weight_out_of_range");
  std::printf("T table_parse_and_check %.6f\n", now() - t); t = now();
  // ---- 1. boundary-pruned search ----------------------------------------------------------------
  std::vector<int64_t> W; std::map<int64_t, int> dof;
  for (int i = 0; i < 256; ++i) if (adm[i] && !dof.count(A[i])) { W.push_back(A[i]); dof[A[i]] = i - 255; }
  std::sort(W.begin(), W.end());
  struct Cand { int64_t S; int d; int64_t w; int32_t ss, sx; int bs, bx; };
  std::vector<Cand> cand; std::set<std::pair<int64_t, int64_t>> seen;
  long long evaluated = 0, intervals = 0;
  for (int64_t w : W) {
    if (w <= 0) continue;
    const int64_t wt = mut == "wrong_weight_threshold" ? w + 1 : w;
    for (int h = H_LO; h <= H_HI; ++h) {
      const int ex = mut == "wrong_band" ? 34 - h : 35 - h;
      const int64_t a = std::max(DOMAIN_LO, int64_t{1} << (31 - h)), b = std::min(DOMAIN_HI, (int64_t{1} << (32 - h)) - 1);
      const long long win = E[h];
      for (int m = 0; m <= 254; ++m) {
        const __int128 beta = ex >= 1 ? ((__int128)m << ex) + ((__int128)1 << (ex - 1)) : (__int128)(m + 1);
        const __int128 num = beta * ((__int128)1 << 31) - ((__int128)1 << 30);
        const __int128 T = num <= 0 ? 0 : (num + wt - 1) / wt;  // ceil: the smallest sh with unsat >= m + 1
        __int128 tlo = T - win, thi = T + win;
        if (mut == "one_sided_window") tlo = T;
        if (mut == "exclude_lower_endpoint") tlo = T - win + 1;
        if (tlo > INT32_MAX) continue;  // no reciprocal reaches this threshold
        int64_t L = a, R = b + 1;
        while (L < R) { const int64_t mid = L + (R - L) / 2; if ((__int128)exact_recip(mid) <= thi) R = mid; else L = mid + 1; }
        const int64_t s0 = L;
        L = a; R = b + 1;
        while (L < R) { const int64_t mid = L + (R - L) / 2; if ((__int128)exact_recip(mid) < tlo) R = mid; else L = mid + 1; }
        const int64_t s1 = L - 1;
        if (s0 > s1) continue;
        ++intervals;
        for (int64_t S = s0; S <= s1; ++S) {
          if (!seen.insert({S, w}).second) continue;
          ++evaluated;
          int exs; const int32_t ss = src_recip(S, &exs);
          if (exs != 35 - h) refuse("internal_band_exponent");
          const int32_t sx = exact_recip(S);
          if (ss == sx) continue;
          const int bs = out_byte(ss, (int32_t)w, 35 - h), bx = out_byte(sx, (int32_t)w, 35 - h);
          if (bs != bx) cand.push_back({S, dof[w], w, ss, sx, bs, bx});
        }
      }
    }
  }
  std::printf("T pruned_search %.6f\n", now() - t); t = now();
  std::printf("I weights %zu intervals %lld evaluated %lld candidates %zu\n", W.size(), intervals, evaluated, cand.size());
  // ---- 2. exact reachability (residue rule mod B[0]) ----------------------------------------------
  const int64_t M = B[255];
  std::vector<int> coin_d; std::vector<int64_t> coin_v;
  for (int i = 0; i < 256; ++i) if (adm[i] && B[i] > 0 && std::find(coin_v.begin(), coin_v.end(), B[i]) == coin_v.end()) { coin_d.push_back(i - 255); coin_v.push_back(B[i]); }
  std::vector<int64_t> dist; std::vector<int> pc; std::vector<int64_t> pr;
  if (!cand.empty() && mut != "no_reachability") {
    const int64_t INF = INT64_MAX / 4;
    dist.assign(M, INF); pc.assign(M, -1); pr.assign(M, -1);
    using P = std::pair<int64_t, int64_t>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    dist[0] = 0; pq.push({0, 0});
    while (!pq.empty()) {
      auto [dv, r] = pq.top(); pq.pop();
      if (dv != dist[r]) continue;
      for (size_t c = 0; c < coin_v.size(); ++c) {
        int64_t nv; if (__builtin_add_overflow(dv, coin_v[c], &nv)) refuse("internal_overflow");
        const int64_t nr = nv % M;
        if (nv < dist[nr]) { dist[nr] = nv; pc[nr] = (int)c; pr[nr] = r; pq.push({nv, nr}); }
      }
    }
  }
  std::printf("T reachability_graph %.6f\n", now() - t); t = now();
  long long nreach = 0, nunreach = 0;
  for (const Cand& c : cand) {
    std::printf("C %lld %d %lld %d %d %d %d ", (long long)c.S, c.d, (long long)c.w, c.ss, c.sx, c.bs, c.bx);
    if (mut == "no_reachability") { std::printf("reachable 0\n"); ++nreach; continue; }
    const int64_t R = c.S - (mut == "no_mandatory_max" ? 0 : M) - (c.d != 0 ? B[c.d + 255] : 0);
    if (R < 0) { std::printf("unreachable_negative 0\n"); ++nunreach; continue; }
    const int64_t r = R % M;
    if (dist[r] > R && mut != "allow_negative_multiplicity") { std::printf("unreachable_gap 0\n"); ++nunreach; continue; }
    std::vector<int64_t> cnt(256, 0);
    cnt[255] += (mut == "no_mandatory_max" ? 0 : 1) + (R - dist[r]) / M;
    if (c.d != 0) cnt[c.d + 255] += 1;
    for (int64_t x = r; x != 0; x = pr[x]) cnt[coin_d[pc[x]] + 255] += 1;
    if (mut == "witness_sum_off_by_max") cnt[255] += 1;
    // the row must exist as written
    __int128 chk = 0; int64_t len = 0; bool neg = false;
    for (int i = 0; i < 256; ++i) { if (cnt[i] < 0) neg = true; if (cnt[i] && !adm[i]) refuse("internal_unadmitted_code"); chk += (__int128)cnt[i] * B[i]; len += cnt[i]; }
    if (neg) refuse("witness_negative_multiplicity");
    if (chk != c.S) refuse("witness_sum_mismatch");
    std::printf("reachable %lld\n", (long long)len); ++nreach;
    for (int i = 0; i < 256; ++i) if (cnt[i]) std::printf("M %d %lld\n", i - 255, (long long)cnt[i]);
  }
  std::printf("T witness_construction %.6f\n", now() - t);
  if (nreach > 0) std::printf("D REQUIRES_SOURCE_RECIPROCAL reachable_witness %zu %lld %lld\n", cand.size(), nreach, nunreach);
  else std::printf("D PROVED_OUTPUT_EQUIVALENT no_reachable_difference %zu %lld %lld\n", cand.size(), nreach, nunreach);
  return 0;
}
