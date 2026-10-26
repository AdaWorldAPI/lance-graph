// D-TFLP Amendment 6.4 follow-up: EXACT reachability of reciprocal candidates (S, d).
//
// reach63 used a bounded min-coin DP (remainders <= 2^21) and could only say "no row found" above
// that bound, which left 72 of 601 candidates UNKNOWN. This program decides every candidate exactly.
//
// A candidate (S, d) is reachable iff some admitted int8 softmax row has its maximum, at least one
// element at difference d, and an admitted-weight sum of exactly S (reference/softmax.h:99-111; an
// element at difference e contributes B[e] = Rescale<12>(exp_on_negative_values(...))). With R the
// remainder after the mandatory max element and the d element,
//     R = S - B[0] - (d != 0 ? B[d] : 0),
// the candidate is reachable iff R >= 0 and R is a non-negative integer combination of the coins
// C = { B[e] > 0 : e admitted } (B[0] is a coin, so the max may repeat). Rows have no length limit
// here; the admission bound S < 2^28 is already imposed by the superset.
//
// Exactness: B[0] is a coin, so R is representable iff R >= minrep[R mod B[0]], where minrep[r] is the
// smallest representable value congruent to r mod B[0] (Dijkstra on the residue graph mod B[0], one
// edge per coin). A witness row is the Dijkstra path to minrep[R mod B[0]] plus (R - minrep)/B[0]
// further maxima. A candidate with R < 0, or with R < minrep[R mod B[0]], is PROVED unreachable.
//
// With argument LIM, also prints "XC idx LIM representable mismatches": a plain unbounded-coin DP over
// [0, LIM] checked against the residue rule (independent cross-check; mismatches must be 0).
//
// stdin: "C idx" + 256 "E d adm resc ex_raw acc" lines, then "Q n" and n lines "S d".
// stdout per candidate: "R S d status len" with status reachable|unreachable_negative|unreachable_gap,
// followed, if reachable, by "M e count" lines (the row as multiplicities over differences).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <vector>

int main(int argc, char** argv) {
  const long long xcheck = argc > 1 ? std::atoll(argv[1]) : 0;
  char tag[8]; int idx;
  while (std::scanf("%7s %d", tag, &idx) == 2) {
    int adm[256]; long long B[256];
    for (int i = 0; i < 256; ++i) {
      char e[4]; int d, ad, resc; long long ex_raw, acc;
      if (std::scanf("%3s %d %d %d %lld %lld", e, &d, &ad, &resc, &ex_raw, &acc) != 6) return 2;
      adm[d + 255] = ad; B[d + 255] = ad ? acc : 0;
    }
    int nq; if (std::scanf("%7s %d", tag, &nq) != 2) return 3;
    const long long M = B[255];
    if (M <= 0) return 4;
    std::vector<int> coin_d; std::vector<long long> coin_v;
    for (int i = 0; i < 256; ++i) if (adm[i] && B[i] > 0) {
      bool dup = false; for (long long v : coin_v) dup |= v == B[i];
      if (!dup) { coin_d.push_back(i - 255); coin_v.push_back(B[i]); }
    }
    const long long INF = (long long)1 << 62;
    std::vector<long long> dist(M, INF); std::vector<int> pc(M, -1); std::vector<long long> pr(M, -1);
    using P = std::pair<long long, long long>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    dist[0] = 0; pq.push({0, 0});
    while (!pq.empty()) {
      auto [dv, r] = pq.top(); pq.pop();
      if (dv != dist[r]) continue;
      for (size_t c = 0; c < coin_v.size(); ++c) {
        const long long nv = dv + coin_v[c], nr = nv % M;
        if (nv < dist[nr]) { dist[nr] = nv; pc[nr] = (int)c; pr[nr] = r; pq.push({nv, nr}); }
      }
    }
    if (xcheck > 0) {  // independent check: plain unbounded-coin DP on [0, xcheck] vs the residue rule
      std::vector<char> rep(xcheck + 1, 0); rep[0] = 1;
      for (long long x = 1; x <= xcheck; ++x)
        for (long long v : coin_v) if (v <= x && rep[x - v]) { rep[x] = 1; break; }
      long long bad = 0, nrep = 0;
      for (long long x = 0; x <= xcheck; ++x) { nrep += rep[x]; bad += (rep[x] != 0) != (x >= dist[x % M]); }
      std::printf("XC %d %lld %lld %lld\n", idx, xcheck, nrep, bad);
    }
    for (int q = 0; q < nq; ++q) {
      long long S; int d; if (std::scanf("%lld %d", &S, &d) != 2) return 5;
      const long long R = S - M - (d != 0 ? B[d + 255] : 0);
      if (R < 0) { std::printf("R %lld %d unreachable_negative 0\n", S, d); continue; }
      const long long r = R % M;
      if (dist[r] > R) { std::printf("R %lld %d unreachable_gap 0\n", S, d); continue; }
      std::vector<long long> cnt(256, 0);
      cnt[255] += 1 + (R - dist[r]) / M; if (d != 0) cnt[d + 255] += 1;
      for (long long x = r; x != 0; x = pr[x]) cnt[coin_d[pc[x]] + 255] += 1;
      long long len = 0, chk = 0;
      for (int i = 0; i < 256; ++i) { if (cnt[i] < 0) return 7; len += cnt[i]; chk += cnt[i] * B[i]; }
      if (chk != S) return 6;  // the reconstruction must be a real row (no negative counts) hitting S exactly
      std::printf("R %lld %d reachable %lld\n", S, d, len);
      for (int i = 0; i < 256; ++i) if (cnt[i]) std::printf("M %d %lld\n", i - 255, cnt[i]);
    }
    std::fflush(stdout);
  }
  return 0;
}
