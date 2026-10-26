// Amendment 6.3 addendum: reachability of the reciprocal candidates found by recipall63.
//
// A candidate (S, d) is reachable iff some int8 softmax row has the row maximum, at least one element
// at difference d, and admitted-weight sum exactly S. Writing the row as multiplicities m_e over admitted
// differences e (each contributes B[e], softmax.h:103-110): S = B[0]*k + B[d] + sum_e m_e B[e], k >= 1.
// This program finds such multiplicities with an unbounded min-coin DP over the remainder (coins = the
// distinct non-zero admitted B values), first reducing a large remainder with copies of B[0].
//
// stdin: "C idx" + 256 "E ..." lines, then "Q n" and n lines "S d".
// stdout: per candidate "R S d k len ok" followed by "M e count" lines (the row), or "R S d - - 0".
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
  char tag[8]; int idx;
  while (std::scanf("%7s %d", tag, &idx) == 2) {
    int adm[256]; long long B[256];
    for (int i = 0; i < 256; ++i) {
      char e[4]; int d, ad, resc; long long ex_raw, acc;
      if (std::scanf("%3s %d %d %d %lld %lld", e, &d, &ad, &resc, &ex_raw, &acc) != 6) return 2;
      adm[d + 255] = ad; B[d + 255] = ad ? acc : 0;
    }
    int nq; if (std::scanf("%7s %d", tag, &nq) != 2) return 3;
    const long long LIM = 1ll << 21;
    // coin = admitted difference with a non-zero weight; keep the most negative d for each B value
    std::vector<int> coin_d; std::vector<long long> coin_v;
    for (int i = 0; i < 256; ++i) if (adm[i] && B[i] > 0) {
      bool dup = false; for (long long v : coin_v) dup |= v == B[i];
      if (!dup) { coin_d.push_back(i - 255); coin_v.push_back(B[i]); }
    }
    std::vector<int> dp(LIM + 1, 1 << 30), from(LIM + 1, -1);
    dp[0] = 0;
    for (long long r = 1; r <= LIM; ++r)
      for (size_t c = 0; c < coin_v.size(); ++c)
        if (coin_v[c] <= r && dp[r - coin_v[c]] + 1 < dp[r]) { dp[r] = dp[r - coin_v[c]] + 1; from[r] = (int)c; }
    for (int q = 0; q < nq; ++q) {
      long long S; int d; if (std::scanf("%lld %d", &S, &d) != 2) return 4;
      long long R = S - B[255] - (d != 0 ? B[d + 255] : 0);
      long long k = 1;
      if (R > LIM) { long long extra = (R - LIM) / B[255] + 1; k += extra; R -= extra * B[255]; }
      if (R < 0 || dp[R] >= (1 << 30)) { std::printf("R %lld %d - - 0\n", S, d); continue; }
      std::vector<long long> cnt(256, 0); cnt[255] += k; if (d != 0) cnt[d + 255] += 1;
      for (long long r = R; r > 0; r -= coin_v[from[r]]) cnt[coin_d[from[r]] + 255] += 1;
      long long len = 0, chk = 0; for (int i = 0; i < 256; ++i) { len += cnt[i]; chk += cnt[i] * B[i]; }
      std::printf("R %lld %d %lld %lld %d\n", S, d, k, len, (int)(chk == S));
      for (int i = 0; i < 256; ++i) if (cnt[i]) std::printf("M %d %lld\n", i - 255, cnt[i]);
    }
    std::fflush(stdout);
  }
  return 0;
}
