// D-TFLP Amendment 6.5 P3: execution cost of the two softmax recipe alternatives against the pinned TFLM
// kernel, on identical admitted workloads, with output parity reported beside every timing.
//
// Arms (int8 SOFTMAX, output 1/256 zp -128), per (calibration, n):
//   kernel        TFLM reference kernel Invoke over R rows x n;
//   source        prepared recipe: weights from TFLM primitives, per row integer sum, TFLM GetReciprocal,
//                 SRDHM + RoundingDivideByPOT, -128, clamp (perf63's recipe);
//   exact         the same recipe with the correctly rounded reciprocal round(2^62 / (S << clz(S))),
//                 computed with one u64 division (S << clz(S) is in [2^31, 2^32), so 2^63 + den and
//                 2 * den fit in 64 bits). SourceParity is claimed for this arm ONLY on a calibration the
//                 6.5 gate certified PROVED_OUTPUT_EQUIVALENT (calibration 9); elsewhere its mismatches
//                 are reported and it is not admitted;
//   recip_only    the per-row reciprocal alone (source vs exact) over the same rows' sums.
// Workload: R rows of n codes from a fixed xorshift stream; every n <= 511 keeps S < 2^28 (admitted).
// Timing: steady_clock, median of 7 runs after 1 warm-up, single thread. Output: CSV lines.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/fixedpoint.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"

using namespace tflite::micro;
using clk = std::chrono::steady_clock;

template <class F> static double median_ns(F f) {
  f();
  std::vector<double> t;
  for (int r = 0; r < 7; ++r) { auto a = clk::now(); f(); t.push_back(std::chrono::duration<double, std::nano>(clk::now() - a).count()); }
  std::sort(t.begin(), t.end()); return t[3];
}
static volatile int64_t sink;

static inline int32_t exact_recip_u64(int32_t S, int* nb) {
  const int hpo = CountLeadingZeros(static_cast<uint32_t>(S));
  *nb = 12 - hpo;
  const uint64_t den = (uint64_t)(uint32_t)S << hpo;
  const uint64_t q = ((uint64_t{1} << 63) + den) / (2 * den);
  return q > (uint64_t)INT32_MAX ? INT32_MAX : (int32_t)q;
}

static void run(int cal, float scale, int zp, float beta, int n, int R) {
  std::vector<int8_t> in((size_t)n * R), out(in.size()), rs(in.size()), rx(in.size());
  uint64_t s = 0x2545F4914F6CDD1Dull ^ (uint64_t)n;
  for (auto& v : in) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v = (int8_t)(s >> 56); }
  int dims_data[] = {2, R, n};
  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(dims_data);
  TfLiteTensor t[2] = {tflite::testing::CreateQuantizedTensor(in.data(), dims, scale, zp),
                       tflite::testing::CreateQuantizedTensor(out.data(), dims, 1.0f / 256.0f, -128)};
  int ia[] = {1, 0}, oa[] = {1, 1};
  TfLiteSoftmaxParams params = {beta};
  KernelRunner runner(tflite::Register_SOFTMAX(), t, 2, tflite::testing::IntArrayFromInts(ia),
                      tflite::testing::IntArrayFromInts(oa), &params);
  if (runner.InitAndPrepare() != kTfLiteOk) { std::printf("ERR prepare\n"); std::exit(2); }
  const double t_kernel = median_ns([&] { runner.Invoke(); });
  int32_t A[256], B[256]; bool adm[256]; int32_t mult; int shift, diff_min;
  const double t_prep = median_ns([&] {
    PreprocessSoftmaxScaling((double)beta, (double)scale, 5, &mult, &shift);
    diff_min = -1.0 * CalculateInputRadius(5, shift);
    for (int d = -255; d <= 0; ++d) {
      adm[d + 255] = d >= diff_min;
      if (!adm[d + 255]) { A[d + 255] = B[d + 255] = 0; continue; }
      const int32_t r = MultiplyByQuantizedMultiplierGreaterThanOne(d, mult, shift);
      const FixedPoint<int32_t, 0> e = exp_on_negative_values(FixedPoint<int32_t, 5>::FromRaw(r));
      A[d + 255] = e.raw(); B[d + 255] = Rescale<12>(e).raw();
    }
  });
  std::vector<int32_t> sums(R);
  auto recipe = [&](bool exact, std::vector<int8_t>& y_all) {
    for (int r = 0; r < R; ++r) {
      const int8_t* x = &in[(size_t)r * n]; int8_t* y = &y_all[(size_t)r * n];
      int mx = -128; for (int c = 0; c < n; ++c) mx = std::max(mx, (int)x[c]);
      int64_t S = 0; for (int c = 0; c < n; ++c) S += B[x[c] - mx + 255];
      sums[r] = (int32_t)S;  // n <= 511: S < 2^28, admitted
      int nb; const int32_t sh = exact ? exact_recip_u64((int32_t)S, &nb) : GetReciprocal((int32_t)S, 12, &nb);
      const int ex = nb + 23;
      for (int c = 0; c < n; ++c) {
        const int k = x[c] - mx + 255;
        int32_t v = adm[k] ? RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(sh, A[k]), ex) - 128 : -128;
        y[c] = (int8_t)std::min(127, std::max(-128, (int)v));
      }
    }
  };
  const double t_src = median_ns([&] { recipe(false, rs); });
  const double t_ex = median_ns([&] { recipe(true, rx); });
  const double t_rs = median_ns([&] { int64_t a = 0; for (int r = 0; r < R; ++r) { int nb; a += GetReciprocal(sums[r], 12, &nb) + nb; } sink = a; });
  const double t_rx = median_ns([&] { int64_t a = 0; for (int r = 0; r < R; ++r) { int nb; a += exact_recip_u64(sums[r], &nb) + nb; } sink = a; });
  size_t ms = 0, mx = 0;
  for (size_t i = 0; i < in.size(); ++i) { ms += out[i] != rs[i]; mx += out[i] != rx[i]; }
  std::printf("cal=%d,n=%d,rows=%d,kernel_ns_per_row=%.1f,source_recipe_ns_per_row=%.1f,exact_recipe_ns_per_row=%.1f,"
              "source_recip_ns_per_row=%.2f,exact_recip_ns_per_row=%.2f,weight_prep_ns=%.0f,source_mismatches=%zu,exact_mismatches=%zu\n",
              cal, n, R, t_kernel / R, t_src / R, t_ex / R, t_rs / R, t_rx / R, t_prep, ms, mx);
}

int main() {
  for (int n : {8, 64, 256, 511}) {
    const int R = (1 << 22) / n;
    run(9, 0.5f, -25, 2.0f, n, R);
    run(10, 1.0f / 256.0f, 20, 1.0f, n, R);
  }
  return 0;
}
