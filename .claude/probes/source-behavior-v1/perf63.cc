// Amendment 6.3 addendum: execution cost of the pinned TFLM reference kernels vs prepared native
// forms, with output parity checked on every timed buffer.
//
//   unary  (TANH, LOGISTIC): TFLM kernel Invoke over N int8 codes  vs  a 256-byte table built once from
//          the same kernel (preparation timed separately) and applied by scalar lookup.
//   softmax (int8, 1/256, -128): TFLM kernel Invoke over R rows x n  vs  the source-compatible recipe:
//          prepared per-calibration weights (TFLM exp_on_negative_values, Rescale<12>), then per row an
//          integer sum, TFLM GetReciprocal, SRDHM + RoundingDivideByPOT, -128, clamp.
// Calibration: input scale 0.05, zp 0 (unary) / scale 0.05, zp 0, beta 1.0 (softmax). Output: CSV lines.
// Timing: steady_clock, median of 7 runs after 1 warm-up. Single thread.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
static volatile int sink;

static void unary(const char* name, TFLMRegistration reg, float out_scale, int out_zp, size_t N) {
  std::vector<int8_t> in(N), out(N), lut_out(N);
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (auto& v : in) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v = (int8_t)(s >> 56); }
  int dims_data[] = {2, 1, (int)N};
  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(dims_data);
  TfLiteTensor t[2] = {tflite::testing::CreateQuantizedTensor(in.data(), dims, 0.05f, 0),
                       tflite::testing::CreateQuantizedTensor(out.data(), dims, out_scale, out_zp)};
  int ia[] = {1, 0}, oa[] = {1, 1};
  KernelRunner runner(reg, t, 2, tflite::testing::IntArrayFromInts(ia), tflite::testing::IntArrayFromInts(oa), nullptr);
  if (runner.InitAndPrepare() != kTfLiteOk) { std::printf("ERR prepare\n"); std::exit(2); }
  const double t_kernel = median_ns([&] { runner.Invoke(); });
  // preparation: the table is the kernel's own output on the 256 codes
  int8_t codes[256], table[256];
  for (int i = 0; i < 256; ++i) codes[i] = (int8_t)(i - 128);
  const double t_prep = median_ns([&] {
    int d2[] = {2, 1, 256};
    TfLiteIntArray* dd = tflite::testing::IntArrayFromInts(d2);
    TfLiteTensor u[2] = {tflite::testing::CreateQuantizedTensor(codes, dd, 0.05f, 0),
                         tflite::testing::CreateQuantizedTensor(table, dd, out_scale, out_zp)};
    KernelRunner r2(reg, u, 2, tflite::testing::IntArrayFromInts(ia), tflite::testing::IntArrayFromInts(oa), nullptr);
    r2.InitAndPrepare(); r2.Invoke();
  });
  const double t_lut = median_ns([&] {
    const int8_t* p = in.data(); int8_t* q = lut_out.data();
    for (size_t i = 0; i < N; ++i) q[i] = table[(uint8_t)(p[i] ^ 0x80)];
    sink = q[N / 2];
  });
  size_t mism = 0; for (size_t i = 0; i < N; ++i) mism += out[i] != lut_out[i];
  std::printf("%s,N=%zu,kernel_ns_per_elem=%.3f,lut_ns_per_elem=%.3f,table_prep_ns=%.0f,mismatches=%zu\n",
              name, N, t_kernel / N, t_lut / N, t_prep, mism);
}

static void softmax(int n, int R) {
  const float scale = 0.05f, beta = 1.0f; const int zp = 0;
  std::vector<int8_t> in((size_t)n * R), out(in.size()), rec(in.size());
  uint64_t s = 0x2545F4914F6CDD1Dull;
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
  // preparation: per-calibration weight tables from TFLM's own primitives (softmax.h:103-110)
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
  bool refused = false;
  const double t_rec = median_ns([&] {
    for (int r = 0; r < R; ++r) {
      const int8_t* x = &in[(size_t)r * n]; int8_t* y = &rec[(size_t)r * n];
      int mx = -128; for (int c = 0; c < n; ++c) mx = std::max(mx, (int)x[c]);
      int64_t S = 0; for (int c = 0; c < n; ++c) S += B[x[c] - mx + 255];
      int nb; if (S > INT32_MAX) { refused = true; continue; }
      const int32_t sh = GetReciprocal((int32_t)S, 12, &nb); const int ex = nb + 23;
      if (ex < 0 || ex > 31) { refused = true; continue; }
      for (int c = 0; c < n; ++c) {
        const int k = x[c] - mx + 255;
        int32_t v = adm[k] ? RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(sh, A[k]), ex) - 128 : -128;
        y[c] = (int8_t)std::min(127, std::max(-128, (int)v));
      }
    }
  });
  size_t mism = 0; for (size_t i = 0; i < in.size(); ++i) mism += out[i] != rec[i];
  std::printf("softmax,n=%d,rows=%d,kernel_ns_per_elem=%.3f,recipe_ns_per_elem=%.3f,weight_prep_ns=%.0f,refused=%d,mismatches=%zu\n",
              n, R, t_kernel / in.size(), t_rec / in.size(), t_prep, (int)refused, mism);
}

int main() {
  unary("tanh", tflite::Register_TANH(), 1.0f / 128.0f, 0, (size_t)1 << 24);
  unary("logistic", tflite::Register_LOGISTIC(), 1.0f / 256.0f, -128, (size_t)1 << 24);
  softmax(64, 1 << 16);
  softmax(256, 1 << 14);
  softmax(511, 1 << 13);
  return 0;
}
