// Amendment 6.3 harness (TFLite Micro, reference kernels). Calls TFLM's own fixed-point
// primitives; never re-implements them.
//
// stdin, one request per line:
//   ltrace <scale_bits> <zp>          LOGISTIC int8 (out 1/256, zp -128): re-executes the loop body of
//                                     reference_integer_ops::Logistic (integer_ops/logistic.h:27-61)
//                                     stage by stage, ABORTS unless every byte equals the real kernel's.
//                                     -> 256 lines: G code input radius in_q4 abs_raw am_raw core_raw
//                                        remainder e_raw rpos_raw raw q23 byte kernel_byte
//   softexp <scale_bits> <beta_bits>  softmax int8 prepare (PreprocessSoftmaxScaling, CalculateInputRadius)
//                                     and, for d = -255..0, the per-difference source values:
//                                     -> "P mult shift diff_min" then 256 lines "E d admitted rescaled exp_raw acc_raw"
//   recip <S_raw>                     GetReciprocal(S, 12, &n) -> "R S shifted_scale n"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/fixedpoint.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"

namespace {
using namespace tflite::micro;

void LTrace(float scale, int zp) {
  using F4 = FixedPoint<int32_t, 4>;
  using F0 = FixedPoint<int32_t, 0>;
  int8_t in[256], kernel[256];
  for (int i = 0; i < 256; ++i) in[i] = static_cast<int8_t>(i - 128);
  int dims_data[] = {2, 1, 256};
  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(dims_data);
  TfLiteTensor tensors[2] = {tflite::testing::CreateQuantizedTensor(in, dims, scale, zp),
                             tflite::testing::CreateQuantizedTensor(kernel, dims, 1.0f / 256.0f, -128)};
  int ia[] = {1, 0}, oa[] = {1, 1};
  KernelRunner runner(tflite::Register_LOGISTIC(), tensors, 2, tflite::testing::IntArrayFromInts(ia),
                      tflite::testing::IntArrayFromInts(oa), nullptr);
  if (runner.InitAndPrepare() != kTfLiteOk || runner.Invoke() != kTfLiteOk) { std::printf("ERR kernel\n"); std::exit(2); }
  // logistic_common.cc int8 prepare, same TFLM functions
  constexpr int kInputIntegerBits = 4;
  const double real = static_cast<double>(scale) * static_cast<double>(1 << (31 - kInputIntegerBits));
  int shift = 0;
  const double q = std::frexp(real, &shift);
  const int32_t mult = static_cast<int32_t>(tflite::TfLiteRound(q * (1ll << 31)));
  const int32_t radius = tflite::CalculateInputRadius(kInputIntegerBits, shift, 31);
  for (int i = 0; i < 256; ++i) {
    const int32_t input = static_cast<int32_t>(in[i]) - zp;
    int32_t in_q4 = 0, abs_raw = 0, am_raw = 0, core_raw = 0, rem = 0, e_raw = 0, rpos_raw = 0, raw = 0, q23 = 0;
    int byte;
    if (input <= -radius) byte = -128;
    else if (input >= radius) byte = 127;
    else {
      in_q4 = MultiplyByQuantizedMultiplier(input, mult, shift);
      F4 a = F4::FromRaw(in_q4);
      // logistic(): fixedpoint.h:863-876
      const int32_t mask_if_positive = MaskIfGreaterThan(a, F4::Zero());
      const int32_t mask_if_zero = MaskIfZero(a);
      F4 abs_input = SelectUsingMask(mask_if_positive, a, -a);
      abs_raw = abs_input.raw();
      // logistic_on_positive_values(): one_over_one_plus_x(exp_on_negative_values(-abs)) (:856-858)
      F4 neg = -abs_input;
      // exp_on_negative_values() internals (:736-783), recorded for attribution only
      const F4 kOneQuarter = F4::template ConstantPOT<-2>();
      F4 mask = kOneQuarter - F4::FromScalarRaw(1);
      F4 am = (neg & mask) - kOneQuarter;
      am_raw = am.raw();
      core_raw = exp_on_interval_between_negative_one_quarter_and_0_excl(Rescale<0>(am)).raw();
      rem = (am - neg).raw();
      F0 e = exp_on_negative_values(neg);
      e_raw = e.raw();
      F0 rpos = one_over_one_plus_x_for_x_in_0_1(e);
      rpos_raw = rpos.raw();
      F0 rneg = F0::One() - rpos;
      const F0 one_half = F0::FromRaw(1 << 30);
      raw = SelectUsingMask(mask_if_zero, one_half, SelectUsingMask(mask_if_positive, rpos, rneg)).raw();
      q23 = RoundingDivideByPOT(raw, 31 - 8);  // integer_ops/logistic.h:53-54
      int32_t c = q23 - 128;
      byte = c < -128 ? -128 : (c > 127 ? 127 : c);
    }
    if (byte != kernel[i]) { std::printf("ERR ltrace %d != kernel %d at code %d\n", byte, kernel[i], i - 128); std::exit(3); }
    std::printf("G %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", i - 128, input, radius, in_q4, abs_raw, am_raw,
                core_raw, rem, e_raw, rpos_raw, raw, q23, byte, kernel[i]);
  }
}

void SoftExp(float scale, float beta) {
  using FD = FixedPoint<int32_t, 5>;  // kScaledDiffIntegerBits (softmax.h:81)
  using F0 = FixedPoint<int32_t, 0>;
  int32_t mult = 0;
  int shift = 0;
  tflite::micro::PreprocessSoftmaxScaling(static_cast<double>(beta), static_cast<double>(scale), 5, &mult, &shift);
  const int diff_min = -1.0 * tflite::micro::CalculateInputRadius(5, shift);  // softmax_common.cc:128-130
  std::printf("P %d %d %d\n", mult, shift, diff_min);
  for (int d = -255; d <= 0; ++d) {
    if (d >= diff_min) {
      const int32_t r = MultiplyByQuantizedMultiplierGreaterThanOne(d, mult, shift);  // softmax.h:104-106
      const F0 e = exp_on_negative_values(FD::FromRaw(r));                          // :107-110, :131
      const int32_t acc = Rescale<12>(e).raw();                                      // :109
      std::printf("E %d 1 %d %d %d\n", d, r, e.raw(), acc);
    } else {
      std::printf("E %d 0 0 0 0\n", d);
    }
  }
}
}  // namespace

int main() {
  char mode[16];
  while (std::scanf("%15s", mode) == 1) {
    if (std::strcmp(mode, "recip") == 0) {
      int32_t s; if (std::scanf("%d", &s) != 1) return 1;
      int n = 0; const int32_t r = tflite::micro::GetReciprocal(s, 12, &n);
      std::printf("R %d %d %d\n", s, r, n);
      continue;
    }
    unsigned a = 0, b = 0;
    if (std::scanf("%x %x", &a, &b) != 2) return 1;
    float scale; std::memcpy(&scale, &a, 4);
    if (std::strcmp(mode, "ltrace") == 0) LTrace(scale, static_cast<int>(static_cast<int32_t>(b)));
    else if (std::strcmp(mode, "softexp") == 0) { float beta; std::memcpy(&beta, &b, 4); SoftExp(scale, beta); }
    std::fflush(stdout);
  }
  return 0;
}
