// Source-behavior lab harness for TFLite Micro int8 TANH / LOGISTIC (Amendment 6.2).
//
// The real kernels (Register_TANH, Register_LOGISTIC) are the oracle. The TANH
// trace re-executes the int8 loop body of reference_integer_ops::Tanh
// (integer_ops/tanh.h) stage by stage with the SAME TFLM fixed-point functions,
// and checks, for every code, that its final byte equals the real kernel's byte;
// a mismatch aborts (the trace is only trusted where it reproduces the kernel).
//
// stdin, one request per line:
//   trace <scale_bits_hex> <zp>   -> 256 lines: T code input radius in_q4 n_raw e_raw t_raw raw_out q24 trace_byte kernel_byte
//   logistic <scale_bits_hex> <zp> -> 1 line: L bits zp prep invoke mult shift radius out[256]   (out scale 1/256, zp -128)
//   chain <scale_bits_hex> <zp>    -> 1 line: C bits zp tanh_out[256] logistic_out[256]
//        TANH(in calib -> out 1/128, 0) as one kernel call, its int8 output tensor
//        fed to LOGISTIC(in 1/128, 0 -> out 1/256, -128) as a second kernel call.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/cppmath.h"
#include "tensorflow/lite/micro/kernels/internal/fixedpoint.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/test_helpers.h"

namespace {

TfLiteStatus RunOp(const TFLMRegistration& reg, const int8_t* in_data, float in_scale, int in_zp,
                   float out_scale, int out_zp, int8_t* out, TfLiteStatus* inv) {
  int8_t input[256];
  std::memcpy(input, in_data, 256);
  int dims_data[] = {2, 1, 256};
  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(dims_data);
  TfLiteTensor tensors[2] = {
      tflite::testing::CreateQuantizedTensor(input, dims, in_scale, in_zp),
      tflite::testing::CreateQuantizedTensor(out, dims, out_scale, out_zp)};
  int inputs_data[] = {1, 0};
  int outputs_data[] = {1, 1};
  tflite::micro::KernelRunner runner(reg, tensors, 2, tflite::testing::IntArrayFromInts(inputs_data),
                                     tflite::testing::IntArrayFromInts(outputs_data), nullptr);
  TfLiteStatus prep = runner.InitAndPrepare();
  *inv = prep == kTfLiteOk ? runner.Invoke() : kTfLiteError;
  return prep;
}

struct Prep { int32_t mult; int shift; int32_t radius; };
Prep Prepare(float scale) {  // tanh.cc / logistic_common.cc int8 prepare, same TFLM functions
  constexpr int kInputIntegerBits = 4;
  const double real = static_cast<double>(scale) * static_cast<double>(1 << (31 - kInputIntegerBits));
  Prep p{};
  const double q = std::frexp(real, &p.shift);
  p.mult = static_cast<int32_t>(tflite::TfLiteRound(q * (1ll << 31)));
  p.radius = tflite::CalculateInputRadius(kInputIntegerBits, p.shift, 31);
  return p;
}

void Codes(int8_t* in) { for (int i = 0; i < 256; ++i) in[i] = static_cast<int8_t>(i - 128); }

void Trace(float scale, int zp) {
  using namespace tflite::micro;  // FixedPoint, exp_on_negative_values, ... live here
  using F4 = FixedPoint<int32_t, 4>;
  int8_t in[256], kernel[256];
  Codes(in);
  TfLiteStatus inv;
  if (RunOp(tflite::Register_TANH(), in, scale, zp, 1.0f / 128.0f, 0, kernel, &inv) != kTfLiteOk || inv != kTfLiteOk) {
    std::printf("ERR prepare/invoke\n"); std::exit(2);
  }
  const Prep p = Prepare(scale);
  for (int i = 0; i < 256; ++i) {
    const int32_t input = static_cast<int32_t>(in[i]) - zp;
    int32_t in_q4 = 0, n_raw = 0, e_raw = 0, t_raw = 0, raw_out = 0, q24 = 0; int byte;
    if (input <= -p.radius) { byte = -128; }
    else if (input >= p.radius) { byte = 127; }
    else {
      in_q4 = tflite::micro::MultiplyByQuantizedMultiplier(input, p.mult, p.shift);
      F4 a = F4::FromRaw(in_q4);
      // tanh(a) from fixedpoint.h, expanded into its stages
      const int32_t mask_if_negative = MaskIfLessThan(a, F4::Zero());
      const int32_t mask_if_zero = MaskIfZero(a);
      F4 n = SelectUsingMask(mask_if_negative, a, -a);
      n_raw = n.raw();
      auto e = exp_on_negative_values(ExactMulByPot<1>(n));
      e_raw = e.raw();
      auto t = one_minus_x_over_one_plus_x_for_x_in_0_1(e);
      t_raw = t.raw();
      using R = FixedPoint<int32_t, 0>;
      raw_out = SelectUsingMask(mask_if_zero, R::Zero(), SelectUsingMask(mask_if_negative, -t, t)).raw();
      q24 = tflite::micro::RoundingDivideByPOT(raw_out, 31 - 7);
      int32_t c = q24 < -128 ? -128 : (q24 > 127 ? 127 : q24);
      byte = c;
    }
    if (byte != kernel[i]) { std::printf("ERR trace %d != kernel %d at code %d\n", byte, kernel[i], i - 128); std::exit(3); }
    std::printf("T %d %d %d %d %d %d %d %d %d %d %d\n", i - 128, input, p.radius, in_q4, n_raw, e_raw, t_raw,
                raw_out, q24, byte, kernel[i]);
  }
}

}  // namespace

int main() {
  char mode[16];
  unsigned int bits = 0;
  int zp = 0;
  while (std::scanf("%15s %x %d", mode, &bits, &zp) == 3) {
    float scale;
    std::memcpy(&scale, &bits, sizeof bits);
    if (std::strcmp(mode, "trace") == 0) { Trace(scale, zp); continue; }
    int8_t in[256], o1[256] = {0}, o2[256] = {0};
    Codes(in);
    TfLiteStatus inv1, inv2;
    if (std::strcmp(mode, "logistic") == 0) {
      TfLiteStatus prep = RunOp(tflite::Register_LOGISTIC(), in, scale, zp, 1.0f / 256.0f, -128, o1, &inv1);
      const Prep p = Prepare(scale);
      std::printf("L %08x %d %d %d %d %d %d", bits, zp, prep == kTfLiteOk, inv1 == kTfLiteOk, p.mult, p.shift, p.radius);
      for (int i = 0; i < 256; ++i) std::printf(" %d", o1[i]);
      std::printf("\n");
    } else if (std::strcmp(mode, "chain") == 0) {
      TfLiteStatus p1 = RunOp(tflite::Register_TANH(), in, scale, zp, 1.0f / 128.0f, 0, o1, &inv1);
      TfLiteStatus p2 = RunOp(tflite::Register_LOGISTIC(), o1, 1.0f / 128.0f, 0, 1.0f / 256.0f, -128, o2, &inv2);
      std::printf("C %08x %d %d", bits, zp, p1 == kTfLiteOk && inv1 == kTfLiteOk && p2 == kTfLiteOk && inv2 == kTfLiteOk);
      for (int i = 0; i < 256; ++i) std::printf(" %d", o1[i]);
      for (int i = 0; i < 256; ++i) std::printf(" %d", o2[i]);
      std::printf("\n");
    }
    std::fflush(stdout);
  }
  return 0;
}
