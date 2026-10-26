// Calibration SWEEP variant of ndarray tools/tflm_oracle/tanh_int8_oracle.cc
// (RunTanh copied verbatim). Reads calibrations from stdin.
//
// Independent oracle for TFLite Micro int8 TANH.
//
// Runs the real TFLM kernel (tflite::Register_TANH(): tanh.cc prepare + the
// reference integer_ops/tanh.h eval) through micro::KernelRunner, once per
// input calibration, over all 256 int8 input codes, and prints the output
// bytes as JSON. Nothing here re-implements the kernel; it only builds
// tensors and calls TFLM. It is compiled against a TFLite Micro checkout and
// linked with that checkout's libtensorflow-microlite.a (see README.md).
//
// The prepared parameters (input_multiplier, input_left_shift,
// input_range_radius) are private to tanh.cc. They are recomputed below with
// the same TFLM functions (frexp, TfLiteRound, CalculateInputRadius) purely
// for the record; they do not feed the output bytes.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/cppmath.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/test_helpers.h"

namespace {

struct Calib {
  float scale;
  int zero_point;
};

constexpr float kOutScale = 1.0f / 128.0f;
constexpr int kOutZeroPoint = 0;

// Runs Register_TANH over all 256 codes. Returns the TfLiteStatus of
// InitAndPrepare (Invoke runs only if prepare succeeded).
TfLiteStatus RunTanh(const Calib& in, float out_scale, int out_zp, int8_t* out,
                     TfLiteStatus* invoke_status) {
  int8_t input[256];
  for (int i = 0; i < 256; ++i) input[i] = static_cast<int8_t>(i - 128);
  int dims_data[] = {2, 1, 256};
  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(dims_data);
  TfLiteTensor tensors[2] = {
      tflite::testing::CreateQuantizedTensor(input, dims, in.scale,
                                             in.zero_point),
      tflite::testing::CreateQuantizedTensor(out, dims, out_scale, out_zp)};
  int inputs_data[] = {1, 0};
  int outputs_data[] = {1, 1};
  const TFLMRegistration registration = tflite::Register_TANH();
  tflite::micro::KernelRunner runner(
      registration, tensors, 2, tflite::testing::IntArrayFromInts(inputs_data),
      tflite::testing::IntArrayFromInts(outputs_data), nullptr);
  TfLiteStatus prep = runner.InitAndPrepare();
  *invoke_status = prep == kTfLiteOk ? runner.Invoke() : kTfLiteError;
  return prep;
}

}  // namespace


// stdin: one calibration per line, "scale_bits_hex zero_point" (f32 bits).
// stdout: one line per calibration:
//   scale_bits zp prepare_ok invoke_ok input_multiplier input_left_shift input_range_radius out[0..256)
int main() {
  unsigned int bits = 0;
  int zp = 0;
  while (std::scanf("%x %d", &bits, &zp) == 2) {
    Calib in;
    std::memcpy(&in.scale, &bits, sizeof bits);
    in.zero_point = zp;
    int8_t out[256] = {0};
    TfLiteStatus inv;
    TfLiteStatus prep = RunTanh(in, kOutScale, kOutZeroPoint, out, &inv);
    constexpr int kInputIntegerBits = 4;
    const double real = static_cast<double>(in.scale) *
                        static_cast<double>(1 << (31 - kInputIntegerBits));
    int shift = 0;
    const double q = std::frexp(real, &shift);
    const int32_t mult = static_cast<int32_t>(tflite::TfLiteRound(q * (1ll << 31)));
    const int32_t radius = tflite::CalculateInputRadius(kInputIntegerBits, shift, 31);
    std::printf("%08x %d %d %d %d %d %d", bits, zp, prep == kTfLiteOk, inv == kTfLiteOk, mult, shift, radius);
    for (int i = 0; i < 256; ++i) std::printf(" %d", out[i]);
    std::printf("\n");
  }
  return 0;
}
