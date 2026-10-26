// Runs TFLite Micro's real int8 SOFTMAX (Register_SOFTMAX via KernelRunner).
// stdin, one case per line: n scale_bits zero_point beta_bits c_0 .. c_{n-1}
// stdout, one line per case: prepare_ok invoke_ok o_0 .. o_{n-1}
// Output contract: int8, scale 1/256, zero point -128.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"

int main() {
  int n; unsigned sb, bb; int zp;
  while (std::scanf("%d %x %d %x", &n, &sb, &zp, &bb) == 4) {
    float scale, beta;
    std::memcpy(&scale, &sb, 4); std::memcpy(&beta, &bb, 4);
    std::vector<int8_t> in(n), out(n);
    for (int i = 0; i < n; ++i) { int v; std::scanf("%d", &v); in[i] = (int8_t)v; }
    int dims_data[] = {2, 1, n};
    TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(dims_data);
    TfLiteTensor tensors[2] = {
        tflite::testing::CreateQuantizedTensor(in.data(), dims, scale, zp),
        tflite::testing::CreateQuantizedTensor(out.data(), dims, 1.0f / 256.0f, -128)};
    int ia[] = {1, 0}, oa[] = {1, 1};
    TfLiteSoftmaxParams params = {beta};
    const TFLMRegistration reg = tflite::Register_SOFTMAX();
    tflite::micro::KernelRunner runner(reg, tensors, 2, tflite::testing::IntArrayFromInts(ia),
                                       tflite::testing::IntArrayFromInts(oa), &params);
    TfLiteStatus p = runner.InitAndPrepare();
    TfLiteStatus v = p == kTfLiteOk ? runner.Invoke() : kTfLiteError;
    std::printf("%d %d", p == kTfLiteOk, v == kTfLiteOk);
    for (int i = 0; i < n; ++i) std::printf(" %d", out[i]);
    std::printf("\n");
  }
  return 0;
}
