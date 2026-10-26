// Amendment 6.4 harness (TFLite Micro, reference kernels). Runs TFLM's own fixed-point primitives and the
// REAL FULLY_CONNECTED kernel (Register_FULLY_CONNECTED via KernelRunner); never re-implements them.
//
// Build (modelled on tanh-sweep-v1/README.md; run from the tflite-micro checkout root,
// /home/user/adaworldapi/tflite-micro, whose prebuilt lib is gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a):
//
//   g++ -std=c++17 -Os -fno-rtti -fno-exceptions -fno-threadsafe-statics
//     -DTFLM_REMOVE_VIRTUAL_DELETE -DTF_LITE_DISABLE_X86_NEON
//     -DKERNELS_OPTIMIZED_FOR_SPEED -DTF_LITE_USE_CTIME -Werror -Wall -Wextra
//     -Wno-unused-parameter -I. -Itensorflow/lite/micro/tools/make/downloads
//     -Itensorflow/lite/micro/tools/make/downloads/flatbuffers/include
//     -Itensorflow/lite/micro/tools/make/downloads/kissfft
//     -Igen/linux_x86_64_default_gcc/genfiles/
//     <lance-graph>/.claude/probes/int8-semantics-v1/tflm64.cc
//     gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a
//     -Wl,--gc-sections -lm -o tflm64
//
//   ./tflm64 fx < fx_cases.txt > tflm_fx.txt
//   ./tflm64 fc < fc_cases.txt > tflm_fc.txt
//
// (The library is built without TFLITE_SINGLE_ROUNDING, so MultiplyByQuantizedMultiplier is the
//  double-rounding form, kernels/internal/common.cc:77-88. Not verified by running this file.)
//
// ---- fx mode: stdin lines ----
//   S a b      -> S a b r    r = SaturatingRoundingDoublingHighMul(int32 a, int32 b)  (fixedpoint.h:328, int32 specialisation)
//   R x e      -> R x e r    r = RoundingDivideByPOT(int32 x, e)                       (fixedpoint.h:356; asserts 0<=e<=31)
//   M x m s    -> M x m s r  r = MultiplyByQuantizedMultiplier(int32 x, m, s)          (common.h:336, common.cc:80-88)
//
// ---- fc mode: stdin blocks (see spec "Exchange formats") ----
//   FC id per_channel batches in_dim out_dim in_scale_bits in_zp out_scale_bits out_zp act_min act_max
//   IN <batches*in_dim int8>      WS <out_dim f32 bit patterns, hex>      W <out_dim*in_dim int8>      B <out_dim int32>
//   (scale bit patterns are parsed with %x; ids are whitespace-free tokens.)
// stdout, per block:
//   FCOUT id prepare_ok invoke_ok <batches*out_dim int8>
//   FCQ id <out_dim multipliers> | <out_dim shifts>
//
// Notes on what the kernel can and cannot be told:
//  * Prepare decides per-channel vs per-tensor from the filter's scale array length: affine_quantization->scale->size > 1
//    (fully_connected_common.cc:75). So per_channel=1 with out_dim==1 runs the PER-TENSOR path inside TFLM; FCQ follows
//    the same rule (the multiplier is the same double expression either way).
//  * The fused-activation range comes from CalculateActivationRangeQuantized (kernel_util.cc:498-521; ActNone ->
//    int8 [-128,127]). A narrower act_min/act_max cannot be handed to the kernel. When the header asks for one, the
//    kernel runs with kTfLiteActNone and the harness clamps its int8 output afterwards to [act_min, act_max]. This is
//    exactly equivalent to an in-kernel clamp because the kernel's last step is max(act_min, min(act_max, v))
//    (integer_ops/fully_connected.h, spec L26-78) and clamping an already-[-128,127]-clamped value to a sub-range
//    equals clamping the unclamped value to that sub-range. A note is written to stderr for such cases.
//  * TFLITE_DCHECK(filter zero_point == 0) (fully_connected_common.cc:147): all filter zero points here are 0.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/fixedpoint.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"

namespace {
using namespace tflite::micro;

float F32FromBits(unsigned bits) {
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

int RunFx() {
  char op[16];
  while (std::scanf("%15s", op) == 1) {
    if (op[0] == 'S') {
      long long a, b;
      if (std::scanf("%lld %lld", &a, &b) != 2) return 2;
      const int32_t r = tflite::micro::SaturatingRoundingDoublingHighMul(static_cast<int32_t>(a), static_cast<int32_t>(b));
      std::printf("S %lld %lld %d\n", a, b, r);
    } else if (op[0] == 'R') {
      long long x;
      int e;
      if (std::scanf("%lld %d", &x, &e) != 2) return 2;
      const int32_t r = tflite::micro::RoundingDivideByPOT<int32_t>(static_cast<int32_t>(x), e);
      std::printf("R %lld %d %d\n", x, e, r);
    } else if (op[0] == 'M') {
      long long x, m;
      int s;
      if (std::scanf("%lld %lld %d", &x, &m, &s) != 3) return 2;
      // Explicit int32_t selects the int32 overload (common.h:336) rather than the int64 one (common.h:339).
      const int32_t r = MultiplyByQuantizedMultiplier(static_cast<int32_t>(x), static_cast<int32_t>(m), s);
      std::printf("M %lld %lld %d %d\n", x, m, s, r);
    } else {
      std::fprintf(stderr, "tflm64 fx: unknown op '%s'\n", op);
      return 2;
    }
  }
  return 0;
}

// Reads whitespace-separated tokens of one kind into a vector; no fixed line buffer.
bool ReadInts(std::vector<int>* v, size_t n) {
  v->resize(n);
  for (size_t i = 0; i < n; ++i) {
    int x;
    if (std::scanf("%d", &x) != 1) return false;
    (*v)[i] = x;
  }
  return true;
}

bool ExpectTag(const char* tag) {
  char t[16];
  if (std::scanf("%15s", t) != 1) return false;
  if (std::strcmp(t, tag) != 0) {
    std::fprintf(stderr, "tflm64 fc: expected tag %s, got %s\n", tag, t);
    return false;
  }
  return true;
}

int RunFc() {
  char tag[16];
  while (std::scanf("%15s", tag) == 1) {
    if (std::strcmp(tag, "FC") != 0) {
      std::fprintf(stderr, "tflm64 fc: expected FC, got %s\n", tag);
      return 2;
    }
    char id[128];
    int per_channel, batches, in_dim, out_dim, in_zp, out_zp, act_min, act_max;
    unsigned in_scale_bits, out_scale_bits;
    if (std::scanf("%127s %d %d %d %d %x %d %x %d %d %d", id, &per_channel, &batches, &in_dim, &out_dim, &in_scale_bits,
                   &in_zp, &out_scale_bits, &out_zp, &act_min, &act_max) != 11) {
      return 2;
    }
    const float in_scale = F32FromBits(in_scale_bits);
    const float out_scale = F32FromBits(out_scale_bits);

    std::vector<int> in_i, w_i, b_i;
    if (!ExpectTag("IN") || !ReadInts(&in_i, static_cast<size_t>(batches) * in_dim)) return 2;
    if (!ExpectTag("WS")) return 2;
    // float array in the test_helpers layout: element 0 is reserved for the length (FloatArrayFromFloats,
    // test_helpers.cc:2071-2077 overwrites it with the int size); data starts at element 1.
    std::vector<float> ws_arr(static_cast<size_t>(out_dim) + 1);
    ws_arr[0] = static_cast<float>(out_dim);
    for (int c = 0; c < out_dim; ++c) {
      unsigned bits;
      if (std::scanf("%x", &bits) != 1) return 2;
      ws_arr[c + 1] = F32FromBits(bits);
    }
    if (!ExpectTag("W") || !ReadInts(&w_i, static_cast<size_t>(out_dim) * in_dim)) return 2;
    if (!ExpectTag("B") || !ReadInts(&b_i, static_cast<size_t>(out_dim))) return 2;

    std::vector<int8_t> in(in_i.size()), w(w_i.size());
    std::vector<int32_t> bias(b_i.size());
    for (size_t i = 0; i < in_i.size(); ++i) in[i] = static_cast<int8_t>(in_i[i]);
    for (size_t i = 0; i < w_i.size(); ++i) w[i] = static_cast<int8_t>(w_i[i]);
    for (size_t i = 0; i < b_i.size(); ++i) bias[i] = static_cast<int32_t>(b_i[i]);
    std::vector<int8_t> out(static_cast<size_t>(batches) * out_dim, 0);

    // ---- FCQ: same helpers and same expressions as Prepare ----
    // Per-channel (scale->size > 1): fully_connected_common.cc:122-133:
    //   effective = double(input_scale) * double(filter_scale[i]) / double(output_scale); QuantizeMultiplier(effective, &sig, &shift)
    // Per-tensor: GetQuantizedConvolutionMultipler (kernel_util.cc:484-492):
    //   double(input_scale) * double(filter->params.scale) / double(output_scale), then QuantizeMultiplier
    //   (fully_connected_common.cc:136-140). QuantizeMultiplier = quantization_util.cc:54-.. (frexp, round(q*2^31)).
    const bool effective_pc = per_channel != 0 && out_dim > 1;
    std::vector<int32_t> qmult(out_dim, 0);
    std::vector<int> qshift(out_dim, 0);
    for (int c = 0; c < out_dim; ++c) {
      const double fs = static_cast<double>(effective_pc ? ws_arr[c + 1] : ws_arr[1]);
      const double eff = static_cast<double>(in_scale) * fs / static_cast<double>(out_scale);
      int32_t sig = 0;
      int sh = 0;
      tflite::micro::QuantizeMultiplier(eff, &sig, &sh);
      qmult[c] = sig;
      qshift[c] = sh;
    }

    // ---- tensors ----
    int in_dims_d[] = {2, batches, in_dim};
    int w_dims_d[] = {2, out_dim, in_dim};
    int b_dims_d[] = {1, out_dim};
    int o_dims_d[] = {2, batches, out_dim};
    TfLiteIntArray* in_dims = tflite::testing::IntArrayFromInts(in_dims_d);
    TfLiteIntArray* w_dims = tflite::testing::IntArrayFromInts(w_dims_d);
    TfLiteIntArray* b_dims = tflite::testing::IntArrayFromInts(b_dims_d);
    TfLiteIntArray* o_dims = tflite::testing::IntArrayFromInts(o_dims_d);

    // Per-channel quantization params, test_helpers.cc:2085-2097 layout: scale[0]/zero_point[0] hold the count.
    std::vector<int> zp_arr(static_cast<size_t>(out_dim) + 1, 0);
    zp_arr[0] = out_dim;
    std::vector<float> bias_scale_arr(static_cast<size_t>(out_dim) + 1);
    bias_scale_arr[0] = static_cast<float>(out_dim);
    for (int c = 0; c < out_dim; ++c) bias_scale_arr[c + 1] = in_scale * ws_arr[c + 1];
    std::vector<int> bzp_arr(static_cast<size_t>(out_dim) + 1, 0);
    bzp_arr[0] = out_dim;
    TfLiteAffineQuantization w_aff, b_aff;
    std::memset(&w_aff, 0, sizeof(w_aff));
    std::memset(&b_aff, 0, sizeof(b_aff));

    TfLiteTensor t_in = tflite::testing::CreateQuantizedTensor(in.data(), in_dims, in_scale, in_zp);
    TfLiteTensor t_w, t_b;
    if (per_channel != 0) {
      // Layout/helper: CreatePerChannelQuantizedTensor (test_helpers.h:365-376), quantized_dimension 0, zero points all 0.
      t_w = tflite::testing::CreatePerChannelQuantizedTensor(
          w.data(), w_dims, tflite::testing::FloatArrayFromFloats(ws_arr.data()), tflite::testing::IntArrayFromInts(zp_arr.data()),
          &w_aff, /*quantized_dimension=*/0);
      // params.scale is only read on the per-tensor path (kernel_util.cc:488); set it so out_dim==1 behaves.
      t_w.params.scale = ws_arr[1];
      t_w.params.zero_point = 0;
      t_b = tflite::testing::CreatePerChannelQuantizedTensor(
          bias.data(), b_dims, tflite::testing::FloatArrayFromFloats(bias_scale_arr.data()),
          tflite::testing::IntArrayFromInts(bzp_arr.data()), &b_aff, /*quantized_dimension=*/0);
      t_b.params.scale = in_scale * ws_arr[1];
      t_b.params.zero_point = 0;
    } else {
      // Per-tensor weights, zero point 0 (CreateQuantizedTensor, test_helpers.h:267-275).
      t_w = tflite::testing::CreateQuantizedTensor(w.data(), w_dims, ws_arr[1], 0);
      // Bias: int32, scale = input_scale * weight_scale, zero point 0 (CreateQuantizedBiasTensor convention,
      // test_helpers.h:289-303). Prepare checks |input_product_scale - bias_scale|/output_scale <= 0.02
      // (kernel_util.cc:456-471).
      t_b = tflite::testing::CreateQuantizedTensor(bias.data(), b_dims, in_scale * ws_arr[1], 0);
    }
    TfLiteTensor t_out = tflite::testing::CreateQuantizedTensor(out.data(), o_dims, out_scale, out_zp);
    TfLiteTensor tensors[4] = {t_in, t_w, t_b, t_out};

    int ia[] = {3, 0, 1, 2}, oa[] = {1, 3};
    TfLiteFullyConnectedParams params;
    std::memset(&params, 0, sizeof(params));
    params.activation = kTfLiteActNone;
    params.weights_format = kTfLiteFullyConnectedWeightsFormatDefault;
    params.keep_num_dims = false;
    params.asymmetric_quantize_inputs = false;
    params.quantized_bias_type = kTfLiteNoType;
    params.quant_spec = nullptr;
    params.quant_spec_size = 0;

    const TFLMRegistration reg = tflite::Register_FULLY_CONNECTED();
    tflite::micro::KernelRunner runner(reg, tensors, 4, tflite::testing::IntArrayFromInts(ia),
                                       tflite::testing::IntArrayFromInts(oa), &params);
    TfLiteStatus p = runner.InitAndPrepare();
    TfLiteStatus v = p == kTfLiteOk ? runner.Invoke() : kTfLiteError;

    if (p != kTfLiteOk) {
      for (auto& o : out) o = 0;
    } else if (act_min != -128 || act_max != 127) {
      std::fprintf(stderr, "tflm64 fc: case %s act range [%d,%d] applied as post-clamp of the ActNone kernel output\n", id,
                   act_min, act_max);
      for (auto& o : out) {
        int x = o;
        if (x < act_min) x = act_min;
        if (x > act_max) x = act_max;
        o = static_cast<int8_t>(x);
      }
    }

    std::printf("FCOUT %s %d %d", id, p == kTfLiteOk, v == kTfLiteOk);
    for (size_t i = 0; i < out.size(); ++i) std::printf(" %d", out[i]);
    std::printf("\n");
    std::printf("FCQ %s", id);
    for (int c = 0; c < out_dim; ++c) std::printf(" %d", qmult[c]);
    std::printf(" |");
    for (int c = 0; c < out_dim; ++c) std::printf(" %d", qshift[c]);
    std::printf("\n");
  }
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: tflm64 fx|fc < cases.txt\n");
    return 2;
  }
  if (std::strcmp(argv[1], "fx") == 0) return RunFx();
  if (std::strcmp(argv[1], "fc") == 0) return RunFc();
  std::fprintf(stderr, "tflm64: unknown mode %s\n", argv[1]);
  return 2;
}
