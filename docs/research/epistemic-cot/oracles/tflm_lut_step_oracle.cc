// Generator of crates/lance-graph-planner/src/nars/data/tflm_lut_step_golden.txt.
// Built against the real TFLite-micro sources (AdaWorldAPI/tflite-micro @ 5f64e4a):
//   g++ -std=c++17 -O1 -I<tflite-micro> tflm_lut_step_oracle.cc
//   ./a.out > tflm_lut_step_golden.txt
// Each line: `L <num> <den> <int8 code> <LUTPopulateInt8 output>`, for the
// learning step f(x) = x * (num/den) at unit scale, zero point 0 (Sym).
// Not built by cargo; kept so the fixture can be regenerated.
#include <cstdio>
#include <cstdint>
#include "tensorflow/lite/micro/kernels/internal/common.h"
using namespace tflite;
static float eta;
static float step(float x) { return x * eta; }
int main() {
  const int rates[][2] = {{1, 4}, {1, 2}, {3, 4}, {1, 1}};
  for (auto& r : rates) {
    eta = (float)r[0] / (float)r[1];
    int8_t lut[256];
    LUTPopulate<int8_t>(1.0f, 0, 1.0f, 0, step, lut);
    for (int v = -128; v <= 127; ++v)
      printf("L %d %d %d %d\n", r[0], r[1], v, (int)lut[(uint8_t)(int8_t)v]);
  }
}
