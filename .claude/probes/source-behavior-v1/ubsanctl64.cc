// D-TFLP Amendment 6.4 follow-up: can-fire control for the UBSan build flags used on recipall63.
// Built with the same flags, each mode must be reported by the sanitizer (non-zero exit):
//   add   the softmax accumulation add of fixedpoint.h:94 on a sum past INT32_MAX (4,096 maxima);
//   shift GetReciprocal (common.h:976) on a wrapped sum of 0 (8,192 maxima).
// "none" runs both operations on admitted values and must exit 0.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/fixedpoint.h"

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "none";
  volatile int32_t one = 524288;  // Rescale<12>(exp(0)) = 2^19, the weight of a row maximum
  int32_t s = 0;
  const int n = std::strcmp(mode, "add") == 0 ? 4096 : 511;
  for (int i = 0; i < n; ++i) s = tflite::micro::Add(s, (int32_t)one);
  int nb = 0;
  const int32_t x = std::strcmp(mode, "shift") == 0 ? 0 : s;
  const int32_t r = tflite::micro::GetReciprocal(x, 12, &nb);
  std::printf("%s sum=%d recip=%d nb=%d\n", mode, s, r, nb);
  return 0;
}
