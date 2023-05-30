#include "npu_resize.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_resize(
    const at::Tensor &self,
    at::IntArrayRef size,
    double cubic_coeff_a,
    int64_t exclude_outside,
    std::string mode,
    std::string nearest_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchvision::npu_resize", "")
      .typed<decltype(npu_resize)>();
  return op.call(self, size, cubic_coeff_a, exclude_outside, mode, nearest_mode);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_resize(Tensor self, int[] size, \
      float cubic_coeff_a=-0.75, int exclude_outside=0, str mode=\"nearest\", \
      str nearest_mode=\"round_prefer_floor\") -> Tensor"));
}

} // namespace ops
} // namespace vision
