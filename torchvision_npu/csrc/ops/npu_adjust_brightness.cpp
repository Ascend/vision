#include "npu_adjust_brightness.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_adjust_brightness(
    const at::Tensor &self,
    at::Scalar factor) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchvision::npu_adjust_brightness", "")
      .typed<decltype(npu_adjust_brightness)>();
  return op.call(self, factor);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_adjust_brightness(Tensor self, Scalar factor) -> Tensor"));
}

} // namespace ops
} // namespace vision
