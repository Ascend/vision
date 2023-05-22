#include "npu_adjust_hue.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_adjust_hue(
    const at::Tensor &self,
    at::Scalar factor) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::npu_adjust_hue", "")
                       .typed<decltype(npu_adjust_hue)>();
  return op.call(self, factor);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_adjust_hue(Tensor self, Scalar factor) -> Tensor"));
}

} // namespace ops
} // namespace vision
