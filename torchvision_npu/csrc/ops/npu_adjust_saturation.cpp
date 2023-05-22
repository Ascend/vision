#include "npu_adjust_saturation.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_adjust_saturation(
    const at::Tensor &self,
    at::Scalar factor) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::npu_adjust_saturation", "")
                       .typed<decltype(npu_adjust_saturation)>();
  return op.call(self, factor);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_adjust_saturation(Tensor self, Scalar factor) -> Tensor"));
}

} // namespace ops
} // namespace vision
