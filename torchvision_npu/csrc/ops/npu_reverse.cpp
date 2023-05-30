#include "npu_reverse.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_reverse(
    const at::Tensor &self,
    at::IntArrayRef axis) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchvision::npu_reverse", "")
      .typed<decltype(npu_reverse)>();
  return op.call(self, axis);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_reverse(Tensor self, int[] axis) -> Tensor"));
}

} // namespace ops
} // namespace vision
