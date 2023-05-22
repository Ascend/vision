#include "npu_pad2d.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_pad2d(
    const at::Tensor &self,
    at::IntArrayRef pad,
    int64_t constant_values,
    std::string mode) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::npu_pad2d", "")
                       .typed<decltype(npu_pad2d)>();
  return op.call(self, pad, constant_values, mode);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_pad2d(Tensor self, int[] pad, int constant_values=0, str mode=\"constant\") -> Tensor"));
}

} // namespace ops
} // namespace vision
