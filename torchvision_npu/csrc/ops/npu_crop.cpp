#include "npu_crop.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_crop(
    const at::Tensor &self,
    at::IntArrayRef size,
    at::IntArrayRef offsets,
    int64_t axis) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchvision::npu_crop", "")
      .typed<decltype(npu_crop)>();
  return op.call(self, size, offsets, axis);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_crop(Tensor self, int[] size, int[] offsets, int axis=2) -> Tensor"));
}

} // namespace ops
} // namespace vision
