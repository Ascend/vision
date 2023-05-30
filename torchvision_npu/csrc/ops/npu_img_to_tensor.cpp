#include "npu_img_to_tensor.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_img_to_tensor(
    const at::Tensor &self) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchvision::npu_img_to_tensor", "")
      .typed<decltype(npu_img_to_tensor)>();
  return op.call(self);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_img_to_tensor(Tensor self) -> Tensor"));
}

} // namespace ops
} // namespace vision
