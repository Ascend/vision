#include "npu_crop_and_resize.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_crop_and_resize(
    const at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> boxes,
    at::IntArrayRef box_index,
    at::IntArrayRef crop_size,
    double extrapolation_value,
    std::string method) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchvision::npu_crop_and_resize", "")
      .typed<decltype(npu_crop_and_resize)>();
  return op.call(self, boxes, box_index, crop_size, extrapolation_value, method);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_crop_and_resize(Tensor self, \
      float[]? boxes, int[] box_index, int[] crop_size, \
      float extrapolation_value=0, str method=\"bilinear\") -> Tensor"));
}

} // namespace ops
} // namespace vision
