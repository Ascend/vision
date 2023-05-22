#include "npu_gaussian_blur.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_gaussian_blur(
    const at::Tensor &self,
    at::IntArrayRef kernel_size,
    c10::optional<c10::ArrayRef<double>> sigma,
    std::string padding_mode) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::npu_gaussian_blur", "")
                       .typed<decltype(npu_gaussian_blur)>();
  return op.call(self, kernel_size, sigma, padding_mode);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_gaussian_blur(Tensor self, int[] kernel_size, float[]? sigma, \
      str padding_mode=\"constant\") -> Tensor"));
}

} // namespace ops
} // namespace vision
