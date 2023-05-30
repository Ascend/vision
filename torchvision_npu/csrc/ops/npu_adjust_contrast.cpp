#include "npu_adjust_contrast.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_adjust_contrast(
    const at::Tensor &self,
    at::Scalar factor,
    std::string mean_mode) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchvision::npu_adjust_contrast", "")
      .typed<decltype(npu_adjust_contrast)>();
  return op.call(self, factor, mean_mode);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_adjust_contrast(Tensor self, Scalar factor, str mean_mode=\"chn_wise\") -> Tensor"));
}

} // namespace ops
} // namespace vision
