#include "npu_decode_jpeg.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_decode_jpeg(
    const at::Tensor &self,
    at::IntArrayRef image_shape,
    int64_t channels,
    bool try_recover_truncated) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchvision::npu_decode_jpeg", "")
      .typed<decltype(npu_decode_jpeg)>();
  return op.call(self, image_shape, channels, try_recover_truncated);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_decode_jpeg(Tensor self, int[] image_shape, \
      int channels=3, bool try_recover_truncated=False) -> Tensor"));
}

} // namespace ops
} // namespace vision
