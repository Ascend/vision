#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_decode_jpeg_aclop(Tensor self, int[] image_shape, \
        int channels=3, bool try_recover_truncated=False) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_decode_jpeg_aclnn(Tensor self, int[] image_shape, int channels=3, \
        bool try_recover_truncated=True) -> Tensor"));
}

} // namespace ops
} // namespace vision
