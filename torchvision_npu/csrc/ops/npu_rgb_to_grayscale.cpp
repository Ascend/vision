#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_rgb_to_grayscale_aclnn(Tensor self, int output_channels_num) -> Tensor"));
}

} // namespace ops
} // namespace vision
