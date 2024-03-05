#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_gaussian_blur_aclop(Tensor self, int[] kernel_size, float[]? sigma, \
        str padding_mode=\"constant\") -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_gaussian_blur_aclnn(Tensor self, int[] kernel_size, float[]? sigma, \
        int padding_mode=0) -> Tensor"));
}

} // namespace ops
} // namespace vision
