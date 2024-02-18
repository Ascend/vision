#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_resize_aclop(Tensor self, int[] size, \
        float cubic_coeff_a=-0.75, int exclude_outside=0, str mode=\"nearest\", \
        str nearest_mode=\"round_prefer_floor\") -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_resize_aclnn(Tensor self, int[] size, int interpolation_mode=1) -> Tensor"));
}

} // namespace ops
} // namespace vision
