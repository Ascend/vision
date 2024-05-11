#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_warp_affine_aclnn(Tensor self, float[]? matrix, int interpolation_mode=1, \
        int padding_mode=0, float[]? fill=None) -> Tensor"));
}

} // namespace ops
} // namespace vision
