#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_rotate_aclnn(Tensor self, float angle, int interpolation_mode=1, bool expand=False, \
        int[] center=[], int padding_mode=0, float[]? fill=None) -> Tensor"));
}

} // namespace ops
} // namespace vision
