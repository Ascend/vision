#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_reverse_aclop(Tensor self, int[] axis) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_horizontal_flip_aclnn(Tensor self) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_vertical_flip_aclnn(Tensor self) -> Tensor"));
}

} // namespace ops
} // namespace vision
