#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_crop_aclop(Tensor self, int[] size, int[] offsets, int axis=2) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_crop_aclnn(Tensor self, int top, int left, int height, int width) -> Tensor"));
}

} // namespace ops
} // namespace vision
