#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_crop_and_resize_aclop(Tensor self, \
        float[]? boxes, int[] box_index, int[] crop_size, \
        float extrapolation_value=0, str method=\"bilinear\") -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_crop_and_resize_aclnn(Tensor self, int top, int left, int height, int width, \
        int[] size, int interpolation_mode=0) -> Tensor"));
}

} // namespace ops
} // namespace vision
