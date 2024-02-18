#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_pad_aclop(Tensor self, int[] pad, int constant_values=0, str mode=\"constant\") -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_pad_aclnn(Tensor self, int[] padding, int padding_mode=0, float[]? fill=None) -> Tensor"));
}

} // namespace ops
} // namespace vision
