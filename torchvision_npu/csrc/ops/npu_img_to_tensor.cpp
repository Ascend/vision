#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_img_to_tensor_aclop(Tensor self) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_img_to_tensor_aclnn(Tensor self) -> Tensor"));
}

} // namespace ops
} // namespace vision
