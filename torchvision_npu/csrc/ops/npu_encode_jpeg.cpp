#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_encode_jpeg_aclnn(Tensor self, int quality) -> Tensor"));
}

} // namespace ops
} // namespace vision
