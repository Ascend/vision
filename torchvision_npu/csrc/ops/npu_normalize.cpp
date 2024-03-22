#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_normalize_aclop(Tensor self, \
        float[]? mean, float[]? variance, ScalarType dtype=float) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_normalize_aclnn(Tensor self, float[]? mean, float[]? std) -> Tensor"));
}

} // namespace ops
} // namespace vision
