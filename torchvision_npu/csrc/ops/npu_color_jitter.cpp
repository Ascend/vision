#include <ATen/ATen.h>
#include <torch/types.h>
#include "../macros.h"

namespace vision {
namespace ops {

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_adjust_brightness_aclop(Tensor self, Scalar factor) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_adjust_brightness_aclnn(Tensor self, float factor) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_adjust_contrast_aclop(Tensor self, Scalar factor, str mean_mode=\"chn_wise\") -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_adjust_contrast_aclnn(Tensor self, float factor) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_adjust_hue_aclop(Tensor self, Scalar factor) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_adjust_hue_aclnn(Tensor self, float factor) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_adjust_saturation_aclop(Tensor self, Scalar factor) -> Tensor"));
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::_adjust_saturation_aclnn(Tensor self, float factor) -> Tensor"));
}

} // namespace ops
} // namespace vision
