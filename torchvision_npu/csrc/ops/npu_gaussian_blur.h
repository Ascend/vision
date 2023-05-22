#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor npu_gaussian_blur(
    const at::Tensor &self,
    at::IntArrayRef kernel_size,
    c10::optional<c10::ArrayRef<double>> sigma,
    std::string padding_mode);

} // namespace ops
} // namespace vision
