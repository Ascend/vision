#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor npu_resize(
    const at::Tensor &self,
    at::IntArrayRef size,
    double cubic_coeff_a,
    int64_t exclude_outside,
    std::string mode,
    std::string nearest_mode);

} // namespace ops
} // namespace vision
