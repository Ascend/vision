#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor npu_adjust_contrast(
    const at::Tensor &self,
    at::Scalar factor,
    std::string mean_mode);

} // namespace ops
} // namespace vision
