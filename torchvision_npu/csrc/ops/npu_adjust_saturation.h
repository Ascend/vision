#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor npu_adjust_saturation(
    const at::Tensor &self,
    at::Scalar factor);

} // namespace ops
} // namespace vision
