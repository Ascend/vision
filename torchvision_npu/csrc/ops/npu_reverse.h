#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor npu_reverse(
    const at::Tensor &self,
    at::IntArrayRef axis);

} // namespace ops
} // namespace vision
