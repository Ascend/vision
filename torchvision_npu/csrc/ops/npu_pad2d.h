#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor npu_pad2d(
    const at::Tensor &self,
    at::IntArrayRef pad,
    int64_t constant_values,
    std::string mode);

} // namespace ops
} // namespace vision
