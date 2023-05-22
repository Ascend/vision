#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor npu_crop(
    const at::Tensor &self,
    at::IntArrayRef size,
    at::IntArrayRef offsets,
    int64_t axis);

} // namespace ops
} // namespace vision
