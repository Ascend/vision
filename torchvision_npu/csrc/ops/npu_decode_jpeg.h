#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor npu_decode_jpeg(
    const at::Tensor &self,
    at::IntArrayRef image_shape,
    int64_t channels,
    bool try_recover_truncated);

} // namespace ops
} // namespace vision
