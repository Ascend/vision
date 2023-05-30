#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor npu_img_to_tensor(
    const at::Tensor &self);

} // namespace ops
} // namespace vision