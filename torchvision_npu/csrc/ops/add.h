#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor add(
    const at::Tensor& a,
    const at::Tensor& b);

} // namespace ops
} // namespace vision
