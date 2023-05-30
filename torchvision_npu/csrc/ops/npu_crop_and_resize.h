#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor npu_crop_and_resize(
    const at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> boxes,
    at::IntArrayRef box_index,
    at::IntArrayRef crop_size,
    double extrapolation_value,
    std::string method);

} // namespace ops
} // namespace vision
