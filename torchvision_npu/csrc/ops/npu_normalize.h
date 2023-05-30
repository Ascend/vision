#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor npu_normalize(
    const at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    at::ScalarType dtype);

VISION_API at::Tensor& npu_normalize_(
    at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    at::ScalarType dtype);

} // namespace ops
} // namespace vision
