#pragma once
 
#include <ATen/ATen.h>
#include <vector>
#include "../../macros.h"
 
namespace vision {
namespace ops {
 
VISION_API at::Tensor normalize_moal(at::Tensor tensor, const std::vector<double>& mean, const std::vector<double>& std, bool inplace = false);
 
} // namespace ops
} // namespace vision
