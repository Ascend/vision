#pragma once
 
#include <ATen/ATen.h>
#include "../../macros.h"
 
namespace vision {
namespace ops {
 
VISION_API at::Tensor to_tensor_moal(const at::Tensor &clip);
 
} // namespace ops
} // namespace vision
