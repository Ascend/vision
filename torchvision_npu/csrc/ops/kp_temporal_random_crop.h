#pragma once

#include <ATen/ATen.h>
#include "../macros.h"

namespace vision {
namespace ops {

    VISION_API at::Tensor kp_temporal_random_crop(const at::Tensor &vframes, const at::Tensor &frame_ind, int64_t thread_num);

} // namespace ops
} // namespace vision
