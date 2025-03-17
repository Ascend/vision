#pragma once

#include <ATen/ATen.h>
#include <vector>
#include "../macros.h"

namespace vision {
namespace ops {

VISION_API at::Tensor yuv2rgb(int64_t nthreads, int64_t width, int64_t height, int64_t color_range, int64_t color_space,
                              const at::Tensor ydata, std::vector<int64_t> src_stride, const at::Tensor udata, const at::Tensor vdata);

} // namespace ops
} // namespace vision
