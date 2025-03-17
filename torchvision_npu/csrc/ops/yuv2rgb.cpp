#include "yuv2rgb.h"
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>
#include <vector>

namespace vision {
namespace ops {

at::Tensor yuv2rgb(int64_t nthreads, int64_t width, int64_t height, int64_t color_range, int64_t color_space,
                   const at::Tensor ydata, std::vector<int64_t> src_stride, const at::Tensor udata,
                   const at::Tensor vdata)
{
    C10_LOG_API_USAGE_ONCE("torchvision_npu.csrc.ops.yuv2rgb.yuv2rgb");
    static auto op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow("torchvision::yuv2rgb", "")
                        .typed<decltype(yuv2rgb)>();
    return op.call(nthreads, width, height, color_range, color_space, ydata, src_stride, udata, vdata);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::yuv2rgb(int nthreads, int width, int height, int color_range, int color_space, Tensor ydata, int[] src_stride, Tensor udata, Tensor vdata) -> Tensor"));
}

} // namespace ops
} // namespace vision
