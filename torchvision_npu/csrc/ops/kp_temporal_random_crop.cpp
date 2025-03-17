#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>
#include "kp_temporal_random_crop.h"

namespace vision {
namespace ops {

at::Tensor kp_temporal_random_crop(const at::Tensor &vframes, const at::Tensor &frame_ind, int64_t thread_num)
{
    C10_LOG_API_USAGE_ONCE("torchvision_npu.csrc.ops.kp_temporal_random_crop.kp_temporal_random_crop");
    static auto op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow("torchvision::kp_temporal_random_crop", "")
                        .typed<decltype(kp_temporal_random_crop)>();

    return op.call(vframes, frame_ind, thread_num);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::kp_temporal_random_crop(Tensor vframes, Tensor frame_ind, int thread_num) -> (Tensor)"));
}

} // namespace ops
} // namespace vision
