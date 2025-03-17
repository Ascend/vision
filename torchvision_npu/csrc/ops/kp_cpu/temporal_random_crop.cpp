#include <ATen/ATen.h>
#include <omp.h>
#include <cstring>
#include <torch/library.h>
#include "FastMemcpy.h"

namespace vision {
namespace ops {

at::Tensor kp_temporal_random_crop_kernel(const at::Tensor &vframes, const at::Tensor &frame_ind, int64_t thread_num)
{
    auto shape = vframes.sizes();
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];
    int out_num_frames = frame_ind.sizes()[0];
    at::Tensor tmp_video = at::empty({out_num_frames, channels, height, width}, vframes.dtype());
    long long video_memcpy_size = channels * height * width * vframes.dtype().itemsize();

#pragma omp parallel for firstprivate(video_memcpy_size, frame_ind) num_threads(thread_num)
    for (int j = 0; j < out_num_frames; ++j) {
        kp_memcpy_fast(tmp_video[j].data_ptr(), vframes[frame_ind[j]].data_ptr(), video_memcpy_size);
    }
    return tmp_video;
}

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::kp_temporal_random_crop"), TORCH_FN(kp_temporal_random_crop_kernel));
}

} // namespace ops
} // namespace vision
