#include "yuv420p_rgb.h"

namespace vision {
namespace ops {
    
    at::Tensor kp_yuv2rgb(int64_t nthreads, int64_t width, int64_t height, int64_t color_range, int64_t color_space,
                          const at::Tensor ydata, std::vector<int64_t> src_stride, const at::Tensor udata,
                          const at::Tensor vdata)
    {
        int channels = 3;
        at::Tensor tmp = at::empty({height, width, channels}, at::kByte);
        if (color_range == KP_AVCOL_RANGE_JPEG) {
            kp_yuv420p_to_rgb_full(nthreads, width, height,
                                   (const uint8_t*)ydata.data_ptr(),
                                   src_stride.data(),
                                   (const uint8_t*)udata.data_ptr(),
                                   (const uint8_t*)vdata.data_ptr(),
                                   (uint8_t*)tmp.data_ptr());
        } else {
            kp_yuv420p_to_rgb_limit(nthreads, width, height,
                                    (const uint8_t*)ydata.data_ptr(),
                                    src_stride.data(),
                                    (const uint8_t*)udata.data_ptr(),
                                    (const uint8_t*)vdata.data_ptr(),
                                    (uint8_t*)tmp.data_ptr());
        }
        return tmp;
    }

    TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
        m.impl(TORCH_SELECTIVE_NAME("torchvision::yuv2rgb"), TORCH_FN(kp_yuv2rgb));
    }

} // namespace ops
} // namespace vision
