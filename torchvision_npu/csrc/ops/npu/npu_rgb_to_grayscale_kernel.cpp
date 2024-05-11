#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;

at::Tensor rgb_to_grayscale_aclnn_kernel(const at::Tensor& self, int64_t output_channels_num)
{
    c10::SmallVector<int64_t, SIZE> output_size = {self.size(0), output_channels_num, self.size(2), self.size(3)};
    at::Tensor result = at::empty(output_size, self.options());
    EXEC_NPU_CMD(acldvppRgbToGrayscale, self, output_channels_num, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_rgb_to_grayscale_aclnn"), TORCH_FN(rgb_to_grayscale_aclnn_kernel));
}

} // namespace ops
} // namespace vision
