#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;

at::Tensor crop_aclop_kernel(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef offsets,
    int64_t axis)
{
    TORCH_CHECK(size.size() == self.sizes().size(),
        "Op[crop_aclop] argument[size] represents output shape, (N, C, H, W).");
    
    at::Tensor result = at::empty(size, self.options());

    at_npu::native::OpCommand cmd;
    cmd.Name("Crop")
        .Input(self)
        .Input(result)
        .Output(result)
        .Attr("axis", axis)
        .Attr("offsets", offsets)
        .Run();

    return result;
}

at::Tensor crop_aclnn_kernel(
    const at::Tensor& self,
    int64_t top, int64_t left, int64_t height, int64_t width)
{
    c10::SmallVector<int64_t, SIZE> output_size = {1, self.size(1), height, width};
    at::Tensor result = at::empty(output_size, self.options());
    EXEC_NPU_CMD(acldvppCrop, self, top, left, height, width, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_crop_aclop"), TORCH_FN(crop_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_crop_aclnn"), TORCH_FN(crop_aclnn_kernel));
}

} // namespace ops
} // namespace vision
