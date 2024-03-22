#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

at::Tensor reverse_aclop_kernel(const at::Tensor& self, at::IntArrayRef axis)
{
    at::Tensor result = at::empty(self.sizes(), self.options());

    at_npu::native::OpCommand cmd;
    cmd.Name("ReverseV2")
        .Input(self)
        .Input(axis, at::kInt)
        .Output(result)
        .Run();

    return result;
}

at::Tensor horizontal_flip_aclnn_kernel(const at::Tensor& self)
{
    at::Tensor result = at::empty(self.sizes(), self.options());
    EXEC_NPU_CMD(acldvppHorizontalFlip, self, result);
    return result;
}

at::Tensor vertical_flip_aclnn_kernel(const at::Tensor& self)
{
    at::Tensor result = at::empty(self.sizes(), self.options());
    EXEC_NPU_CMD(acldvppVerticalFlip, self, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_reverse_aclop"), TORCH_FN(reverse_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_horizontal_flip_aclnn"), TORCH_FN(horizontal_flip_aclnn_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_vertical_flip_aclnn"), TORCH_FN(vertical_flip_aclnn_kernel));
}

} // namespace ops
} // namespace vision
