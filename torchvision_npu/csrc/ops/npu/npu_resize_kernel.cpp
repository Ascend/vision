#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;

at::Tensor& resize_aclop_kernel_impl(
    const at::Tensor& self,
    at::IntArrayRef size,
    double cubic_coeff_a,
    int64_t exclude_outside,
    std::string mode,
    std::string nearest_mode,
    at::Tensor& result)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Resize")
        .Input(self)
        .Input(size, at::kFloat)
        .Input(size, at::kFloat)
        .Input(result.sizes(), at::kInt)
        .Output(result)
        .Attr<std::string>("coordinate_transformation_mode", "pytorch_half_pixel")
        .Attr<float>("cubic_coeff_a", cubic_coeff_a)
        .Attr("exclude_outside", exclude_outside)
        .Attr("mode", mode)
        .Attr("nearest_mode", nearest_mode)
        .Run();

    return result;
}

at::Tensor resize_aclop_kernel(
    const at::Tensor& self,
    at::IntArrayRef size,
    double cubic_coeff_a,
    int64_t exclude_outside,
    std::string mode,
    std::string nearest_mode)
{
    TORCH_CHECK(size.size() == 2,
        "Op[resize_aclop] argument[size] should have 2 elements: (height, width).");

    c10::SmallVector<int64_t, SIZE> output_size = {self.size(0), self.size(1), size[0], size[1]};
    at::Tensor result = at::empty(output_size, self.options());

    resize_aclop_kernel_impl(
        self, size,
        cubic_coeff_a, exclude_outside, mode, nearest_mode,
        result);

    return result;
}

at::Tensor resize_aclnn_kernel(
    const at::Tensor& self,
    at::IntArrayRef size,
    int64_t interpolation_mode)
{
    c10::SmallVector<int64_t, SIZE> output_size = {1, self.size(1), size[0], size[1]};
    at::Tensor result = at::empty(output_size, self.options());
    EXEC_NPU_CMD(acldvppResize, self, interpolation_mode, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_resize_aclop"), TORCH_FN(resize_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_resize_aclnn"), TORCH_FN(resize_aclnn_kernel));
}

} // namespace ops
} // namespace vision
