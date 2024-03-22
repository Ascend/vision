#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;

at::Tensor& gaussian_blur_aclop_kernel_impl(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    c10::optional<c10::ArrayRef<double>> sigma,
    std::string padding_mode,
    at::Tensor& result)
{
    c10::SmallVector<float, SIZE> sigmas = {
        static_cast<float>(sigma.value()[0]),
        static_cast<float>(sigma.value()[1])};

    at_npu::native::OpCommand cmd;
    cmd.Name("GaussianBlur")
        .Input(self)
        .Output(result)
        .Attr("kernel_size", kernel_size)
        .Attr("sigma", sigmas)
        .Attr("padding_mode", padding_mode)
        .Run();

    return result;
}

at::Tensor gaussian_blur_aclop_kernel(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    c10::optional<c10::ArrayRef<double>> sigma,
    std::string padding_mode)
{
    TORCH_CHECK(sigma.has_value(),
        "Op[gaussian_blur_aclop] argument[sigma] is mandatory");

    at::Tensor result = at::empty(self.sizes(), self.options());

    gaussian_blur_aclop_kernel_impl(
        self,
        kernel_size, sigma,
        padding_mode,
        result);

    return result;
}

at::Tensor gaussian_blur_aclnn_kernel(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    c10::optional<c10::ArrayRef<double>> sigma,
    int64_t padding_mode)
{
    TORCH_CHECK(sigma.has_value() && (sigma.value().size() == 2),
                "Param[sigma] is required and size=2.");

    std::vector<float> s_vec = array_to_vector_cast<float, double>(sigma.value());
    at::ArrayRef<float> sigma_cast(s_vec);

    at::Tensor result = at::empty(self.sizes(), self.options());
    EXEC_NPU_CMD(acldvppGaussianBlur, self, kernel_size, sigma_cast, padding_mode, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_gaussian_blur_aclop"), TORCH_FN(gaussian_blur_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_gaussian_blur_aclnn"), TORCH_FN(gaussian_blur_aclnn_kernel));
}

} // namespace ops
} // namespace vision
