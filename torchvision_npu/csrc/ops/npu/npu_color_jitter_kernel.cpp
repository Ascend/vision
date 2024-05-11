#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

at::Tensor adjust_brightness_aclop_kernel(const at::Tensor& self, at::Scalar factor)
{
    at::Tensor result = at::empty(self.sizes(), self.options());

    at_npu::native::OpCommand cmd;
    cmd.Name("AdjustBrightnessV2")
        .Input(self)
        .Input(factor, at::ScalarType::Float)
        .Output(result)
        .Run();

    return result;
}

at::Tensor adjust_brightness_aclnn_kernel(const at::Tensor& self, double factor)
{
    at::Tensor result = at::empty(self.sizes(), self.options());
    float factor_cast = static_cast<float>(factor);
    EXEC_NPU_CMD(acldvppAdjustBrightness, self, factor_cast, result);
    return result;
}

at::Tensor adjust_contrast_aclop_kernel(const at::Tensor& self, at::Scalar factor, std::string mean_mode)
{
    at::Tensor result = at::empty(self.sizes(), self.options());

    at_npu::native::OpCommand cmd;
    cmd.Name("AdjustContrast")
        .Input(self)
        .Input(factor, at::ScalarType::Float)
        .Output(result)
        .Attr<std::string>("data_format", "CHW")
        .Attr("mean_mode", mean_mode)
        .Run();

    return result;
}

at::Tensor adjust_contrast_aclnn_kernel(const at::Tensor& self, double factor)
{
    at::Tensor result = at::empty(self.sizes(), self.options());
    float factor_cast = static_cast<float>(factor);
    EXEC_NPU_CMD(acldvppAdjustContrast, self, factor_cast, result);
    return result;
}

at::Tensor adjust_hue_aclop_kernel(const at::Tensor& self, at::Scalar factor)
{
    at::Tensor result = at::empty(self.sizes(), self.options());

    at_npu::native::OpCommand cmd;
    cmd.Name("AdjustHue")
        .Input(self)
        .Input(factor, at::ScalarType::Float)
        .Output(result)
        .Attr<std::string>("data_format", "CHW")
        .Run();

    return result;
}

at::Tensor adjust_hue_aclnn_kernel(const at::Tensor& self, double factor)
{
    at::Tensor result = at::empty(self.sizes(), self.options());
    float factor_cast = static_cast<float>(factor);
    EXEC_NPU_CMD(acldvppAdjustHue, self, factor_cast, result);
    return result;
}

at::Tensor adjust_saturation_aclop_kernel(const at::Tensor& self, at::Scalar factor)
{
    at::Tensor result = at::empty(self.sizes(), self.options());

    at_npu::native::OpCommand cmd;
    cmd.Name("AdjustSaturationV2")
        .Input(self)
        .Input(factor, at::ScalarType::Float)
        .Output(result)
        .Attr<std::string>("data_format", "CHW")
        .Run();

    return result;
}

at::Tensor adjust_saturation_aclnn_kernel(const at::Tensor& self, double factor)
{
    at::Tensor result = at::empty(self.sizes(), self.options());
    float factor_cast = static_cast<float>(factor);
    EXEC_NPU_CMD(acldvppAdjustSaturation, self, factor_cast, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_adjust_brightness_aclop"), TORCH_FN(adjust_brightness_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_adjust_brightness_aclnn"), TORCH_FN(adjust_brightness_aclnn_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_adjust_contrast_aclop"), TORCH_FN(adjust_contrast_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_adjust_contrast_aclnn"), TORCH_FN(adjust_contrast_aclnn_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_adjust_hue_aclop"), TORCH_FN(adjust_hue_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_adjust_hue_aclnn"), TORCH_FN(adjust_hue_aclnn_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_adjust_saturation_aclop"), TORCH_FN(adjust_saturation_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_adjust_saturation_aclnn"), TORCH_FN(adjust_saturation_aclnn_kernel));
}

} // namespace ops
} // namespace vision
