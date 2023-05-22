#include "pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

at::Tensor &npu_adjust_contrast_kernel_impl(
    const at::Tensor &self,
    at::Scalar factor,
    std::string mean_mode,
    at::Tensor &result) {
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

at::Tensor npu_adjust_contrast_kernel(
    const at::Tensor &self,
    at::Scalar factor,
    std::string mean_mode) {
  at::Tensor result = at_npu::native::OpPreparation::ApplyTensor(self);

  npu_adjust_contrast_kernel_impl(self, factor, mean_mode, result);

  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::npu_adjust_contrast"), TORCH_FN(npu_adjust_contrast_kernel));
}

} // namespace ops
} // namespace vision
