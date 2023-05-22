#include "pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

at::Tensor &npu_adjust_saturation_kernel_impl(
    const at::Tensor &self,
    at::Scalar factor,
    at::Tensor &result) {
  at_npu::native::OpCommand cmd;
  cmd.Name("AdjustSaturationV2")
      .Input(self)
      .Input(factor, at::ScalarType::Float)
      .Output(result)
      .Attr<std::string>("data_format", "CHW")
      .Run();

  return result;
}

at::Tensor npu_adjust_saturation_kernel(
    const at::Tensor &self,
    at::Scalar factor) {
  at::Tensor result = at_npu::native::OpPreparation::ApplyTensor(self);

  npu_adjust_saturation_kernel_impl(self, factor, result);

  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::npu_adjust_saturation"), TORCH_FN(npu_adjust_saturation_kernel));
}

} // namespace ops
} // namespace vision
