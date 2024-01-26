#include "pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;

at::Tensor &npu_gaussian_blur_kernel_impl(
    const at::Tensor &self,
    at::IntArrayRef kernel_size,
    c10::optional<c10::ArrayRef<double>> sigma,
    std::string padding_mode,
    at::Tensor &result) {
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

at::Tensor npu_gaussian_blur_kernel(
    const at::Tensor &self,
    at::IntArrayRef kernel_size,
    c10::optional<c10::ArrayRef<double>> sigma,
    std::string padding_mode) {
  TORCH_CHECK(sigma.has_value(),
      "Op[npu_gaussian_blur] argument[sigma] is mandatory");

  at::Tensor result = at::empty(self.sizes(), self.options());

  npu_gaussian_blur_kernel_impl(
      self,
      kernel_size, sigma,
      padding_mode,
      result);

  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::npu_gaussian_blur"), TORCH_FN(npu_gaussian_blur_kernel));
}

} // namespace ops
} // namespace vision
