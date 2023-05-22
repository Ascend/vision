#include "pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

at::Tensor &npu_resize_kernel_impl(
    const at::Tensor &self,
    at::IntArrayRef size,
    double cubic_coeff_a,
    int64_t exclude_outside,
    std::string mode,
    std::string nearest_mode,
    at::Tensor &result) {
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

at::Tensor npu_resize_kernel(
    const at::Tensor &self,
    at::IntArrayRef size,
    double cubic_coeff_a,
    int64_t exclude_outside,
    std::string mode,
    std::string nearest_mode) {
  TORCH_CHECK(size.size() == 2,
      "Op[npu_resize] argument[size] should have 2 elements: (height, width).");

  c10::SmallVector<int64_t, at_npu::native::N> output_size = {self.size(0), self.size(1), size[0], size[1]};
  
  at::Tensor result = at_npu::native::OpPreparation::ApplyTensor(self, output_size);

  npu_resize_kernel_impl(
      self, size,
      cubic_coeff_a, exclude_outside, mode, nearest_mode,
      result);

  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::npu_resize"), TORCH_FN(npu_resize_kernel));
}

} // namespace ops
} // namespace vision
