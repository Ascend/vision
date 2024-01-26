#include "pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

at::Tensor &npu_normalize_kernel_impl(
    const at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    at::ScalarType dtype,
    at::Tensor &result) {
  TORCH_CHECK(mean.has_value() && variance.has_value(),
      "Op[npu_normalize] arguments[mean]&[variance] are mandatory");
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
      "Op[npu_normalize] output dtype can only be float16 or float32");

  int64_t typeEnum = dtype == at::kHalf ? 1 : 0;
  std::vector<int64_t> para_shape = {1, 3, 1, 1};

  at_npu::native::OpCommand cmd;
  cmd.Name("NormalizeV2")
      .Input(self)
      .Input(mean.value(), para_shape, at::kFloat)
      .Input(variance.value(), para_shape, at::kFloat)
      .Output(result)
      .Attr("dtype", typeEnum)
      .Run();

  return result;
}

at::Tensor npu_normalize_kernel(
    const at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    at::ScalarType dtype) {
  auto output_size = self.sizes();

  at::Tensor result = at::empty(output_size, self.options().dtype(dtype));

  npu_normalize_kernel_impl(self, mean, variance, dtype, result);

  return result;
}

at::Tensor& npu_normalize_inplace_kernel(
    at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    at::ScalarType dtype) {
  //The PytorchAdapter hides check_match interface. Temporarily replace with clone and copy_.
  auto self_clone = self.clone();
  self.copy_(self_clone);
  npu_normalize_kernel_impl(self, mean, variance, dtype, self);
  return self;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::npu_normalize"), TORCH_FN(npu_normalize_kernel));
  m.impl(TORCH_SELECTIVE_NAME("torchvision::npu_normalize_"), TORCH_FN(npu_normalize_inplace_kernel));
}

} // namespace ops
} // namespace vision
