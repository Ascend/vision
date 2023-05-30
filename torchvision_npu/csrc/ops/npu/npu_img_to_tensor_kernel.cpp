#include "pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

at::Tensor &npu_img_to_tensor_kernel_impl(
    const at::Tensor &self,
    at::Tensor &result) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ImgToTensor")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}

at::Tensor npu_img_to_tensor_kernel(
    const at::Tensor &self) {
  TORCH_CHECK(self.dtype() == at::kByte,
      "Op[npu_img_to_tensor] input dtype should be uint8.");
  
  auto output_size = at_npu::native::input_same_output_size(self);

  at::Tensor result = at_npu::native::OpPreparation::ApplyTensorWithFormat(
      output_size,
      self.options().dtype(at::kFloat),
      at_npu::native::CalcuOpUtil::GetTensorNpuFormat(self));

  npu_img_to_tensor_kernel_impl(self, result);

  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::npu_img_to_tensor"), TORCH_FN(npu_img_to_tensor_kernel));
}

} // namespace ops
} // namespace vision
