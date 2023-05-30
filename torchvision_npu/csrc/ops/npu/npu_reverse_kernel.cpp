#include "pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

at::Tensor &npu_reverse_kernel_impl(
    const at::Tensor &self,
    at::IntArrayRef axis,
    at::Tensor &result) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ReverseV2")
      .Input(self)
      .Input(axis, at::kInt)
      .Output(result)
      .Run();

  return result;
}

at::Tensor npu_reverse_kernel(
    const at::Tensor &self,
    at::IntArrayRef axis) {
  at::Tensor result = at_npu::native::OpPreparation::ApplyTensor(self);

  npu_reverse_kernel_impl(self, axis, result);

  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::npu_reverse"), TORCH_FN(npu_reverse_kernel));
}

} // namespace ops
} // namespace vision
