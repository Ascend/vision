#include "pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

at::Tensor &npu_crop_kernel_impl(
    const at::Tensor &self,
    int64_t axis,
    at::IntArrayRef offsets,
    at::Tensor &result) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Crop")
      .Input(self)
      .Input(result)
      .Output(result)
      .Attr("axis", axis)
      .Attr("offsets", offsets)
      .Run();

  return result;
}

at::Tensor npu_crop_kernel(
    const at::Tensor &self,
    at::IntArrayRef size,
    at::IntArrayRef offsets,
    int64_t axis) {
  TORCH_CHECK(size.size() == self.sizes().size(),
      "Op[npu_crop] argument[size] represents output shape, (N, C, H, W).");
  
  at::Tensor result = at_npu::native::OpPreparation::ApplyTensor(self, size);

  npu_crop_kernel_impl(self, axis, offsets, result);

  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::npu_crop"), TORCH_FN(npu_crop_kernel));
}

} // namespace ops
} // namespace vision
