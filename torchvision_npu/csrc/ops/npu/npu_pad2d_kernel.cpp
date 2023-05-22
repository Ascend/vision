#include "pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

at::Tensor &npu_pad2d_kernel_impl(
    const at::Tensor &self,
    at::IntArrayRef pad,
    int64_t constant_values,
    std::string mode,
    at::Tensor &result) {
  c10::SmallVector<int64_t, at_npu::native::N> pad1 = {0, 0};
  c10::SmallVector<int64_t, at_npu::native::N> pad2 = {pad[2], pad[3]};
  c10::SmallVector<int64_t, at_npu::native::N> pad3 = {pad[0], pad[1]};
  c10::IntArrayRef p0(pad1);
  c10::IntArrayRef p1(pad1);
  c10::IntArrayRef p2(pad2);
  c10::IntArrayRef p3(pad3);
  c10::SmallVector<c10::IntArrayRef, at_npu::native::N> p = {p0, p1, p2, p3};
  at::ArrayRef<c10::IntArrayRef> paddings(p);

  at_npu::native::OpCommand cmd;
  cmd.Name("PadV3D")
      .Input(self, "x")
      .Output(result)
      .Attr("paddings", paddings)
      .Attr("constant_values", constant_values)
      .Attr("mode", mode)
      .Attr("paddings_contiguous", true)
      .Run();

  return result;
}

at::Tensor npu_pad2d_kernel(
    const at::Tensor &self,
    at::IntArrayRef pad,
    int64_t constant_values,
    std::string mode) {
  TORCH_CHECK(pad.size() == 4,
      "Op[npu_pad2d] argument[pad] should have 4 elements: (pad_left, pad_right, pad_top, pad_bottom).");

  auto output_size = at_npu::native::replication_pad2d_npu_output_size(self, pad);

  at::Tensor result = at_npu::native::OpPreparation::ApplyTensor(self, output_size);

  npu_pad2d_kernel_impl(
      self,
      pad,
      constant_values, mode,
      result);

  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::npu_pad2d"), TORCH_FN(npu_pad2d_kernel));
}

} // namespace ops
} // namespace vision
