#include "pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;
const int N = 32;

c10::SmallVector<int64_t, SIZE> replication_pad2d_npu_output_size(const at::Tensor &self, c10::IntArrayRef padding)
{
    TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "tensor self's dimension must be 3 or 4");
    int64_t N = self.dim() == 3 ? 1 : self.size(-4);
    int64_t C = self.size(-3);
    int64_t H = self.size(-2);
    int64_t W = self.size(-1);
    int64_t padding_l = 0;
    int64_t padding_r = 0;
    int64_t padding_t = 0;
    int64_t padding_b = 0;
    if (!padding.empty() && padding.size() == 1) {
        padding_l = padding[0];
        padding_r = padding[0];
        padding_t = padding[0];
        padding_b = padding[0];
    } else if (!padding.empty() && 4 == padding.size()) {
        padding_l = padding[0];
        padding_r = padding[1];
        padding_t = padding[2];
        padding_b = padding[3];
    }
    int64_t Ho = H + padding_t + padding_b;
    int64_t Wo = W + padding_l + padding_r;

    c10::SmallVector<int64_t, SIZE> outputSize = {N, C, Ho, Wo};
    return outputSize;
}


at::Tensor &npu_pad2d_kernel_impl(
    const at::Tensor &self,
    at::IntArrayRef pad,
    int64_t constant_values,
    std::string mode,
    at::Tensor &result) {
  c10::SmallVector<int64_t, N> pad1 = {0, 0};
  c10::SmallVector<int64_t, N> pad2 = {pad[2], pad[3]};
  c10::SmallVector<int64_t, N> pad3 = {pad[0], pad[1]};
  c10::IntArrayRef p0(pad1);
  c10::IntArrayRef p1(pad1);
  c10::IntArrayRef p2(pad2);
  c10::IntArrayRef p3(pad3);
  c10::SmallVector<c10::IntArrayRef, N> p = {p0, p1, p2, p3};
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

  auto output_size = replication_pad2d_npu_output_size(self, pad);

  at::Tensor result = at::empty(output_size, self.options());

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
