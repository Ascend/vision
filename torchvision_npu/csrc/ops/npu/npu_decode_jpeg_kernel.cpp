#include "pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;

constexpr int64_t ratio = 1;
constexpr bool fancy_upscaling = true;
constexpr float acceptable_fraction = 1.0;
const std::string dct_method = "";
const std::string dst_img_format = "CHW";

c10::SmallVector<int64_t, SIZE> npu_decode_jpeg_output_size(
    at::IntArrayRef image_shape,
    int64_t channels) {
  TORCH_CHECK(image_shape.size() == 3,
      "Op[npu_decode_jpeg] argument[image_shape] should have 3 elements: (height, width, channels).");

  int64_t H = image_shape[0];
  int64_t W = image_shape[1];
  int64_t C = image_shape[2];

  c10::SmallVector<int64_t, SIZE> output_size;
  if (channels == 0) {
    output_size = {C, H, W};
  } else {
    output_size = {channels, H, W};
  }
  
  return output_size;
}

at::Tensor &npu_decode_jpeg_kernel_impl(
    const at::Tensor &self,
    int64_t channels,
    bool try_recover_truncated,
    at::Tensor &result) {
  at_npu::native::OpCommand cmd;
  cmd.Name("DecodeJpeg")
      .Input(self, "", c10::nullopt, "string")
      .Output(result)
      .Attr("channels", channels)
      .Attr("ratio", ratio)
      .Attr("fancy_upscaling", fancy_upscaling)
      .Attr("try_recover_truncated", try_recover_truncated)
      .Attr("acceptable_fraction", acceptable_fraction)
      .Attr("dct_method", dct_method)
      .Attr("dst_img_format", dst_img_format)
      .Run();

  return result;
}

at::Tensor npu_decode_jpeg_kernel(
    const at::Tensor &self,
    at::IntArrayRef image_shape,
    int64_t channels,
    bool try_recover_truncated) {
  // calculate the output size
  auto output_size = npu_decode_jpeg_output_size(image_shape, channels);

  // construct the output tensor of the NPU
  at::Tensor result = at::empty(output_size, self.options().dtype(at::kByte));

  // calculate the output result of the NPU
  npu_decode_jpeg_kernel_impl(
      self,
      channels, try_recover_truncated,
      result);

  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::npu_decode_jpeg"), TORCH_FN(npu_decode_jpeg_kernel));
}

} // namespace ops
} // namespace vision
