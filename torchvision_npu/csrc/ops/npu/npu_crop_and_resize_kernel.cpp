#include "pytorch_npu_helper.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

at::Tensor &npu_crop_and_resize_kernel_impl(
    const at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> boxes,
    at::IntArrayRef box_index,
    at::IntArrayRef crop_size,
    double extrapolation_value,
    std::string method,
    at::Tensor &result) {
  std::vector<int64_t> boxes_shape = {static_cast<int64_t>(boxes->size()) / 4, 4};
  at_npu::native::OpCommand cmd;
  cmd.Name("CropAndResizeV2")
      .Input(self)
      .Input(boxes.value(), boxes_shape, at::kFloat)
      .Input(box_index, at::kInt)
      .Input(crop_size, at::kInt)
      .Output(result)
      .Attr<float>("extrapolation_value", extrapolation_value)
      .Attr("method", method)
      .Attr("dtype", result.scalar_type())
      .Run();

  return result;
}

at::Tensor npu_crop_and_resize_kernel(
    const at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> boxes,
    at::IntArrayRef box_index,
    at::IntArrayRef crop_size,
    double extrapolation_value,
    std::string method) {
  TORCH_CHECK(boxes.has_value(),
      "Op[npu_crop_and_resize] argument[boxes] is mandatory");
  TORCH_CHECK(crop_size.size() == 2,
      "Op[npu_crop_and_resize] argument[crop_size] should have 2 elements: (height, width).");

  c10::SmallVector<int64_t, at_npu::native::SIZE> output_size = {
      static_cast<int64_t>(box_index.size()), self.size(1), crop_size[0], crop_size[1]};

  at::Tensor result = at_npu::native::OpPreparation::ApplyTensor(self, output_size);

  npu_crop_and_resize_kernel_impl(
      self,
      boxes, box_index, crop_size,
      extrapolation_value, method,
      result);

  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::npu_crop_and_resize"), TORCH_FN(npu_crop_and_resize_kernel));
}

} // namespace ops
} // namespace vision
