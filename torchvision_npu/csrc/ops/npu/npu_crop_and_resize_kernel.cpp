#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;

at::Tensor& crop_and_resize_aclop_kernel_impl(
    const at::Tensor& self,
    c10::optional<c10::ArrayRef<double>> boxes,
    at::IntArrayRef box_index,
    at::IntArrayRef crop_size,
    double extrapolation_value,
    std::string method,
    at::Tensor& result)
{
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

at::Tensor crop_and_resize_aclop_kernel(
    const at::Tensor& self,
    c10::optional<c10::ArrayRef<double>> boxes,
    at::IntArrayRef box_index,
    at::IntArrayRef crop_size,
    double extrapolation_value,
    std::string method)
{
    TORCH_CHECK(boxes.has_value(),
        "Op[crop_and_resize_aclop] argument[boxes] is mandatory");
    TORCH_CHECK(crop_size.size() == 2,
        "Op[crop_and_resize_aclop] argument[crop_size] should have 2 elements: (height, width).");

    c10::SmallVector<int64_t, SIZE> output_size = {
        static_cast<int64_t>(box_index.size()), self.size(1), crop_size[0], crop_size[1]};

    at::Tensor result = at::empty(output_size, self.options());
    crop_and_resize_aclop_kernel_impl(
        self,
        boxes, box_index, crop_size,
        extrapolation_value, method,
        result);

    return result;
}

at::Tensor crop_and_resize_aclnn_kernel(
    const at::Tensor& self,
    int64_t top, int64_t left, int64_t height, int64_t width,
    at::IntArrayRef size,
    int64_t interpolation_mode)
{
    c10::SmallVector<int64_t, SIZE> output_size = {1, self.size(1), size[0], size[1]};
    at::Tensor result = at::empty(output_size, self.options());
    EXEC_NPU_CMD(acldvppCropAndResize, self, top, left, height, width, size, interpolation_mode, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_crop_and_resize_aclop"), TORCH_FN(crop_and_resize_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_crop_and_resize_aclnn"), TORCH_FN(crop_and_resize_aclnn_kernel));
}

} // namespace ops
} // namespace vision
