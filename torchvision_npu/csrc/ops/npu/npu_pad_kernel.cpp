#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;

at::Tensor& pad_aclop_kernel_impl(
    const at::Tensor& self,
    at::IntArrayRef pad,
    int64_t constant_values,
    std::string mode,
    at::Tensor& result)
{
    std::vector<int64_t> pad1 = {0, 0};
    std::vector<int64_t> pad2 = {pad[2], pad[3]};
    std::vector<int64_t> pad3 = {pad[0], pad[1]};
    c10::IntArrayRef p0(pad1);
    c10::IntArrayRef p1(pad1);
    c10::IntArrayRef p2(pad2);
    c10::IntArrayRef p3(pad3);
    c10::SmallVector<c10::IntArrayRef, 32> p = {p0, p1, p2, p3};
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

at::Tensor pad_aclop_kernel(
    const at::Tensor& self,
    at::IntArrayRef pad,
    int64_t constant_values,
    std::string mode)
{
    TORCH_CHECK(pad.size() == 4,
        "Op[pad_aclop] argument[pad] should have 4 elements: (pad_left, pad_right, pad_top, pad_bottom).");
    int64_t h = self.size(2) + pad[2] + pad[3];
    int64_t w = self.size(3) + pad[0] + pad[1];
    TORCH_CHECK(h > 0 && w > 0,
        "Op[pad_aclop] outputsize h, w should be greater than 0");
    c10::SmallVector<int64_t, SIZE> output_size = {1, self.size(1), h, w};
    at::Tensor result = at::empty(output_size, self.options());

    pad_aclop_kernel_impl(
        self,
        pad,
        constant_values, mode,
        result);

    return result;
}

at::Tensor pad_aclnn_kernel(
    const at::Tensor& self,
    at::IntArrayRef padding,
    int64_t padding_mode,
    c10::optional<c10::ArrayRef<double>> fill)
{
    TORCH_CHECK(padding.size() == 4,
        "Op[acldvppPad] argument[padding] should have 4 elements: (pad_left, pad_top, pad_right, pad_bottom).");
    std::vector<float> f_vec;
    if (fill.has_value()) {
        TORCH_CHECK(fill.value().size() == 3, "Param[fill] size should be 3.");
        f_vec = array_to_vector_cast<float, double>(fill.value());
    } else {
        f_vec = {0, 0, 0};
    }
    at::ArrayRef<float> fill_cast(f_vec);

    int64_t h = self.size(2) + padding[1] + padding[3];
    int64_t w = self.size(3) + padding[0] + padding[2];
    TORCH_CHECK(h > 0 && w > 0,
        "Op[pad_aclnn] outputsize h, w should be greater than 0");
    c10::SmallVector<int64_t, SIZE> output_size = {1, self.size(1), h, w};
    at::Tensor result = at::empty(output_size, self.options());

    EXEC_NPU_CMD(acldvppPad, self, padding, padding_mode, fill_cast, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_pad_aclop"), TORCH_FN(pad_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_pad_aclnn"), TORCH_FN(pad_aclnn_kernel));
}

} // namespace ops
} // namespace vision
