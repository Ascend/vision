#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;

constexpr int64_t ratio = 1;
constexpr bool fancy_upscaling = true;
constexpr float acceptable_fraction = 1.0;
const std::string dct_method = "";
const std::string dst_img_format = "CHW";

c10::SmallVector<int64_t, SIZE> decode_jpeg_output_size(
    at::IntArrayRef image_shape,
    int64_t channels)
{
    TORCH_CHECK(image_shape.size() == 3,
        "Op[decode_jpeg_aclop] argument[image_shape] should have 3 elements: (height, width, channels).");

    int64_t h = image_shape[0];
    int64_t w = image_shape[1];
    int64_t c = image_shape[2];

    c10::SmallVector<int64_t, SIZE> output_size;
    if (channels == 0) {
        output_size = {1, c, h, w};
    } else {
        output_size = {1, channels, h, w};
    }
    
    return output_size;
}

at::Tensor decode_jpeg_aclop_kernel(const at::Tensor& self,
                                    at::IntArrayRef image_shape,
                                    int64_t channels,
                                    bool try_recover_truncated)
{
    auto output_size = decode_jpeg_output_size(image_shape, channels);
    output_size = {output_size[1], output_size[2], output_size[3]};
    at::Tensor result = at::empty(output_size, self.options().dtype(at::kByte));

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

at::Tensor decode_jpeg_aclnn_kernel(const at::Tensor& self,
                                    at::IntArrayRef image_shape,
                                    int64_t channels,
                                    bool try_recover_truncated)
{
    auto output_size = decode_jpeg_output_size(image_shape, channels);
    at::Tensor result = at::empty(output_size, self.options().dtype(at::kByte));
    EXEC_NPU_CMD(acldvppDecodeJpeg, self, channels, try_recover_truncated, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_decode_jpeg_aclop"), TORCH_FN(decode_jpeg_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_decode_jpeg_aclnn"), TORCH_FN(decode_jpeg_aclnn_kernel));
}

} // namespace ops
} // namespace vision
