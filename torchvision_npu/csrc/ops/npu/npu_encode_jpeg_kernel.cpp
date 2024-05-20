#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;
constexpr uint32_t jpege_header_size = 640U;
constexpr uint32_t start_align_bytes = 128U;
constexpr uint32_t memory_align_size = 2097152U;

at::Tensor encode_jpeg_aclnn_kernel(const at::Tensor& self, int64_t quality)
{
    TORCH_CHECK(self.scalar_type() == at::kByte,
        "input datatype must be uint8, but got ", self.scalar_type());

    TORCH_CHECK((self.dim() == 4) && (self.size(0) == 1) &&
        ((self.size(1) == 1) || (self.size(1) == 3)),
        "input format must be NCHW(N=1, C=1or3), but got sizes ", self.sizes());

    TORCH_CHECK((self.size(2) % 2 == 0) && (self.size(3) % 2 == 0),
        "input height and width must be 2 aligned, but got height ",
        self.size(2), ", width ", self.size(3));

    at::Tensor self_contiguous = self.contiguous();

    uint32_t encode_size = ALIGN_UP(self.size(3), 16U) * ALIGN_UP(self.size(2), 16U) * 3 / 2 +
        jpege_header_size + start_align_bytes;
    encode_size = ALIGN_UP(encode_size, memory_align_size);
    c10::SmallVector<int64_t, SIZE> output_size = {encode_size};
    at::Tensor result = at::empty(output_size, self_contiguous.options());
    EXEC_NPU_CMD(acldvppEncodeJpeg, self_contiguous, quality, result);

    int64_t* view_dims = nullptr;
    uint64_t view_dims_num = 0;
    static const auto aclGetViewShape = GET_OP_API_FUNC(aclGetViewShape);
    TORCH_CHECK(aclGetViewShape != nullptr,  "Func[aclGetViewShape] is not found.");
    auto ret = aclGetViewShape(ConvertType(result), &view_dims, &view_dims_num);
    TORCH_CHECK((ret == 0) && (view_dims_num == 1),
        "aclGetViewShape failed, ret=", ret, ", viewDimsNum =", view_dims_num);

    std::vector<int64_t> out_size(view_dims, view_dims + view_dims_num);
    c10::SmallVector<int64_t, SIZE> view_size = {out_size[0]};

    return result.resize_(view_size);
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_encode_jpeg_aclnn"), TORCH_FN(encode_jpeg_aclnn_kernel));
}

} // namespace ops
} // namespace vision
