#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

at::Tensor warp_perspective_aclnn_kernel(
    const at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> matrix,
    int64_t interpolation_mode,
    int64_t padding_mode,
    c10::optional<c10::ArrayRef<double>> fill)
{
    TORCH_CHECK(matrix.has_value() && (matrix.value().size() == 9), "Param[matrix] is required and size=9.");

    std::vector<float> m_vec = array_to_vector_cast<float, double>(matrix.value());
    at::ArrayRef<float> matrix_cast(m_vec);

    std::vector<float> f_vec;
    if (fill.has_value()) {
        TORCH_CHECK(fill.value().size() == 3, "Param[fill] size should be 3.");
        f_vec = array_to_vector_cast<float, double>(fill.value());
    } else {
        f_vec = {0, 0, 0};
    }
    at::ArrayRef<float> fill_cast(f_vec);

    at::Tensor result = at::empty(self.sizes(), self.options());
    EXEC_NPU_CMD(acldvppWarpPerspective, self, matrix_cast, interpolation_mode, padding_mode, fill_cast, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_warp_perspective_aclnn"), TORCH_FN(warp_perspective_aclnn_kernel));
}

} // namespace ops
} // namespace vision
