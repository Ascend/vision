#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

at::Tensor solarize_aclnn_kernel(
    const at::Tensor& self,
    c10::optional<c10::ArrayRef<double>> threshold)
{
    TORCH_CHECK(threshold.has_value() && ((threshold.value().size() == 1) || (threshold.value().size() == 2)),
                "Param[threshold] is required and size=1or2.");

    std::vector<float> t_vec = array_to_vector_cast<float, double>(threshold.value());
    at::ArrayRef<float> threshold_cast(t_vec);

    at::Tensor result = at::empty(self.sizes(), self.options());
    EXEC_NPU_CMD(acldvppSolarize, self, threshold_cast, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_solarize_aclnn"), TORCH_FN(solarize_aclnn_kernel));
}

} // namespace ops
} // namespace vision
