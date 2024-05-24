#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

at::Tensor normalize_aclop_kernel(const at::Tensor& self,
                                  c10::optional<c10::ArrayRef<double>> mean,
                                  c10::optional<c10::ArrayRef<double>> variance,
                                  at::ScalarType dtype)
{
    TORCH_CHECK(mean.has_value() && variance.has_value(), "Param[mean] and [variance] are required.");
    TORCH_CHECK((dtype == at::kHalf) || (dtype == at::kFloat), "Param[dtype] should be float16 or float32");

    at::Tensor result = at::empty(self.sizes(), self.options().dtype(dtype));
    int64_t type_enum = (dtype == at::kHalf) ? 1 : 0;
    std::vector<int64_t> param_shape = {1, 3, 1, 1};

    at_npu::native::OpCommand cmd;
    cmd.Name("NormalizeV2")
        .Input(self)
        .Input(mean.value(), param_shape, at::kFloat)
        .Input(variance.value(), param_shape, at::kFloat)
        .Output(result)
        .Attr("dtype", type_enum)
        .Run();

    return result;
}

at::Tensor normalize_aclnn_kernel(const at::Tensor& self,
                                  c10::optional<c10::ArrayRef<double>> mean,
                                  c10::optional<c10::ArrayRef<double>> std)
{
    TORCH_CHECK(mean.has_value() && std.has_value(), "Param[mean] and [std] are required.");

    std::vector<float> m_vec = array_to_vector_cast<float, double>(mean.value());
    at::ArrayRef<float> mean_cast(m_vec);
    std::vector<float> s_vec = array_to_vector_cast<float, double>(std.value());
    at::ArrayRef<float> std_cast(s_vec);

    auto self_ = self.contiguous();
    at::Tensor result = at::empty(self.sizes(), self.options().dtype(at::kFloat));
    EXEC_NPU_CMD(acldvppNormalize, self_, mean_cast, std_cast, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_normalize_aclop"), TORCH_FN(normalize_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_normalize_aclnn"), TORCH_FN(normalize_aclnn_kernel));
}

} // namespace ops
} // namespace vision
