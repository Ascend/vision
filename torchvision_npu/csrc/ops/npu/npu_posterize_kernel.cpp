#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

at::Tensor posterize_aclnn_kernel(const at::Tensor& self, int64_t bits)
{
    at::Tensor result = at::empty(self.sizes(), self.options());
    EXEC_NPU_CMD(acldvppPosterize, self, bits, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_posterize_aclnn"), TORCH_FN(posterize_aclnn_kernel));
}

} // namespace ops
} // namespace vision
