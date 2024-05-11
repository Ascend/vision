#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

at::Tensor invert_aclnn_kernel(const at::Tensor& self)
{
    at::Tensor result = at::empty(self.sizes(), self.options());
    EXEC_NPU_CMD(acldvppInvert, self, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_invert_aclnn"), TORCH_FN(invert_aclnn_kernel));
}

} // namespace ops
} // namespace vision
