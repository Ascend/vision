#include <ATen/ATen.h>
#include <torch/library.h>

#include "op_api_common.hpp"

namespace vision {
namespace ops {

namespace {

at::Tensor img_to_tensor_aclop_kernel(const at::Tensor& self)
{
    TORCH_CHECK(self.dtype() == at::kByte,
        "Op[img_to_tensor_aclop] input dtype should be uint8.");
    
    auto output_size = self.sizes();
    at::Tensor result = at::empty(output_size, self.options().dtype(at::kFloat));

    at_npu::native::OpCommand cmd;
    cmd.Name("ImgToTensor")
        .Input(self)
        .Output(result)
        .Run();

    return result;
}

at::Tensor img_to_tensor_aclnn_kernel(const at::Tensor& self)
{
    at::Tensor result = at::empty(self.sizes(), self.options().dtype(at::kFloat));
    EXEC_NPU_CMD(acldvppImgToTensor, self, result);
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_img_to_tensor_aclop"), TORCH_FN(img_to_tensor_aclop_kernel));
    m.impl(TORCH_SELECTIVE_NAME("torchvision::_img_to_tensor_aclnn"), TORCH_FN(img_to_tensor_aclnn_kernel));
}

} // namespace ops
} // namespace vision
