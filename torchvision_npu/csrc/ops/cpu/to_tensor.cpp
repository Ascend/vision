// to_tensor.cpp
 
#include "to_tensor.h"
 
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>
 
namespace vision {
namespace ops {
 
at::Tensor to_tensor_moal(const at::Tensor &clip)
{
    C10_LOG_API_USAGE_ONCE("torchvision_npu.csrc.ops.to_tensor_moal.to_tensor_moal");
    static auto op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow("torchvision::to_tensor_moal", "")
                        .typed<decltype(to_tensor_moal)>();
    return op.call(clip);
}
 
TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::to_tensor_moal(Tensor clip) -> Tensor"));
}
 
} // namespace ops
} // namespace vision
