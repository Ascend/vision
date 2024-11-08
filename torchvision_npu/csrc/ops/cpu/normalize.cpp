#include "normalize.h"
 
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>
 
namespace vision {
namespace ops {
 
at::Tensor normalize_moal(at::Tensor tensor, const std::vector<double>& mean, const std::vector<double>& std, bool inplace /* = false */)
{
    C10_LOG_API_USAGE_ONCE("torchvision_npu.csrc.ops.normalize_moal.normalize_moal");
    static auto op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow("torchvision::normalize_moal", "")
                        .typed<decltype(normalize_moal)>();
    return op.call(tensor, mean, std, inplace);
}
 
TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def(TORCH_SELECTIVE_SCHEMA(
        "torchvision::normalize_moal(Tensor tensor, float[] mean, float[] std, bool inplace=False) -> Tensor"));
}
 
} // namespace ops
} // namespace vision
