#include "dvpp_init.h"

#ifndef MOBILE
#include <Python.h>
#endif
#include <torch/library.h>

#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include "ops/npu/op_api_common.hpp"

namespace vision {
void dvpp_init()
{
    static const auto getFuncAddr = GetOpApiFuncAddr("acldvppInit");
    TORCH_CHECK(getFuncAddr != nullptr,  "Func[acldvppInit] is not found.");
    auto func = reinterpret_cast<DvppInitFunc>(getFuncAddr);
    auto status = func(nullptr);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
    m.def("_dvpp_init", &dvpp_init);
}
} // namespace vision
