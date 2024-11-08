#include <ATen/ATen.h>
#include <torch/library.h>
#include "common_include.h"
#include "waiter.h"
 
namespace vision {
namespace ops {
 
at::Tensor to_tensor_kernel(const at::Tensor &clip)
{
    TORCH_CHECK(clip.dtype() == at::kByte, "input type error");
    auto sizes = clip.sizes();
    if (!clip.is_contiguous() && sizes[1] == 3) {
        auto opt = clip.options().dtype(at::kFloat);
        at::Tensor result = torch::empty(sizes, opt);
 
        const uint8_t* clipPtr = clip.data_ptr<uint8_t>();
        float* resultPtr = result.data_ptr<float>();
        uint64_t batchsize = sizes[0];
        double mul = 0.0039215686274509803921568627451;
 
        auto threadNum = at::get_num_threads();
        TORCH_CHECK(threadNum != 0, "Thread Number cannot be 0")
        auto grain = batchsize / threadNum;
 
        int h = sizes[2];
        int w = sizes[3];
 
        if (threadNum == 1) {
            for (uint64_t i = 0; i < batchsize; ++i) {
                for (uint64_t j = 0; j < h * w; ++j) {
                    resultPtr[i * 3 * h * w + 0 * h * w + j] = clipPtr[i * 3 * h * w + 3 * j + 0] * mul;
                    resultPtr[i * 3 * h * w + 1 * h * w + j] = clipPtr[i * 3 * h * w + 3 * j + 1] * mul;
                    resultPtr[i * 3 * h * w + 2 * h * w + j] = clipPtr[i * 3 * h * w + 3 * j + 2] * mul;
                }
            }
        } else {
            Waiter waiter(threadNum);
 
            for (uint64_t t = 0; t < threadNum; ++t) {
                uint64_t startIdx = t * grain;
                uint64_t endIdx = (t == threadNum - 1) ? batchsize : (startIdx + grain);
 
                std::thread([&, startIdx, endIdx]() {
                    for (uint64_t i = startIdx; i < endIdx; ++i) {
                        for (uint64_t j = 0; j < h * w; ++j) {
                            resultPtr[i * 3 * h * w + 0 * h * w + j] = clipPtr[i * 3 * h * w + 3 * j + 0] * mul;
                            resultPtr[i * 3 * h * w + 1 * h * w + j] = clipPtr[i * 3 * h * w + 3 * j + 1] * mul;
                            resultPtr[i * 3 * h * w + 2 * h * w + j] = clipPtr[i * 3 * h * w + 3 * j + 2] * mul;
                        }
                    }
                    waiter.FinishedOne();
                }).detach();
            }
 
            waiter.WaitAll();
        }
 
        return result;
    } else {
        at::Tensor clipContig = clip.contiguous();
        auto opt = clipContig.options().dtype(at::kFloat);
        at::Tensor result = torch::empty(clipContig.sizes(), opt);
 
        const uint8_t* clipPtr = clipContig.data_ptr<uint8_t>();
        auto elmNum = result.numel();
        float* resultPtr = result.data_ptr<float>();
 
        double mul = 0.0039215686274509803921568627451;
        auto threadNum = at::get_num_threads();
        TORCH_CHECK(threadNum != 0, "Thread Number cannot be 0")
        auto grain = elmNum / threadNum;
 
        if (threadNum == 1) {
            for (uint64_t i = 0; i < elmNum; i++) {
                resultPtr[i] = clipPtr[i] * mul;
            }
        } else {
            Waiter waiter(threadNum);
 
            for (uint64_t i = 0; i < threadNum; i++) {
                uint64_t startIdx = i * grain;
                uint64_t endIdx = (i == threadNum - 1) ? elmNum : (startIdx + grain);
 
                std::thread([&, startIdx, endIdx]() {
                    for (uint64_t i = startIdx; i < endIdx; i++) {
                        resultPtr[i] = clipPtr[i] * mul;
                    }
                    waiter.FinishedOne();
                }).detach();
            }
 
            waiter.WaitAll();
        }
 
        return result;
    }
}
 
TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::to_tensor_moal"), TORCH_FN(to_tensor_kernel));
}
 
} // namespace ops
} // namespace vision
