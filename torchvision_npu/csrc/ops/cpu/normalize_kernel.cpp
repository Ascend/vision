#include <ATen/ATen.h>
#include <torch/library.h>
#include "common_include.h"
#include "waiter.h"
 
namespace vision {
namespace ops {
 
at::Tensor normalize_kernel(at::Tensor tensor, const std::vector<double>& mean, const std::vector<double>& std, bool inplace = false)
{
    TORCH_CHECK(tensor.is_floating_point(), "Input tensor should be a float tensor.")
    size_t image_size = 3;
    TORCH_CHECK(tensor.ndimension() > image_size, "Expected tensor to be a tensor image of size (..., C, H, W).")

    at::Tensor ret;
    if (inplace) {
        ret = tensor;
    } else {
        ret = torch::empty_like(tensor);
    }
 
    auto sizes = tensor.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int height = sizes[2];
    int width = sizes[3];
 
    std::vector<float> mean_tensor(mean.begin(), mean.end());
    std::vector<float> inv_std_tensor(std.size());
    TORCH_CHECK(mean.size() >= channels && std.size() >= channels,
                "mean size and std size should be greater than to channels")
    for (size_t i = 0; i < std.size(); i++) {
        TORCH_CHECK(std[i] != 0, "std evaluated to zero, leading to division by zero.")
        inv_std_tensor[i] = 1.0 / std[i];
    }
 
    auto tensor_ptr = tensor.data_ptr<float>();
    auto ret_ptr = ret.data_ptr<float>();
 
    auto threadNum = at::get_num_threads();
#pragma omp parallel for num_threads(threadNum)
    for (uint64_t idx = 0; idx < batch_size * channels; idx ++) {
        int c = idx % channels;
        uint64_t offset = static_cast<uint64_t>(idx) * height * width;
        for (uint64_t j = 0; j < height * width; j ++) {
            ret_ptr[offset + j] = (tensor_ptr[offset + j] - mean_tensor[c]) * inv_std_tensor[c];
        }
    }
    return ret;
}
 
TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
    m.impl(TORCH_SELECTIVE_NAME("torchvision::normalize_moal"), TORCH_FN(normalize_kernel));
}
 
} // namespace ops
} // namespace vision
