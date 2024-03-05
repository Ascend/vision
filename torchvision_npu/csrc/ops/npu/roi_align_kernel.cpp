// Copyright (c) 2022, Huawei Technologies.All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pytorch_npu_helper.hpp"

#include <cfloat>
#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

template <typename T>
void roi_align_forward_kernel_impl(
    const at::Tensor& input,
    const float spatial_scale,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned,
    const at::Tensor& rois,
    at::Tensor& output)
{
    int64_t roi_end_mode = aligned ? 2 : 0;

    at_npu::native::OpCommand cmd;
    cmd.Name("ROIAlign")
        .Input(input)
        .Input(rois)
        .Output(output)
        .Attr("spatial_scale", spatial_scale)
        .Attr("pooled_height", pooled_height)
        .Attr("pooled_width", pooled_width)
        .Attr("sample_num", sampling_ratio)
        .Attr("roi_end_mode", roi_end_mode)
        .Run();
}

template <typename T>
void roi_align_backward_kernel_impl(
    const at::Tensor& grad_y,
    const float spatial_scale,
    int64_t height,
    int64_t width,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned,
    at::Tensor& grad_x,
    const at::Tensor& rois)
{
    int64_t roi_end_mode = aligned ? 1 : 0;
    auto xdiff_shape  = grad_x.sizes();

    at_npu::native::OpCommand cmd;
    cmd.Name("ROIAlignGrad")
        .Input(grad_y)
        .Input(rois)
        .Output(grad_x)
        .Attr("xdiff_shape", xdiff_shape)
        .Attr("spatial_scale", spatial_scale)
        .Attr("pooled_height", pooled_height)
        .Attr("pooled_width", pooled_width)
        .Attr("sample_num", sampling_ratio)
        .Attr("roi_end_mode", roi_end_mode)
        .Run();
}

at::Tensor roi_align_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned)
{
    TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");

    at::TensorArg input_t{input, "input", 1};
    at::TensorArg rois_t{rois, "rois", 2};

    at::CheckedFrom c = "roi_align_forward_kernel";
    at::checkAllSameType(c, {input_t, rois_t});

    auto num_rois = rois.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    at::Tensor output = at::zeros(
        {num_rois, channels, pooled_height, pooled_width}, input.options());
    if (output.numel() == 0)
        return output;

    auto input_ = input.contiguous();
    auto rois_ = rois.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "roi_align_forward_kernel", [&] {
            roi_align_forward_kernel_impl<scalar_t>(
                input_,
                spatial_scale,
                channels,
                height,
                width,
                pooled_height,
                pooled_width,
                sampling_ratio,
                aligned,
                rois_,
                output);
        });
    return output;
}

at::Tensor roi_align_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned)
{
    at::TensorArg grad_t{grad, "grad", 1};
    at::TensorArg rois_t{rois, "rois", 2};

    at::CheckedFrom c = "roi_align_backward_kernel";
    at::checkAllSameType(c, {grad_t, rois_t});

    at::Tensor grad_input =
        at::zeros({batch_size, channels, height, width}, grad.options());

    // handle possibly empty gradients
    if (grad.numel() == 0) {
        return grad_input;
    }

    auto rois_ = rois.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad.scalar_type(), "roi_align_backward_kernel", [&] {
            roi_align_backward_kernel_impl<scalar_t>(
                grad,
                spatial_scale,
                height,
                width,
                pooled_height,
                pooled_width,
                sampling_ratio,
                aligned,
                grad_input,
                rois_);
        });
    return grad_input;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
    m.impl(
        TORCH_SELECTIVE_NAME("torchvision::roi_align"),
        TORCH_FN(roi_align_forward_kernel));
    m.impl(
        TORCH_SELECTIVE_NAME("torchvision::_roi_align_backward"),
        TORCH_FN(roi_align_backward_kernel));
}

} // namespace ops
} // namespace vision
