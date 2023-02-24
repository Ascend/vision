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

#include <float.h>
#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

template <typename T>
void roi_pool_forward_kernel_impl(
    const at::Tensor& input,
    const float spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    const at::Tensor& rois,
    int num_rois,
    at::Tensor& output,
    at::Tensor& argmax) {
  int64_t pooled_height_64 = pooled_height;
  int64_t pooled_width_64 = pooled_width;
  int64_t pooled_channel = 1;
  at::Tensor roi_actual_num = at_npu::native::OpPreparation::ApplyTensor(
      {}, rois.options().dtype(at::kInt), rois);

  at_npu::native::OpCommand cmd;
  cmd.Name("RoiPoolingWithArgMax")
      .Input(input)
      .Input(rois)
      .Input(roi_actual_num)
      .Output(output)
      .Output(argmax)
      .Attr("pooled_h", pooled_height_64)
      .Attr("pooled_w", pooled_width_64)
      .Attr("spatial_scale_h", spatial_scale)
      .Attr("spatial_scale_w", spatial_scale)
      .Attr("pool_channel", pooled_channel)
      .Run();

}

template <typename T>
void roi_pool_backward_kernel_impl(
    const at::Tensor& grad,
    const at::Tensor& input,
    at::Tensor& output,
    const at::Tensor& argmax,
    const float spatial_scale,
    int num_rois,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    const at::Tensor& rois) {
  int64_t pooled_height_64 = pooled_height;
  int64_t pooled_width_64 = pooled_width;
  int64_t pooled_channel = 1;
  at::Tensor roi_actual_num = at_npu::native::OpPreparation::ApplyTensor(
      {}, rois.options().dtype(at::kInt), rois);

  at_npu::native::OpCommand cmd;
  cmd.Name("RoiPoolingGradWithArgMax")
      .Input(grad)
      .Input(input)
      .Input(rois)
      .Input(roi_actual_num)
      .Input(argmax)
      .Output(output)
      .Attr("pooled_h", pooled_height_64)
      .Attr("pooled_w", pooled_width_64)
      .Attr("spatial_scale_h", spatial_scale)
      .Attr("spatial_scale_w", spatial_scale)
      .Attr("pool_channel", pooled_channel)
      .Run();
}

std::tuple<at::Tensor, at::Tensor> roi_pool_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_pool_forward_kernel";
  at::checkAllSameType(c, {input_t, rois_t});

  int num_rois = rois.size(0);
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  at::Tensor argmax = at::zeros(
      {num_rois, channels, pooled_height, pooled_width},
      input.options().dtype(at::kInt));

  if (output.numel() == 0) {
    return std::make_tuple(output, argmax);
  }

  auto input_ = input.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "roi_pool_forward_kernel", [&] {
        roi_pool_forward_kernel_impl<scalar_t>(
            input_,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois_,
            num_rois,
            output,
            argmax);
      });
  return std::make_tuple(output, argmax);
}

at::Tensor roi_pool_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  TORCH_CHECK(
      rois.size(1) == 5, "Tensor rois should have shape as Tensor[K, 5]");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_pool_backward_kernel";
  at::checkAllSameType(c, {grad_t, rois_t});

  auto num_rois = rois.size(0);

  at::Tensor grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());
  at::Tensor output =
      at::zeros({batch_size, channels, height, width}, grad.options());

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "roi_pool_backward_kernel", [&] {
        roi_pool_backward_kernel_impl<scalar_t>(
            grad,
            grad_input,
            output,
            argmax,
            spatial_scale,
            num_rois,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois_);
      });
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_pool"),
      TORCH_FN(roi_pool_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_roi_pool_backward"),
      TORCH_FN(roi_pool_backward_kernel));
}

} // namespace ops
} // namespace vision

