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

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

const int SIZE = 8;

template <typename scalar_t>
at::Tensor nms_kernel_impl(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    double iou_threshold) {
    at::Tensor iou_threshold_y = at::empty({}, boxes.options().dtype(at::kFloat)).fill_(iou_threshold);
    at::Tensor scores_threshold_y = at::empty({}, boxes.options().dtype(at::kFloat)).fill_(0);
    at::Tensor max_outputsize_y = at::empty({}, boxes.options().dtype(at::kInt)).fill_(boxes.size(0));
    c10::SmallVector<int64_t, SIZE> outputsize = {boxes.size(0)};
    at::Tensor output = at::empty(outputsize, boxes.options().dtype(at::kInt)).fill_(-1);
    at_npu::native::OpCommand cmd;
    cmd.Name("NonMaxSuppressionV3")
      .Input(boxes)
      .Input(scores)
      .Input(max_outputsize_y)
      .Input(iou_threshold_y)
      .Input(scores_threshold_y)
      .Output(output)
      .Run();
    auto outputsizeBool = at::gt(output, -1);
    auto outputsizeInt = outputsizeBool.to(at::ScalarType::Int);
    auto countLen = at::sum(outputsizeInt, at::ScalarType::Int);
    at::Tensor actual_output = output.slice(0, 0, countLen.item().toLong());
    actual_output = actual_output.to(at::kLong);
    return actual_output;

}

at::Tensor nms_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {


  auto result = at::empty({0}, dets.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(dets.scalar_type(), "nms_kernel", [&] {
    result = nms_kernel_impl<scalar_t>(dets, scores, iou_threshold);
  });
  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel));
}

} // namespace ops
} // namespace vision
