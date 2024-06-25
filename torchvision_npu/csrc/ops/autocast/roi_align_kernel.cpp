// Copyright (c) 2024, Huawei Technologies.All rights reserved.
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

#include "utils.hpp"
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace vision {
namespace ops {

namespace {

at::Tensor roi_align_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned)
{
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastPrivateUse1);
    static auto roi_align_handle = c10::Dispatcher::singleton()
                .findSchemaOrThrow("torchvision::roi_align", "")
                .typed<at::Tensor (const at::Tensor &, const at::Tensor &, double, int64_t, int64_t, int64_t, bool)>();
    return roi_align_handle.call(
        vision::autocast::cached_cast_npu(input, at::kFloat),
        vision::autocast::cached_cast_npu(rois, at::kFloat),
        spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned).to(input.scalar_type());
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, AutocastPrivateUse1, m)
{
    m.impl(
        TORCH_SELECTIVE_NAME("torchvision::roi_align"),
        TORCH_FN(roi_align_autocast));
}

} // namespace ops
} // namespace vision
