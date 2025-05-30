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

#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>


namespace vision {
namespace autocast {

inline at::Tensor cached_cast_npu(const at::Tensor& self, at::ScalarType dtype)
{
    return at::autocast::cached_cast(dtype, self, c10::DeviceType::PrivateUse1);
}

}
}

