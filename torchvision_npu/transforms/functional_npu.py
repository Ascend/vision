# Copyright (c) 2022, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import numpy as np
import torch
import torch_npu
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def normalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    if not inplace:
        return torch_npu.image_normalize(tensor, mean, std, 0)
    return torch_npu.image_normalize_(tensor, mean, std, 0)


def hflip(img: Tensor) -> Tensor:
    return torch_npu.reverse(img, axis=[3])


def resized_crop(
    img: Tensor,
    i: int,
    j: int,
    h: int,
    w: int,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR
) -> Tensor:
    interpolations = [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]
    if interpolation not in interpolations:
        raise ValueError(f'Tochvision_NPU cannot support this interpolation method')
    width, height = img.shape[-1], img.shape[-2]
    if width <= 1 or height <= 1:
        return img
    boxes = np.minimum([i / (height - 1), j / (width - 1), (i + h) / (height - 1), (j + w) / (width - 1)], 1).tolist()
    box_index = [0]
    crop_size = size
    return torch_npu.crop_and_resize(img, boxes, box_index, crop_size, method=interpolation.value)


def to_tensor(pic) -> Tensor:
    return torch_npu.img_to_tensor(pic)
