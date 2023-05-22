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
from torchvision.transforms import functional_tensor as F_t

_interpolation_tv2npu = {"nearest": "nearest", "bilinear": "linear", "bicubic": "cubic"}
_gb_kernel_size = [1, 3, 5]


def normalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    if not inplace:
        return torch_npu.image_normalize(tensor, mean=mean, variance=std, dtype=0)
    return torch_npu.image_normalize_(tensor, mean=mean, variance=std, dtype=0)


def vflip(img: Tensor) -> Tensor:
    return torch_npu.reverse(img, axis=[2])


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
        raise ValueError("Npu can only support nearest and bilinear interpolation mode.")
    width, height = img.shape[-1], img.shape[-2]
    if width <= 1 or height <= 1:
        return img
    boxes = np.minimum([i / (height - 1), j / (width - 1), (i + h) / (height - 1), (j + w) / (width - 1)], 1).tolist()
    box_index = [0]
    crop_size = size
    return torch_npu.crop_and_resize(img,
        boxes=boxes, box_index=box_index, crop_size=crop_size, method=interpolation.value)


def to_tensor(pic) -> Tensor:
    return torch_npu.img_to_tensor(pic)


def resize(img: Tensor, size: List[int], interpolation: str = "bilinear") -> Tensor:
    if not isinstance(size, (int, tuple, list)):
        raise TypeError("Got inappropriate size arg")
    if not isinstance(interpolation, str):
        raise TypeError("Got inappropriate interpolation arg")

    if interpolation not in ["nearest", "bilinear", "bicubic"]:
        raise ValueError("This interpolation mode is unsupported with Tensor input")

    if isinstance(size, tuple):
        size = list(size)

    if isinstance(size, list) and len(size) not in [1, 2]:
        raise ValueError("Size must be an int or a 1 or 2 element tuple/list, not a "
                         "{} element tuple/list".format(len(size)))

    w, h = img.shape[-1], img.shape[-2]

    if isinstance(size, int):
        size_w, size_h = size, size
    elif len(size) < 2:
        size_w, size_h = size[0], size[0]
    else:
        size_w, size_h = size[1], size[0]  # Convention (h, w)

    if isinstance(size, int) or len(size) < 2:
        if w < h:
            size_h = int(size_w * h / w)
        else:
            size_w = int(size_h * w / h)

        if (w <= h and w == size_w) or (h <= w and h == size_h):
            return img

    sizes = [size_h, size_w]
    mode = _interpolation_tv2npu.get(interpolation)

    return torch.ops.torchvision.npu_resize(img, size=sizes, mode=mode)


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    size = [img.shape[0], img.shape[1], height, width]
    axis = 2 # crop start from the 3th(Height) axis
    offsets = [top, left] # crop start point(the upper left corner) coordinate

    return torch.ops.torchvision.npu_crop(img, size=size, axis=axis, offsets=offsets)


def pad(img: Tensor, padding: List[int], fill: int = 0, padding_mode: str = "constant") -> Tensor:
    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (int, float)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list) and len(padding) not in [1, 2, 4]:
        raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

    if isinstance(padding, int):
        if torch.jit.is_scripting():
            # This maybe unreachable
            raise ValueError("padding can't be an int while torchscripting, set it as a list [value, ]")
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 1:
        pad_left = pad_right = pad_top = pad_bottom = padding[0]
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    p = [pad_left, pad_right, pad_top, pad_bottom]

    if min(p) < 0:
        raise ValueError("pad value ({}) is not non-negative.".format(min(p)))

    return torch.ops.torchvision.npu_pad2d(img, pad=p, constant_values=fill, mode=padding_mode)


def adjust_brightness(img: Tensor, brightness_factor: float) -> Tensor:
    if brightness_factor < 0:
        raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))

    return torch.ops.torchvision.npu_adjust_brightness(img, factor=brightness_factor)


def adjust_contrast(img: Tensor, contrast_factor: float) -> Tensor:
    if contrast_factor < 0:
        raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))

    return torch.ops.torchvision.npu_adjust_contrast(img, factor=contrast_factor)


def adjust_hue(img: Tensor, hue_factor: float) -> Tensor:
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    image_num_channels = img.shape[-3] if img.ndim > 2 else 1
    if image_num_channels == 1: # Match PIL behaviour
        return img

    return torch.ops.torchvision.npu_adjust_hue(img, factor=hue_factor)


def adjust_saturation(img: Tensor, saturation_factor: float) -> Tensor:
    if saturation_factor < 0:
        raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))

    return torch.ops.torchvision.npu_adjust_saturation(img, factor=saturation_factor)


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    if kernel_size[0] not in _gb_kernel_size or kernel_size[1] not in _gb_kernel_size:
        raise ValueError("sigma value must be in range {}.".format(_gb_kernel_size))

    # reflect mode is closer to the native implementation
    return torch.ops.torchvision.npu_gaussian_blur(img, kernel_size=kernel_size, sigma=sigma, padding_mode="reflect")
