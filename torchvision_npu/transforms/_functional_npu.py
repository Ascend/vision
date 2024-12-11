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

from typing import Optional, Tuple, List

import numpy as np
import torch
import torch_npu
from torch import Tensor
from ._utils import deal_with_tensor_batch

_interpolation_crop_and_resize_int2str = {0: "bilinear", 1: "nearest", 2: "bicubic"}
_interpolation_resize_int2str = {0: "linear", 1: "nearest", 2: "cubic"}
_padding_mode_int2str = {0: "constant", 1: "edge", 2: "reflect", 3: "symmetric"}
_gb_kernel_size = [1, 3, 5]


def _assert_channels(img: Tensor, permitted: List[int]) -> None:
    c = img.shape[1]
    if c not in permitted:
        raise TypeError("Input image tensor permitted channel values are {}, but found {}".format(permitted, c))


def _assert_mode(mode: int, supported_modes: List[int]) -> None:
    if mode not in supported_modes:
        raise ValueError("Interpolation mode '{}' is unsupported with Tensor input".format(interpolation))


@deal_with_tensor_batch
def _normalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    result = tensor
    result = torch.ops.torchvision._normalize_aclnn(tensor, mean=mean, std=std)
    if not inplace:
        return result
    tensor = result.clone()
    return tensor


@deal_with_tensor_batch
def _vflip(img: Tensor) -> Tensor:
    return torch.ops.torchvision._vertical_flip_aclnn(img)


@deal_with_tensor_batch
def _hflip(img: Tensor) -> Tensor:
    return torch.ops.torchvision._horizontal_flip_aclnn(img)


@deal_with_tensor_batch
def _resized_crop(img: Tensor, crop_param: List[int], size: List[int], interpolation: int = 0) -> Tensor:
    i, j, h, w = [p for p in crop_param]
    return torch.ops.torchvision._crop_and_resize_aclnn(img, top=i, left=j, height=h, width=w,
                                                            size=size, interpolation_mode=interpolation)


@deal_with_tensor_batch
def _to_tensor(pic) -> Tensor:
    return torch.ops.torchvision._img_to_tensor_aclnn(pic)


@deal_with_tensor_batch
def _resize(img: Tensor, size: List[int], interpolation: int = 0) -> Tensor:
    if not isinstance(size, (int, tuple, list)):
        raise TypeError("Got inappropriate size arg")
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
            size_h = size_w * h / w
        else:
            size_w = size_h * w / h

        if w <= h and w == size_w:
            return img
        if h <= w and h == size_h:
            return img

    sizes = [int(size_h), int(size_w)]

    return torch.ops.torchvision._resize_aclnn(img, size=sizes, interpolation_mode=interpolation)


@deal_with_tensor_batch
def _crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    return torch.ops.torchvision._crop_aclnn(img, top=top, left=left, height=height, width=width)


@deal_with_tensor_batch
def _pad(img: Tensor, padding: List[int], fill: int = 0, padding_mode: int = 0) -> Tensor:
    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (int, float)):
        raise TypeError("Got inappropriate fill arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list) and len(padding) not in [1, 2, 4]:
        raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

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

    p = [pad_left, pad_top, pad_right, pad_bottom]
    if min(p) < 0:
        raise ValueError("pad value ({}) is not non-negative.".format(min(p)))

    fill = [fill, fill, fill]
    return torch.ops.torchvision._pad_aclnn(img, padding=p, padding_mode=padding_mode, fill=fill)


@deal_with_tensor_batch
def _adjust_brightness(img: Tensor, brightness_factor: float) -> Tensor:
    if brightness_factor < 0:
        raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))

    return torch.ops.torchvision._adjust_brightness_aclnn(img, factor=brightness_factor)


@deal_with_tensor_batch
def _adjust_contrast(img: Tensor, contrast_factor: float) -> Tensor:
    if contrast_factor < 0:
        raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))

    return torch.ops.torchvision._adjust_contrast_aclnn(img, factor=contrast_factor)


@deal_with_tensor_batch
def _adjust_hue(img: Tensor, hue_factor: float) -> Tensor:
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    image_num_channels = img.shape[-3]
    if image_num_channels == 1: # Match PIL behaviour
        return img

    return torch.ops.torchvision._adjust_hue_aclnn(img, factor=hue_factor)


@deal_with_tensor_batch
def _adjust_saturation(img: Tensor, saturation_factor: float) -> Tensor:
    if saturation_factor < 0:
        raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))

    return torch.ops.torchvision._adjust_saturation_aclnn(img, factor=saturation_factor)


@deal_with_tensor_batch
def _gaussian_blur(img: Tensor, kernel_size: List[int], sigma: List[float]) -> Tensor:
    if kernel_size[0] not in _gb_kernel_size or kernel_size[1] not in _gb_kernel_size:
        raise ValueError("sigma value must be in range {}.".format(_gb_kernel_size))

    padding_mode = 2 # reflect mode is closer to the native implementation

    return torch.ops.torchvision._gaussian_blur_aclnn(img, kernel_size=kernel_size, sigma=sigma,
                                                          padding_mode=padding_mode)


@deal_with_tensor_batch
def _rotate(img: Tensor, rotate_force_param: List,
           center: Optional[List[int]] = None, fill: Optional[List[float]] = None
) -> Tensor:
    angle, interpolation, expand = [p for p in rotate_force_param]
    _assert_mode(interpolation, [0, 1])
    
    if center is None:
        center = []
    else:
        center = [int(x) for x in center]

    if fill is not None:
        if isinstance(fill, float):
            fill = [fill, fill, fill]
        if isinstance(fill, (tuple, list)):
            fill = [fill[0], fill[0], fill[0]]

    return torch.ops.torchvision._rotate_aclnn(img, angle=angle, interpolation_mode=interpolation, expand=expand,
        center=center, padding_mode=0, fill=fill)


@deal_with_tensor_batch
def _affine(img: Tensor, matrix: List[float], interpolation: int = 1, fill: Optional[List[float]] = None
) -> Tensor:
    _assert_mode(interpolation, [0, 1])
    if len(matrix) != 6:
        raise ValueError(f"Argument matrix should have 6 float values but have {len(matrix)}")

    if fill is not None:
        if isinstance(fill, float):
            fill = [fill, fill, fill]
        if isinstance(fill, (tuple, list)):
            fill = [fill[0], fill[0], fill[0]]

    return torch.ops.torchvision._warp_affine_aclnn(img, matrix=matrix, interpolation_mode=interpolation,
        padding_mode=0, fill=fill)


@deal_with_tensor_batch
def _perspective(img: Tensor, matrix: List[float], interpolation: int = 0, fill: Optional[List[float]] = None
) -> Tensor:
    _assert_mode(interpolation, [0, 1])
    if len(matrix) != 8:
        raise ValueError(f"Argument matrix should have 8 float values but have {len(matrix)}")

    matrix.append(1)

    if fill is not None:
        if isinstance(fill, float):
            fill = [fill, fill, fill]
        if isinstance(fill, (tuple, list)):
            fill = [fill[0], fill[0], fill[0]]

    return torch.ops.torchvision._warp_perspective_aclnn(img, matrix=matrix, interpolation_mode=interpolation,
        padding_mode=0, fill=fill)


@deal_with_tensor_batch
def _rgb_to_grayscale(img: Tensor, num_output_channels: int = 1) -> Tensor:
    _assert_channels(img, [3])

    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')

    return torch.ops.torchvision._rgb_to_grayscale_aclnn(img, output_channels_num=num_output_channels)


@deal_with_tensor_batch
def _posterize(img: Tensor, bits: int) -> Tensor:
    if img.dtype != torch.uint8:
        raise TypeError("Only torch.uint8 image tensors are supported, but found {}".format(img.dtype))
    _assert_channels(img, [1, 3])

    return torch.ops.torchvision._posterize_aclnn(img, bits=bits)


@deal_with_tensor_batch
def _solarize(img: Tensor, threshold: float) -> Tensor:
    _assert_channels(img, [1, 3])

    threshold = [threshold]
    return torch.ops.torchvision._solarize_aclnn(img, threshold=threshold)


@deal_with_tensor_batch
def _invert(img: Tensor) -> Tensor:
    _assert_channels(img, [1, 3])

    return torch.ops.torchvision._invert_aclnn(img)
