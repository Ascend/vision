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

from typing import List, Optional

import torch
from torch import Tensor
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import functional_tensor as F_t
from torchvision.transforms import functional_pil as F_pil

from ..utils import _log_api_usage_once
from . import functional_npu as F_npu


def patch_transform_methods():
    setattr(torchvision.transforms.functional, "normalize_ori", torchvision.transforms.functional.normalize)
    torchvision.transforms.functional.normalize = normalize
    torchvision.transforms.functional.hflip = hflip
    torchvision.transforms.functional.resized_crop = resized_crop
    setattr(torchvision.transforms.functional, "to_tensor_ori", torchvision.transforms.functional.to_tensor)
    torchvision.transforms.functional.to_tensor = to_tensor

def normalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    """Normalize a float tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(normalize)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"img should be Tensor Image. Got {type(tensor)}")


    if tensor.device.type == 'npu':
        return F_npu.normalize(tensor, mean, std, inplace)

    return F.normalize_ori(tensor, mean=mean, std=std, inplace=inplace)


def hflip(img: Tensor) -> Tensor:
    """Horizontally flip the given image.

    Args:
        img (PIL Image or Tensor): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of leading
            dimensions.

    Returns:
        PIL Image or Tensor:  Horizontally flipped image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(hflip)


    if not isinstance(img, torch.Tensor):
        return F_pil.hflip(img)

    if hasattr(img, 'device') and hasattr(img.device, 'type') and img.device.type == 'npu':
       return F_npu.hflip(img)

    return F_t.hflip(img)

def resized_crop(
    img: Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: torchvision.transforms.InterpolationMode = torchvision.transforms.InterpolationMode.BILINEAR,
    antialias: Optional[bool] = None,
) -> Tensor:
    """Crop the given image and resize it to desired size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.NEAREST_EXACT``, ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are
            supported.
            For backward compatibility integer values (e.g. ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
        antialias (bool, optional): antialias flag. If ``img`` is PIL Image, the flag is ignored and anti-alias
            is always used. If ``img`` is Tensor, the flag is False by default and can be set to True for
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` modes.
            This can help making the output for PIL images and tensors closer.
    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(resized_crop)
    if isinstance(img, torch.Tensor) and img.device.type == 'npu':
        return F_npu.resized_crop(img, top, left, height, width, size, interpolation)
    img = F.crop(img, top, left, height, width)
    img = F.resize(img, size, interpolation)
    return img


def to_tensor(pic) -> Tensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if isinstance(pic, torch.Tensor):
        if pic.device.type == 'npu':
            return pic.to(dtype=torch.get_default_dtype(), non_blocking=True).div(255)
    return torchvision.transforms.functional.to_tensor_ori(pic)