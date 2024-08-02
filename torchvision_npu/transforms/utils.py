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

from functools import wraps
from typing import Callable, Any

from torch import Tensor
from PIL import Image, ImageOps
import numpy as np
import torch

from typing_extensions import Concatenate, ParamSpec
from torchvision.transforms import functional_pil as F_pil

__all__ = [
    "MAX_VALUES_BY_DTYPE",
    "clip",
    "preserve_shape",
    "preserve_channel_dim",
    "is_rgb_image",
    "is_grayscale_image",
    "_is_numpy_image",
    "_is_numpy",
    "get_num_channels",
    "_maybe_process_in_chunks",
    "_pillow2array"
]

P = ParamSpec("P")

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}


def clip(img: np.ndarray, dtype: np.dtype, maxval: float) -> np.ndarray:
    return np.clip(img, 0, maxval).astype(dtype)


def preserve_shape(
        func: Callable[Concatenate[np.ndarray, P], np.ndarray]
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Preserve shape of the image"""

    @wraps(func)
    def wrapped_function(img: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function


def preserve_channel_dim(
        func: Callable[Concatenate[np.ndarray, P], np.ndarray]
) -> Callable[Concatenate[np.ndarray, P], np.ndarray]:
    """Preserve dummy channel dim."""

    @wraps(func)
    def wrapped_function(img: np.ndarray, *args: P.args, **kwargs: P.kwargs) -> np.ndarray:
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == 3 and shape[-1] == 1 and len(result.shape) == 2:
            result = np.expand_dims(result, axis=-1)
        return result

    return wrapped_function


def deal_with_tensor_batch(func: Callable) -> Callable:
    """Deal with multi batch of tensor in npu"""

    @wraps(func)
    def wrapped_function(img: Tensor, *args: P.args, **kwargs: P.kwargs) -> Tensor:
        if img.ndim == 4:    
            processed_tensors = []
            for i in range(img.shape[0]):
                tensor = func(img[i].unsqueeze(0), *args, **kwargs)
                processed_tensors.append(tensor)
            batch_tensor = torch.cat(processed_tensors, dim=0)
        elif img.ndim == 3:
            batch_tensor = func(img.unsqueeze(0), *args, **kwargs).squeeze(0)
        else:
            raise ValueError('Expected tensor to be a tensor image of size (C, H, W) or (N, C, H, W). Got tensor.size() = '
                         '{}.'.format(img.size()))
        return batch_tensor
        
    return wrapped_function


def is_rgb_image(image: np.ndarray) -> bool:
    return len(image.shape) == 3 and image.shape[-1] == 3


def is_grayscale_image(image: np.ndarray) -> bool:
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)


def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}


def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


def get_num_channels(image: np.ndarray) -> int:
    return image.shape[2] if len(image.shape) == 3 else 1


def _maybe_process_in_chunks(
        process_fn: Callable[Concatenate[np.ndarray, P], np.ndarray], **kwargs
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.

    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.

    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.

    Returns:
        numpy.ndarray: Transformed image.

    """

    @wraps(process_fn)
    def __process_fn(img: np.ndarray) -> np.ndarray:
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i: index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index: index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def _pillow2array(img, flag: str = 'color', channel_order: str = 'bgr') -> np.ndarray:
    """
    Convert a pillow image to numpy array.
    Args:
        img: (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag: (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order: The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.


    Returns: np.ndarray: The converted numpy array

    """

    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ['color', 'grayscale']:
            img = ImageOps.exif_transpose(img)
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag in ['color', 'color_ignore_orientation']:
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag in ['grayscale', 'grayscale_ignore_orientation']:
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", '
                f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
                f' but got {flag}')
    return array
