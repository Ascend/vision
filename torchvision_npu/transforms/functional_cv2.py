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

import numbers
from typing import List, Sequence, Union, Any

from PIL import Image
import cv2
import numpy as np

try:
    import accimage
except ImportError:
    accimage = None

from .utils import (
    MAX_VALUES_BY_DTYPE,
    clip,
    preserve_shape,
    preserve_channel_dim,
    is_rgb_image,
    is_grayscale_image,
    _is_numpy,
    get_num_channels,
    _maybe_process_in_chunks,
    _pillow2array
)


@preserve_shape
def hflip(img):
    # Opencv is faster than numpy only in case of
    # non-gray scale 8bits images
    if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
        return cv2.flip(img, 1)
    return np.ascontiguousarray(img[:, ::-1, ...])


def vflip(img):
    return np.ascontiguousarray(img[::-1, ...])


@preserve_channel_dim
def resize(img, size, interpolation=cv2.INTER_LINEAR):
    if isinstance(size, int) or len(size) == 1:
        if isinstance(size, Sequence):
            size = size[0]
        h, w = img.shape[:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
    else:
        ow, oh = size

    resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(oh, ow), interpolation=interpolation)
    return resize_fn(img)


def crop(img, top, left, height, width):
    x_min = left
    y_min = top
    x_max = left + width
    y_max = top + height
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )

    padding_bottom = 0
    padding_right = 0
    img_height, img_width = img.shape[:2]
    if y_max > img_height:
        padding_bottom = y_max - img_height
        y_max = img_height
    if x_max > img_width:
        padding_right = x_max - img_width
        x_max = img_width

    crop_img = img[y_min:y_max, x_min:x_max]
    if padding_bottom > 0 or padding_right > 0:
        crop_img = cv2.copyMakeBorder(crop_img, 0, padding_bottom, 0, padding_right, cv2.BORDER_CONSTANT, 0)

    return crop_img


def pad(img, padding, fill, padding_mode):
    if isinstance(fill, tuple):
        assert len(fill) == img.shape[-1]

    # check fill
    elif not isinstance(fill, numbers.Number):
        raise TypeError('fill must be a int or a tuple. '
                        f'But received {type(fill)}')

    # check padding
    if isinstance(padding, Sequence) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
        else:
            padding = (padding[0], padding[1], padding[2], padding[3])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(img, padding[1], padding[3], padding[0], padding[2], border_type[padding_mode], value=fill)

    return img


@preserve_channel_dim
def rotate(img, angle, interpolation=cv2.INTER_LINEAR, expand=False, center=None, fill=None):
    height, width = img.shape[:2]
    # for images we use additional shifts of (0.5, 0.5) as otherwise
    # we get an ugly black border for 90deg rotations
    if center is None:
        center = (width / 2 - 0.5, height / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    if expand:
        new_w = int(height * np.abs(matrix[0, 1]) + width * np.abs(matrix[0, 0]))
        new_h = int(height * np.abs(matrix[0, 0]) + width * np.abs(matrix[0, 1]))
        matrix[0, 2] += (new_w - width) / 2
        matrix[1, 2] += (new_h - height) / 2
        width = new_w + 2
        height = new_h + 2

    border_value = fill
    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderValue=border_value
    )
    return warp_fn(img)


@preserve_channel_dim
def affine(img, matrix, interpolation=cv2.INTER_LINEAR, fill=None):
    height, width = img.shape[:2]
    border_value = fill
    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderValue=border_value
    )
    return warp_fn(img)


def invert(img):
    return MAX_VALUES_BY_DTYPE[img.dtype] - img


@preserve_channel_dim
def perspective(img, matrix, interpolation=cv2.INTER_LINEAR, fill=None):
    height, width = img.shape[:2]
    border_value = fill
    warp_fn = _maybe_process_in_chunks(
        cv2.warpPerspective, M=matrix, dsize=(width, height), flags=interpolation, borderValue=border_value
    )
    return warp_fn(img)


def _adjust_brightness_uint8(img, factor):
    lut = np.arange(0, 256) * factor
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


@preserve_shape
def adjust_brightness(img, factor):
    if factor == 0:
        return np.zeros_like(img)
    elif factor == 1:
        return img

    if img.dtype == np.uint8:
        return _adjust_brightness_uint8(img, factor)

    return clip(img * factor, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])


def _adjust_contrast_uint8(img, factor, mean):
    lut = np.arange(0, 256) * factor
    lut = lut + mean * (1 - factor)
    lut = clip(lut, img.dtype, 255)

    return cv2.LUT(img, lut)


@preserve_shape
def adjust_contrast(img, factor):
    if factor == 1:
        return img

    if is_grayscale_image(img):
        mean = img.mean()
    else:
        mean = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()

    if factor == 0:
        if img.dtype != np.float32:
            mean = int(mean + 0.5)
        return np.full_like(img, mean, dtype=img.dtype)

    if img.dtype == np.uint8:
        return _adjust_contrast_uint8(img, factor, mean)

    return clip(
        img.astype(np.float32) * factor + mean * (1 - factor),
        img.dtype,
        MAX_VALUES_BY_DTYPE[img.dtype],
    )


@preserve_shape
def adjust_saturation(img, factor, gamma=0):
    if factor == 1:
        return img

    if is_grayscale_image(img):
        gray = img
        return gray
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if factor == 0:
        return gray

    result = cv2.addWeighted(img, factor, gray, 1 - factor, gamma=gamma)
    if img.dtype == np.uint8:
        return result

    # OpenCV does not clip values for float dtype
    return clip(result, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])


def _adjust_hue_uint8(img, factor):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lut = np.arange(0, 256, dtype=np.int16)
    lut = np.mod(lut + 180 * factor, 180).astype(np.uint8)
    img[..., 0] = cv2.LUT(img[..., 0], lut)

    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def adjust_hue(img, factor):
    if is_grayscale_image(img):
        return img

    if factor == 0:
        return img

    if img.dtype == np.uint8:
        return _adjust_hue_uint8(img, factor)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[..., 0] = np.mod(img[..., 0] + factor * 360, 360)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


@preserve_shape
def posterize(img, bits):
    bits = np.uint8(bits)

    if img.dtype != np.uint8:
        raise TypeError("Image must have uint8 channel type")
    if np.any((bits < 0) | (bits > 8)):
        raise ValueError("bits must be in range [0, 8]")

    if not bits.shape or len(bits) == 1:
        if bits == 0:
            return np.zeros_like(img)
        if bits == 8:
            return img.copy()

        lut = np.arange(0, 256, dtype=np.uint8)
        mask = ~np.uint8(2 ** (8 - bits) - 1)
        lut &= mask

        return cv2.LUT(img, lut)

    if not is_rgb_image(img):
        raise TypeError("If bits is iterable image must be RGB")

    result_img = np.empty_like(img)
    for i, channel_bits in enumerate(bits):
        if channel_bits == 0:
            result_img[..., i] = np.zeros_like(img[..., i])
        elif channel_bits == 8:
            result_img[..., i] = img[..., i].copy()
        else:
            lut = np.arange(0, 256, dtype=np.uint8)
            mask = ~np.uint8(2 ** (8 - channel_bits) - 1)
            lut &= mask

            result_img[..., i] = cv2.LUT(img[..., i], lut)

    return result_img


def solarize(img, threshold=128):
    """Invert all pixel values above a threshold.

    Args:
        img (numpy.ndarray): The image to solarize.
        threshold (int): All pixels above this greyscale level are inverted.

    Returns:
        numpy.ndarray: Solarized image.

    """
    dtype = img.dtype
    max_val = MAX_VALUES_BY_DTYPE[dtype]

    if dtype == np.dtype("uint8"):
        lut = [(i if i < threshold else max_val - i) for i in range(max_val + 1)]

        prev_shape = img.shape
        img = cv2.LUT(img, np.array(lut, dtype=dtype))

        if len(prev_shape) != len(img.shape):
            img = np.expand_dims(img, -1)
        return img

    result_img = img.copy()
    cond = img >= threshold
    result_img[cond] = max_val - result_img[cond]
    return result_img


def adjust_sharpness(img, factor=1.):
    kernel = np.array([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]]) / 13
    assert isinstance(kernel, np.ndarray), \
        f'kernel must be of type np.ndarray, but got {type(kernel)} instead.'
    assert kernel.ndim == 2, \
        f'kernel must have a dimension of 2, but got {kernel.ndim} instead.'

    degenerated = cv2.filter2D(img, -1, kernel)
    sharpened_img = cv2.addWeighted(
        img.astype(np.float32), factor, degenerated.astype(np.float32),
        1 - factor, 0)
    sharpened_img = np.clip(sharpened_img, 0, 255)
    return sharpened_img.astype(img.dtype)


def autocontrast(img, cutoff=0):
    n_bins = 256

    def tune_channel(ch):
        n = ch.size
        cut = cutoff * n // 100
        if cut == 0:
            high, low = ch.max(), ch.min()
        else:
            hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
            low = np.argwhere(np.cumsum(hist) > cut)
            low = 0 if low.shape[0] == 0 else low[0]
            high = np.argwhere(np.cumsum(hist[::-1]) > cut)
            high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]
        if high <= low:
            table = np.arange(n_bins)
        else:
            scale = (n_bins - 1) / (high - low)
            offset = -low * scale
            table = np.arange(n_bins) * scale + offset
            table[table < 0] = 0
            table[table > n_bins - 1] = n_bins - 1
        table = table.clip(0, 255).astype(np.uint8)
        return table[ch]

    channels = [tune_channel(ch) for ch in cv2.split(img)]
    out = cv2.merge(channels)
    return out


def equalize(img):
    if img.dtype != np.uint8:
        raise TypeError("Image must have uint8 channel type")
    result_img = np.empty_like(img)
    for i in range(3):
        result_img[..., i] = cv2.equalizeHist(img[..., i])
    return result_img


@preserve_shape
def gaussian_blur(img, ksize, sigma):
    blur_fn = _maybe_process_in_chunks(cv2.GaussianBlur, ksize=ksize, sigmaX=sigma[0], sigmaY=sigma[1])
    return blur_fn(img)


def to_grayscale(img, num_output_channels):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if num_output_channels == 1:
        return img
    elif num_output_channels == 3:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError('num_output_channels should be either 1 or 3')


def _get_image_size(img):
    if isinstance(img, np.ndarray):
        height, width = img.shape[:2]
        return [width, height]
    if isinstance(img, Image.Image):
        return img.size
    raise TypeError('Unexpected type {}'.format(type(img)))


def _get_image_num_channels(img) -> int:
    if isinstance(img, np.ndarray):
        return img.shape[2] if len(img.shape) == 3 else 1
    if isinstance(img, Image.Image):
        return 1 if img.mode == 'L' else 3
    raise TypeError('Unexpected type {}'.format(type(img)))
