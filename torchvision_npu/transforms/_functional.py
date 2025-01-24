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

import math
import numbers
import warnings
from typing import List, Tuple, Optional, Union

import numpy as np
from PIL import Image
import cv2

import torch
from torch import Tensor
import torchvision
from torchvision.transforms import functional as F
from torchvision.transforms import _functional_tensor as F_t
from torchvision.transforms import _functional_pil as F_pil
from torchvision.transforms import InterpolationMode
from torchvision.utils import _log_api_usage_once

from . import _functional_npu as F_npu
from . import _functional_cv2 as F_cv2
from ._utils import _pillow2array, _is_numpy, _is_numpy_image

try:
    import accimage
except ImportError:
    accimage = None


def patch_transform_methods():
    setattr(torchvision.transforms.functional, "normalize_ori", torchvision.transforms.functional.normalize)
    torchvision.transforms.functional.normalize = normalize
    torchvision.transforms.functional.hflip = hflip
    setattr(torchvision.transforms.functional, "to_tensor_ori", torchvision.transforms.functional.to_tensor)
    torchvision.transforms.functional.to_tensor = to_tensor
    torchvision.transforms.functional.pil_to_tensor = pil_to_tensor
    torchvision.transforms.functional.vflip = vflip
    setattr(torchvision.transforms.functional, "resize_ori", torchvision.transforms.functional.resize)
    torchvision.transforms.functional.resize = resize
    torchvision.transforms.functional.crop = crop
    torchvision.transforms.functional.pad = pad
    setattr(torchvision.transforms.functional, "rotate_ori", torchvision.transforms.functional.rotate)
    torchvision.transforms.functional.rotate = rotate
    setattr(torchvision.transforms.functional, "affine_ori", torchvision.transforms.functional.affine)
    torchvision.transforms.functional.affine = affine
    torchvision.transforms.functional.invert = invert
    torchvision.transforms.functional.perspective = perspective
    torchvision.transforms.functional.adjust_brightness = adjust_brightness
    torchvision.transforms.functional.adjust_contrast = adjust_contrast
    torchvision.transforms.functional.adjust_saturation = adjust_saturation
    torchvision.transforms.functional.adjust_hue = adjust_hue
    torchvision.transforms.functional.posterize = posterize
    torchvision.transforms.functional.solarize = solarize
    torchvision.transforms.functional.adjust_sharpness = adjust_sharpness
    torchvision.transforms.functional.autocontrast = autocontrast
    torchvision.transforms.functional.equalize = equalize
    setattr(torchvision.transforms.functional, "gaussian_blur_ori", torchvision.transforms.functional.gaussian_blur)
    torchvision.transforms.functional.gaussian_blur = gaussian_blur
    torchvision.transforms.functional.rgb_to_grayscale = rgb_to_grayscale
    torchvision.transforms.functional.get_image_size = _get_image_size
    torchvision.transforms.functional._get_image_num_channels = _get_image_num_channels


cv2_interpolation_mapping = {
    InterpolationMode.NEAREST: cv2.INTER_NEAREST,
    InterpolationMode.BILINEAR: cv2.INTER_LINEAR,
    InterpolationMode.BICUBIC: cv2.INTER_CUBIC,
    InterpolationMode.LANCZOS: cv2.INTER_LANCZOS4,
}


def _get_image_size(img: Tensor) -> List[int]:
    """Returns image size as [w, h]
    """
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2._get_image_size(img)

    if isinstance(img, torch.Tensor):
        return F_t.get_image_size(img)

    return F_pil.get_image_size(img)


def _get_image_num_channels(img: Tensor) -> int:
    """Returns number of image channels
    """
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2._get_image_num_channels(img)

    if isinstance(img, torch.Tensor):
        return F_t._get_image_num_channels(img)

    return F_pil._get_image_num_channels(img)


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

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if torchvision.get_image_backend() == 'moal':
        is_float_tensor = tensor.dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float64]
        if not (is_float_tensor and tensor.device.type == 'cpu'):
            raise TypeError(f"Normalize moal only support cpu float tensor.")
        return torch.ops.torchvision.normalize_moal(tensor, mean=mean, std=std, inplace=inplace)

    return F.normalize_ori(tensor, mean=mean, std=std, inplace=inplace)


def hflip(img: Tensor) -> Tensor:
    """Horizontally flip the given image.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of leading
            dimensions.

    Returns:
        PIL Image or Tensor or numpy.ndarray:  Horizontally flipped image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(hflip)

    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.hflip(img)

    if not isinstance(img, torch.Tensor):
        return F_pil.hflip(img)

    return F_t.hflip(img)


def to_tensor(pic) -> Tensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if isinstance(pic, torch.Tensor) and pic.dtype == torch.uint8 and pic.device.type == 'npu':
        return F_npu._to_tensor(pic)
    if torchvision.get_image_backend() == 'moal':
        if not (isinstance(pic, torch.Tensor) and pic.dtype == torch.uint8 and pic.device.type == 'cpu'):
            raise TypeError(f"ToTensor moal only support cpu uint8 tensor input.")
        return torch.ops.torchvision.to_tensor_moal(pic)
    return F.to_tensor_ori(pic)


def pil_to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray to a tensor of the same type.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.PILToTensor` for more details.

    .. note::

        A deep copy of the underlying array is performed.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(pil_to_tensor)
    if not (F_pil._is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if accimage is not None and isinstance(pic, accimage.Image):
        # accimage format is always uint8 internally, so always return uint8 here
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.uint8)
        pic.copyto(nppic)
        return torch.as_tensor(nppic)

    # handle numpy array
    if _is_numpy(pic):
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        return img

    # handle PIL Image
    img = torch.as_tensor(np.array(pic, copy=True))
    img = img.view(pic.size[1], pic.size[0], F_pil.get_image_num_channels(pic))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1))
    return img


def vflip(img: Tensor) -> Tensor:
    """Vertically flip the given image.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to be flipped. If img
            is a Tensor, it is expected to be in [..., H, W] format,
            where ... means it can have an arbitrary number of leading
            dimensions.

    Returns:
        PIL Image or Tensor or numpy.ndarray:  Vertically flipped image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(vflip)

    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.vflip(img)

    if not isinstance(img, torch.Tensor):
        return F_pil.vflip(img)

    return F_t.vflip(img)


def _compute_resized_output_size(
    image_size: Tuple[int, int], size: List[int], max_size: Optional[int] = None
) -> List[int]:
    if len(size) == 1:  # specified size only for the smallest edge
        h, w = image_size
        short, long = (w, h) if w <= h else (h, w)
        requested_new_short = size if isinstance(size, int) else size[0]

        new_short, new_long = requested_new_short, int(requested_new_short * long / short)

        if max_size is not None:
            if max_size <= requested_new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short / new_long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    else:  # specified both h and w
        new_w, new_h = size[1], size[0]
    return [new_h, new_w]


def resize(
    img: Tensor,
    size: List[int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[Union[str, bool]] = True,
) -> Tensor:
    r"""Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. warning::
        The output image might be different depending on its type: when downsampling, the interpolation of PIL images
        and tensors is slightly different, because PIL applies antialiasing. This may lead to significant differences
        in the performance of a network. Therefore, it is preferable to train and serve a model with the same input
        types. See also below the ``antialias`` parameter, which can help making the output of PIL images and tensors
        closer.

    Args:
        img (PIL Image or Tensor): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.NEAREST_EXACT``, ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are
            supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        max_size (int, optional): The maximum allowed for the longer edge of
            the resized image. If the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``,
            ``size`` will be overruled so that the longer edge is equal to
            ``max_size``.
            As a result, the smaller edge may be shorter than ``size``. This
            is only supported if ``size`` is an int (or a sequence of length
            1 in torchscript mode).
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True``: will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The current default is ``None`` **but will change to** ``True`` **in
            v0.17** for the PIL and Tensor backends to be consistent.

    Returns:
        PIL Image or Tensor: Resized image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(resize)

    if torchvision.get_image_backend() != 'cv2':
        return F.resize_ori(img, size, interpolation, max_size, antialias)

    if isinstance(interpolation, int):
        interpolation = F._interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if isinstance(size, (list, tuple)):
        if len(size) not in [1, 2]:
            raise ValueError(
                f"Size must be an int or a 1 or 2 element tuple/list, not a {len(size)} element tuple/list"
            )
        if max_size is not None and len(size) != 1:
            raise ValueError(
                "max_size should only be passed if size specifies the length of the smaller edge, "
                "i.e. size should be an int or a sequence of length 1 in torchscript mode."
            )

    image_width, image_height = _get_image_size(img)
    if isinstance(size, int):
        size = [size]
    output_size = _compute_resized_output_size((image_height, image_width), size, max_size)

    if [image_height, image_width] == output_size:
        return img

    if not isinstance(img, np.ndarray):
        raise TypeError(
            "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
    if interpolation not in cv2_interpolation_mapping:
        raise TypeError("Opencv does not support box and hamming interpolation")
    else:
        cv2_interpolation = cv2_interpolation_mapping[interpolation]
    return F_cv2.resize(img, size=output_size, interpolation=cv2_interpolation)


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """Crop the given image at specified location and output size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL Image or Tensor or numpy.ndarray: Cropped image.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(crop)
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.crop(img, top, left, height, width)

    if not isinstance(img, torch.Tensor):
        return F_pil.crop(img, top, left, height, width)

    return F_t.crop(img, top, left, height, width)


def pad(img: Tensor, padding: List[int], fill: int = 0, padding_mode: str = "constant") -> Tensor:
    r"""Pad the given image on all sides with the given "pad" value.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means at most 2 leading dimensions for mode reflect and symmetric,
    at most 3 leading dimensions for mode edge,
    and an arbitrary number of leading dimensions for mode constant

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to be padded.
        padding (int or sequence): Padding on each border. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
            In torchscript mode padding as single int is not supported, use a sequence of length 1: ``[padding, ]``.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0.
            If a tuple of length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or str or tuple value is supported for PIL Image.
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value on the edge of the image,
                    if input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image (without repeating the last value on the edge)

                       padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                       will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image (repeating the last value on the edge)

                         padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                         will result in [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        PIL Image or Tensor or numpy.ndarray: Padded image.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(pad)
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)

    if not isinstance(img, torch.Tensor):
        return F_pil.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)

    return F_t.pad(img, padding=padding, fill=fill, padding_mode=padding_mode)


def _get_affine_matrix(
    center: List[float], angle: float, translate: List[float], scale: float, shear: List[float]
) -> List[float]:
    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    sy_cos = math.cos(sy)
    if sy_cos != 0:
        a = math.cos(rot - sy) / sy_cos
        b = -math.cos(rot - sy) * math.tan(sx) / sy_cos - math.sin(rot)
        c = math.sin(rot - sy) / sy_cos
        d = -math.sin(rot - sy) * math.tan(sx) / sy_cos + math.cos(rot)
    else:
        raise ValueError("Zero division error, math.cos(sy)=0.")

    matrix = [a, b, 0.0, c, d, 0.0]
    matrix = [x * scale for x in matrix]
    # Apply inverse of center translation: RSS * C^-1
    matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
    matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
    # Apply translation and center : T * C * RSS * C^-1
    matrix[2] += cx + tx
    matrix[5] += cy + ty

    return matrix


def rotate(
        img: Tensor, angle: float, interpolation: InterpolationMode = InterpolationMode.NEAREST,
        expand: bool = False, center: Optional[List[int]] = None,
        fill: Optional[List[float]] = None
) -> Tensor:
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): image to be rotated.
        angle (number): rotation angle value in degrees, counter-clockwise.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
            In torchscript mode single int/float value is not supported, please use a sequence
            of length 1: ``[value, ]``.
            If input is PIL Image, the options is only available for ``Pillow>=5.2.0``.

    Returns:
        PIL Image or Tensor or numpy.ndarray: Rotated image.

    """
    if torchvision.get_image_backend() != 'cv2':
        return F.rotate_ori(img, angle, interpolation, expand, center, fill)

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(rotate)

    # Backward compatibility with integer value
    if isinstance(interpolation, int):
        warnings.warn(
            "Argument interpolation should be of type InterpolationMode instead of int. "
            "Please, use InterpolationMode enum."
        )
        interpolation = F._interpolation_modes_from_int(interpolation)

    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if not isinstance(img, np.ndarray):
        raise TypeError(
            "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
    if interpolation not in cv2_interpolation_mapping:
        raise TypeError("Opencv does not support box and hamming interpolation")
    else:
        cv2_interpolation = cv2_interpolation_mapping[interpolation]
    return F_cv2.rotate(img, angle=angle, interpolation=cv2_interpolation, expand=expand, center=center,
                        fill=fill)


def affine(
    img: Tensor,
    angle: float,
    translate: List[int],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[List[float]] = None,
    center: Optional[List[int]] = None,
) -> Tensor:
    """Apply affine transformation on the image keeping image center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): image to transform.
        angle (number): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (sequence of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float or sequence): shear angle value in degrees between -180 to 180, clockwise direction.
            If a sequence is specified, the first value corresponds to a shear parallel to the x-axis, while
            the second value corresponds to a shear parallel to the y-axis.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.
        center (sequence, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.

    Returns:
        PIL Image or Tensor or numpy.ndarray: Transformed image.
    """
    if torchvision.get_image_backend() != 'cv2':
        return F.affine_ori(img, angle, translate, scale, shear, interpolation, fill, center)

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(affine)

    if isinstance(interpolation, int):
        interpolation = F._interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if not isinstance(translate, (list, tuple)):
        raise TypeError("Argument translate should be a sequence")

    if len(translate) != 2:
        raise ValueError("Argument translate should be a sequence of length 2")

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    if not isinstance(shear, (numbers.Number, (list, tuple))):
        raise TypeError("Shear should be either a single value or a sequence of two values")

    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, tuple):
        translate = list(translate)

    if isinstance(shear, numbers.Number):
        shear = [shear, 0.0]

    if isinstance(shear, tuple):
        shear = list(shear)

    if len(shear) == 1:
        shear = [shear[0], shear[0]]

    if len(shear) != 2:
        raise ValueError(f"Shear should be a sequence containing two values. Got {shear}")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    width, height = _get_image_size(img)

    if not isinstance(img, np.ndarray):
        raise TypeError(
            "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
    if interpolation not in cv2_interpolation_mapping:
        raise TypeError("Opencv does not support box and hamming interpolation")
    cv2_interpolation = cv2_interpolation_mapping[interpolation]
    if center is None:
        center = [width * 0.5, height * 0.5]
    M = F._get_inverse_affine_matrix(center, -angle, translate, scale, shear)
    return F_cv2.affine(img, matrix=np.array(M).reshape(2, 3), interpolation=cv2_interpolation, fill=fill)


def invert(img: Tensor) -> Tensor:
    """Invert the colors of an RGB/grayscale image.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to have its colors inverted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Returns:
        PIL Image or Tensor or numpy.ndarray: Color inverted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(invert)
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.invert(img)

    if not isinstance(img, torch.Tensor):
        return F_pil.invert(img)

    return F_t.invert(img)


def perspective(
    img: Tensor,
    startpoints: List[List[int]],
    endpoints: List[List[int]],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    fill: Optional[List[float]] = None,
) -> Tensor:
    """Perform perspective transform of the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to be transformed.
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.

    Returns:
        PIL Image or Tensor or numpy.ndarray: transformed Image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(perspective)

    coeffs = F._get_perspective_coeffs(startpoints, endpoints)

    if isinstance(interpolation, int):
        interpolation = F._interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    # cv2 perspective matrix get different
    if torchvision.get_image_backend() == 'cv2':
        if interpolation not in cv2_interpolation_mapping:
            raise TypeError("Opencv does not support box and hamming interpolation")
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        cv2_interpolation = cv2_interpolation_mapping[interpolation]
        coeffs = cv2.getPerspectiveTransform(np.float32(startpoints), np.float32(endpoints))
        return F_cv2.perspective(img, coeffs, interpolation=cv2_interpolation, fill=fill)

    if not isinstance(img, torch.Tensor):
        pil_interpolation = F.pil_modes_mapping[interpolation]
        return F_pil.perspective(img, coeffs, interpolation=pil_interpolation, fill=fill)

    return F_t.perspective(img, coeffs, interpolation=interpolation.value, fill=fill)


def adjust_brightness(img: Tensor, brightness_factor: float) -> Tensor:
    """Adjust brightness of an image.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to be adjusted.
        If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
        where ... means it can have an arbitrary number of leading dimensions.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image or Tensor or numpy.ndarray: Brightness adjusted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(adjust_brightness)
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.adjust_brightness(img, brightness_factor)

    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_brightness(img, brightness_factor)

    return F_t.adjust_brightness(img, brightness_factor)


def adjust_contrast(img: Tensor, contrast_factor: float) -> Tensor:
    """Adjust contrast of an image.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to be adjusted.
        If img is torch Tensor, it is expected to be in [..., 3, H, W] format,
        where ... means it can have an arbitrary number of leading dimensions.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image or Tensor or numpy.ndarray: Contrast adjusted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(adjust_contrast)
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.adjust_contrast(img, contrast_factor)

    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_contrast(img, contrast_factor)

    return F_t.adjust_contrast(img, contrast_factor)


def adjust_saturation(img: Tensor, saturation_factor: float) -> Tensor:
    """Adjust color saturation of an image.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to be adjusted.
        If img is torch Tensor, it is expected to be in [..., 3, H, W] format,
        where ... means it can have an arbitrary number of leading dimensions.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image or Tensor or numpy.ndarray: Saturation adjusted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(adjust_saturation)
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.adjust_saturation(img, saturation_factor)

    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_saturation(img, saturation_factor)

    return F_t.adjust_saturation(img, saturation_factor)


def adjust_hue(img: Tensor, hue_factor: float) -> Tensor:
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details from wikipedia official website.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to be adjusted.
        If img is torch Tensor, it is expected to be in [..., 3, H, W] format,
        where ... means it can have an arbitrary number of leading dimensions.
        If img is PIL Image mode "1", "L", "I", "F" and modes with transparency (alpha channel) are not supported.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image or Tensor or numpy.ndarray: Hue adjusted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(adjust_hue)
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.adjust_hue(img, hue_factor)

    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_hue(img, hue_factor)
    if img.device.type == 'npu':
        return F_npu._adjust_hue(img, hue_factor)

    return F_t.adjust_hue(img, hue_factor)


def posterize(img: Tensor, bits: int) -> Tensor:
    """Posterize an image by reducing the number of bits for each color channel.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to have its colors posterized.
            If img is torch Tensor, it should be of type torch.uint8 and
            it is expected to be in [..., 1 or 3, H, W] format, where ... means
            it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".
        bits (int): The number of bits to keep for each channel (0-8).
    Returns:
        PIL Image or Tensor or numpy.ndarray: Posterized image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(posterize)
    if not (0 <= bits <= 8):
        raise ValueError('The number if bits should be between 0 and 8. Got {}'.format(bits))

    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.posterize(img, bits)

    if not isinstance(img, torch.Tensor):
        return F_pil.posterize(img, bits)

    return F_t.posterize(img, bits)


def solarize(img: Tensor, threshold: float) -> Tensor:
    """Solarize an RGB/grayscale image by inverting all pixel values above a threshold.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to have its colors inverted.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".
        threshold (float): All pixels equal or above this value are inverted.
    Returns:
        PIL Image or Tensor or numpy.ndarray: Solarized image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(solarize)
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.solarize(img, threshold)

    if not isinstance(img, torch.Tensor):
        return F_pil.solarize(img, threshold)

    return F_t.solarize(img, threshold)


def adjust_sharpness(img: Tensor, sharpness_factor: float) -> Tensor:
    """Adjust the sharpness of an image.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to be adjusted.
        If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
        where ... means it can have an arbitrary number of leading dimensions.
        sharpness_factor (float):  How much to adjust the sharpness. Can be
            any non negative number. 0 gives a blurred image, 1 gives the
            original image while 2 increases the sharpness by a factor of 2.

    Returns:
        PIL Image or Tensor or numpy.ndarray: Sharpness adjusted image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(adjust_sharpness)
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.adjust_sharpness(img, sharpness_factor)

    if not isinstance(img, torch.Tensor):
        return F_pil.adjust_sharpness(img, sharpness_factor)

    return F_t.adjust_sharpness(img, sharpness_factor)


def autocontrast(img: Tensor) -> Tensor:
    """Maximize contrast of an image by remapping its
    pixels per channel so that the lowest becomes black and the lightest
    becomes white.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image on which autocontrast is applied.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Returns:
        PIL Image or Tensor or numpy.ndarray: An image that was autocontrasted.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(autocontrast)
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.autocontrast(img)

    if not isinstance(img, torch.Tensor):
        return F_pil.autocontrast(img)

    return F_t.autocontrast(img)


def equalize(img: Tensor) -> Tensor:
    """Equalize the histogram of an image by applying
    a non-linear mapping to the input in order to create a uniform
    distribution of grayscale values in the output.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image on which equalize is applied.
            If img is torch Tensor, it is expected to be in [..., 1 or 3, H, W] format,
            where ... means it can have an arbitrary number of leading dimensions.
            If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".

    Returns:
        PIL Image or Tensor or numpy.ndarray: An image that was equalized.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(equalize)
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.equalize(img)

    if not isinstance(img, torch.Tensor):
        return F_pil.equalize(img)

    return F_t.equalize(img)


def gaussian_blur(img: Tensor, kernel_size: List[int], sigma: Optional[List[float]] = None) -> Tensor:
    """Performs Gaussian blurring on the image by given kernel.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): Image to be blurred
        kernel_size (sequence of ints or int): Gaussian kernel size. Can be a sequence of integers
            like ``(kx, ky)`` or a single integer for square kernels.
            In torchscript mode kernel_size as single int is not supported, use a sequence of length 1: ``[ksize, ]``.
        sigma (sequence of floats or float, optional): Gaussian kernel standard deviation. Can be a
            sequence of floats like ``(sigma_x, sigma_y)`` or a single float to define the
            same sigma in both X/Y directions. If None, then it is computed using
            ``kernel_size`` as ``sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8``.
            Default, None. In torchscript mode sigma as single float is
            not supported, use a sequence of length 1: ``[sigma, ]``.

    Returns:
        PIL Image or Tensor or numpy.ndarray: Gaussian Blurred version of the image.
    """
    if torchvision.get_image_backend != 'cv2':
        return F.gaussian_blur_ori(img, kernel_size, sigma)

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(gaussian_blur)
    if not isinstance(kernel_size, (int, list, tuple)):
        raise TypeError('kernel_size should be int or a sequence of integers. Got {}'.format(type(kernel_size)))
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if len(kernel_size) != 2:
        raise ValueError('If kernel_size is a sequence its length should be 2. Got {}'.format(len(kernel_size)))
    for ksize in kernel_size:
        if ksize % 2 == 0 or ksize < 0:
            raise ValueError('kernel_size should have odd and positive integers. Got {}'.format(kernel_size))

    if sigma is None:
        sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

    if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
        raise TypeError('sigma should be either float or sequence of floats. Got {}'.format(type(sigma)))
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
        sigma = [sigma[0], sigma[0]]
    if len(sigma) != 2:
        raise ValueError('If sigma is a sequence, its length should be 2. Got {}'.format(len(sigma)))
    for s in sigma:
        if s <= 0.:
            raise ValueError('sigma should have positive values. Got {}'.format(sigma))

    t_img = img
    if not isinstance(img, np.ndarray):
        raise TypeError(
            "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
    return F_cv2.gaussian_blur(t_img, kernel_size, sigma)


def rgb_to_grayscale(img: Tensor, num_output_channels: int = 1) -> Tensor:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(rgb_to_grayscale)
    """Convert RGB image to grayscale version of image.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Note:
        Please, note that this method supports only RGB images as input. For inputs in other color spaces,
        please, consider using meth:`~torchvision.transforms.functional.to_grayscale` with PIL Image.

    Args:
        img (PIL Image or Tensor or numpy.ndarray): RGB Image to be converted to grayscale.
        num_output_channels (int): number of channels of the output image. Value can be 1 or 3. Default, 1.

    Returns:
        PIL Image or Tensor or numpy.ndarray: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel

            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if torchvision.get_image_backend() == 'cv2':
        if not isinstance(img, np.ndarray):
            raise TypeError(
                "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
        return F_cv2.to_grayscale(img, num_output_channels)

    if not isinstance(img, torch.Tensor):
        return F_pil.to_grayscale(img, num_output_channels)

    return F_t.rgb_to_grayscale(img, num_output_channels)

