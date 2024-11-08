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

import torch
import torch_npu
import torchvision
from torchvision_npu.datasets import _add_datasets_folder
from torchvision_npu.ops._deform_conv import patch_deform_conv

from ._extensions import _HAS_OPS
from .ops._roi_pool import patch_roi_pool
from .transforms._functional import patch_transform_methods
from .io import patch_io_methods
from .utils._dataloader import add_dataloader_method
from .version import __version__ as __version__


_image_backend = "PIL"
_video_backend = "pyav"


def _set_image_backend(backend):
    """
    Specifies the package used to load images.

    Args:
        backend (string): Name of the image backend. one of {'npu', 'PIL', 'cv2', 'accimage'}.
            The :mod:`accimage` package uses the Intel IPP library. It is
            generally faster than PIL, but does not support as many operations.
    """
    global _image_backend
    if backend not in ["PIL", "accimage", "npu", "cv2", "moal"]:
        raise ValueError(f"Invalid backend '{backend}'. Options are 'npu', 'PIL' , 'cv2', 'moal' and 'accimage'")
    _image_backend = backend
    if backend == 'npu':
        # Use acldvpp func by default
        torch.npu.set_compile_mode(jit_compile=False)
        torch.ops.torchvision._dvpp_init()
    print('transform image backend: ', torchvision.get_image_backend())
    if torchvision.get_image_backend() == 'cv2':
        print('If you use the cv2 backend, must install opencv-python already and the input must be np.ndarray, '
              'otherwise an exception will be thrown.')


def _get_image_backend():
    """
    Gets the name of the package used to load images
    """
    return _image_backend


def _init_dvpp_video():
    torch_npu.npu.set_device(torch_npu.npu.current_device())
    ret = torch.ops.torchvision._dvpp_sys_init()
    if ret != 0:
        raise RuntimeError(f"_dvpp_sys_init failed {ret}")


def _set_video_backend(backend):
    """
    Specifies the package used to decode videos.

    Args:
        backend (string): Name of the video backend. one of {'pyav', 'video_reader', "cuda", 'npu'}.
            The :mod:`pyav` package uses the 3rd party PyAv library. It is a Pythonic
            binding for the FFmpeg libraries.
            The :mod:`video_reader` package includes a native C++ implementation on
            top of FFMPEG libraries, and a python API of TorchScript custom operator.
            It generally decodes faster than :mod:`pyav`, but is perhaps less robust.
            The :mod:`npu` package uses the npu to process video by DVPP interface.

    .. note::
        Building with FFMPEG is disabled by default in the latest `main`. If you want to use the 'video_reader'
        backend, please compile torchvision from source.
    """
    global _video_backend
    if backend not in ["pyav", "video_reader", "cuda", "npu"]:
        raise ValueError(f"Invalid video backend {backend}. Options are 'pyav', 'video_reader', 'cuda' and 'npu'")
    if backend == "video_reader" and not io._HAS_VIDEO_OPT:
        message = "video_reader video backend is not available. Please compile torchvision from source and try again"
        raise RuntimeError(message)
    elif backend == "cuda" and not io._HAS_GPU_VIDEO_DECODER:
        message = "cuda video backend is not available."
        raise RuntimeError(message)
    elif backend == "npu":
        _init_dvpp_video()
    _video_backend = backend
    print('transform video backend: ', torchvision.get_video_backend())


def _get_video_backend():
    """
    Returns the currently active video backend used to decode videos.

    Returns:
        str: Name of the video backend. one of {'pyav', 'video_reader', 'cuda', 'npu'}.
    """

    return _video_backend


def _patch_init_methods():
    torchvision.set_image_backend = _set_image_backend
    torchvision.get_image_backend = _get_image_backend
    torchvision.set_video_backend = _set_video_backend
    torchvision.get_video_backend = _get_video_backend


def _apply_class_patches():
    _patch_init_methods()
    _add_datasets_folder()
    patch_transform_methods()
    patch_io_methods()
    patch_roi_pool()
    patch_deform_conv()
    add_dataloader_method()


_apply_class_patches()
