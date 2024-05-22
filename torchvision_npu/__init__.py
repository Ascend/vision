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

import torchvision
from torchvision_npu.datasets import add_datasets_folder
from torchvision_npu.ops.deform_conv import patch_deform_conv

from .extensions import _HAS_OPS
from .ops.roi_pool import patch_roi_pool
from .transforms.functional import patch_transform_methods
from .io import patch_io_methods
from .version import __version__ as __version__


_image_backend = "npu"


def set_image_backend(backend):
    """
    Specifies the package used to load images.

    Args:
        backend (string): Name of the image backend. one of {'npu', 'PIL', 'cv2', 'accimage'}.
            The :mod:`accimage` package uses the Intel IPP library. It is
            generally faster than PIL, but does not support as many operations.
    """
    global _image_backend
    if backend not in ["PIL", "accimage", "npu", "cv2"]:
        raise ValueError(f"Invalid backend '{backend}'. Options are 'npu', 'PIL' , 'cv2' and 'accimage'")
    _image_backend = backend
    print('transform backend: ', torchvision.get_image_backend())
    if torchvision.get_image_backend() == 'cv2':
        print('If you use the cv2 backend, must install opencv-python already and the input must be np.ndarray, '
              'otherwise an exception will be thrown.')


def get_image_backend():
    """
    Gets the name of the package used to load images
    """
    return _image_backend


def patch_init_methods():
    torchvision.set_image_backend = set_image_backend
    torchvision.get_image_backend = get_image_backend


def apply_class_patches():
    patch_init_methods()
    add_datasets_folder()
    patch_transform_methods()
    patch_io_methods()
    patch_roi_pool()
    patch_deform_conv()


apply_class_patches()
