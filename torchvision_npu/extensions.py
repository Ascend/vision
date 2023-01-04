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

_HAS_OPS = False

def _has_ops():
    return False


def _register_extensions():
    import os
    import importlib
    import torch
    import torch_npu

    # load the custom_op_library and register the custom ops
    lib_dir = os.path.dirname(__file__)

    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("_C")
    if ext_specs is None:
        raise ImportError
    torch.ops.load_library(ext_specs.origin)


try:
    _register_extensions()
    _HAS_OPS = True

    def _has_ops():  # noqa: F811
        return True
except (ImportError, OSError):
    pass


def _assert_has_ops():
    if not _has_ops():
        raise RuntimeError(
            "Couldn't load custom C++ ops. This can happen if your PyTorch and "
            "torchvision versions are incompatible, or if you had errors while compiling "
            "torchvision from source. For further information on the compatible versions, check "
            "https://github.com/pytorch/vision#installation for the compatibility matrix. "
            "Please check your PyTorch version with torch.__version__ and your torchvision "
            "version with torchvision.__version__ and verify if they are compatible, and if not "
            "please reinstall torchvision so that it matches your PyTorch install."
        )

