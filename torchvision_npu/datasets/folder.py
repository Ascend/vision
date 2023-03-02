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

import sys
from typing import Any, Tuple, Callable, Optional
import warnings

import numpy as np
import torch
import torch_npu
import torchvision
from PIL import Image

from torchvision.datasets import folder as fold
from torchvision_npu.datasets.decode_jpeg import extract_jpeg_shape, pack


def add_datasets_folder():
    torchvision.__name__ = 'torchvision_npu'
    torchvision.datasets.DatasetFolder = DatasetFolder
    torchvision.datasets.ImageFolder = ImageFolder


_npu_accelerate = ["ToTensor", "Normalize", "RandomHorizontalFlip", "RandomResizedCrop"]


def npu_rollback(transform) -> bool:
    def check_unsupported(t) -> bool:
        if t.__class__.__name__ not in _npu_accelerate:
            warnings.warn("[{}] cannot accelerate. Roll back to native implementation."
                          .format(t.__class__.__name__))
            torchvision.set_image_backend('PIL')
            return True
        return False

    if transform.__class__.__name__ == "Compose":
        for t in transform.transforms:
            if check_unsupported(t):
                return True
        return False

    return check_unsupported(transform)


class DatasetFolder(fold.DatasetFolder):

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root,
                                            loader=loader,
                                            extensions=extensions,
                                            transform=transform,
                                            target_transform=target_transform,
                                            is_valid_file=is_valid_file)

        self.accelerate_enable = False
        self.device = torch.device("cpu")

        if torchvision.get_image_backend() == 'npu':
            if npu_rollback(self.transform):
                return
            if torch_npu.npu.is_available():
                self.accelerate_enable = True
                self.device = torch_npu.npu.current_device()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            if sample.device.type == 'npu':
                sample = sample.cpu().squeeze(0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def set_accelerate_npu(self, npu: int = -1, force_npu: bool = False) -> None:
        """
        Set devive and forcibly enable NPU for data preprocecssing process.

        Args:
            npu(int): Device id to set for DP worker process. -1 denotes using the device set by the main process.
            force_npu(bool): When roll back to native implementation, set this True can forcibly enable NPU.
                             In this case, operators that aren't supported by DVPP run on AICPU.
        """
        if torchvision.get_image_backend() == 'npu' or force_npu:
            torchvision.set_image_backend('npu')
            self.accelerate_enable = True
            self.device = torch_npu.npu.current_device() if npu == -1 else npu
        else:
            warnings.warn("Not Enable NPU")


def cv2_loader(path: str) -> Any:
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return np.asarray(img)


def npu_loader(path: str) -> Any:
    with open(path, "rb") as f:
        f.seek(0)
        prefix = f.read(16)
        if prefix[:3] == b"\xff\xd8\xff":
            f.seek(0)
            image_shape = extract_jpeg_shape(f)

            f.seek(0)
            bytes_string = f.read()
            arr = np.frombuffer(bytes_string, dtype=np.uint8)
            addr = 16
            length = len(bytes_string)
            addr_arr = list(map(int, pack('<Q', addr)))
            len_arr = list(map(int, pack('<Q', length)))
            arr = np.hstack((addr_arr, len_arr, arr, [0]))
            arr = np.array(arr, dtype=np.uint8)
            uint8_tensor = torch.tensor(arr).npu(non_blocking=True)
            channels = 3

            img = torch_npu.decode_jpeg(uint8_tensor, image_shape=image_shape, channels=channels)
            return img.unsqueeze(0)

        else:
            img = torch.from_numpy(np.array(fold.pil_loader(path))).permute((2, 0, 1)).contiguous()
            return img.unsqueeze(0).npu(non_blocking=True)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == 'npu':
        return npu_loader(path)
    elif get_image_backend() == "accimage":
        return fold.accimage_loader(path)
    else:
        return fold.pil_loader(path)


class ImageFolder(fold.ImageFolder, DatasetFolder):

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root=root,
                                          transform=transform,
                                          target_transform=target_transform,
                                          loader=loader,
                                          is_valid_file=is_valid_file)
