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
import os
from struct import pack
from typing import Any, Tuple, Callable, Optional

import numpy as np
from PIL import Image

import torch
import torch_npu
import torchvision
from torchvision.datasets import folder as fold
from torchvision_npu.datasets._decode_jpeg import extract_jpeg_shape
from torchvision_npu._utils import PathManager


_npu_set_first = True

_npu_accelerate_list = [
    "ToTensor", "Normalize", "Resize",
    "CenterCrop", "Pad", "RandomCrop",
    "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop", "RandomSizedCrop", "FiveCrop", "TenCrop",
    "ColorJitter", "RandomRotation", "RandomAffine", "Grayscale", "RandomGrayscale",
    "RandomPerspective", "GaussianBlur", "RandomInvert", "RandomPosterize",
    "RandomSolarize"]


def _add_datasets_folder():
    torchvision.__name__ = 'torchvision_npu'
    torchvision.datasets.DatasetFolder = DatasetFolder
    torchvision.datasets.ImageFolder = ImageFolder
    torchvision.datasets.folder.default_loader = default_loader


def _assert_image_3d(img: torch.Tensor):
    if img.ndim != 3:
        raise ValueError('img is not 3D, got shape ({}).'.format(img.shape))


def npu_rollback(transform) -> bool:
    def check_unsupported(t) -> bool:
        if t.__class__.__name__ not in _npu_accelerate_list:
            print("Warning: Cannot accelerate [{}]. Roll back to native PIL implementation."
                .format(t.__class__.__name__), file=sys.stderr)
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
        self.device = "cpu"
        self.backend = torchvision.get_image_backend()
        if self.backend == 'npu':
            if npu_rollback(self.transform):
                self.backend = torchvision.get_image_backend()
                return
            if torch_npu.npu.is_available():
                self.accelerate_enable = True
                self.device = "npu:{}".format(torch_npu.npu.current_device())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            if isinstance(sample, torch.Tensor) and sample.device.type == 'npu':
                sample = sample.cpu().squeeze(0)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def set_accelerate_npu(self, npu: int = -1) -> None:
        """
        Set devive for data preprocecssing process.

        Args:
            npu(int): Device id to set for DP worker process. -1 denotes using the device set by the main process.
        """
        if self.backend == 'npu':
            self.accelerate_enable = True
            self.device = "npu:{}".format(torch_npu.npu.current_device() if npu == -1 else npu)
        else:
            print("Warning: Not Enable NPU", file=sys.stderr)


def _cv2_loader(path: str) -> Any:
    path = os.path.realpath(path)
    PathManager.check_directory_path_readable(path)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img_rgb = img.convert('RGB')
        img.close()
        return np.asarray(img_rgb)


def _npu_loader(path: str) -> Any:
    path = os.path.realpath(path)
    PathManager.check_directory_path_readable(path)
    with open(path, "rb") as f:
        f.seek(0)
        prefix = f.read(16)
        # DVPP only provides DecodeJpeg op currently
        if prefix[:3] == b"\xff\xd8\xff":
            f.seek(0)
            image_shape = extract_jpeg_shape(f)

            f.seek(0)
            bytes_string = f.read()
            arr = np.frombuffer(bytes_string, dtype=np.uint8)
            uint8_tensor = torch.tensor(arr).npu(non_blocking=True)
            channels = 3

            return torch.ops.torchvision._decode_jpeg_aclnn(
                uint8_tensor, image_shape=image_shape, channels=channels)

        # For other imgae types, use PIL to decode, then convert to npu tensor with NCHW format.
        else:
            img = torch.from_numpy(np.array(fold.pil_loader(path)))
            _assert_image_3d(img)
            img = img.permute((2, 0, 1)).contiguous()
            return img.unsqueeze(0).npu(non_blocking=True)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'npu':
        return _npu_loader(path)
    elif get_image_backend() == 'cv2':
        return _cv2_loader(path)
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
