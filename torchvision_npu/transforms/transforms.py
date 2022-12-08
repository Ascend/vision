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
import numpy as np
import torch
from torch import Tensor
import torch_npu
import torchvision
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as TRANS
from torchvision.transforms.functional import InterpolationMode


__accelerate__ = ["Compose", "ToTensor", "Normalize", "RandomHorizontalFlip", "RandomResizedCrop"]
__interpolation__ = [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]


def add_transform_methods():
    torchvision.__name__ = 'torchvision_npu'
    torchvision.transforms.Compose.__call__ = Compose.__call__
    torchvision.transforms.ToTensor.__call__ = ToTensor.__call__
    torchvision.transforms.RandomResizedCrop.forward = RandomResizedCrop.forward
    torchvision.transforms.RandomHorizontalFlip.forward = RandomHorizontalFlip.forward
    torchvision.transforms.Normalize.forward = Normalize.forward


class Compose:
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            raise TypeError("accelerate can only support Tensor")
        for t in self.transforms:
            if t.__class__.__name__ in __accelerate__:
                if img.device.type != 'npu':
                    img = img.unsqueeze(0).npu(non_blocking=True)
            else:
                if img.device.type != 'cpu':
                    img = img.cpu().squeeze(0)
            img = t(img)
        return img


class ToTensor:
    @classmethod
    def __call__(self, pic):
        if pic.dtype == torch.uint8:
            return pic.to(dtype=torch.get_default_dtype(), non_blocking=True).div(255)
        return pic


class Normalize(torch.nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:
        mean = torch.tensor(self.mean).npu(non_blocking=True)
        std = torch.tensor(self.std).npu(non_blocking=True)
        if mean.ndim == 1:
            mean = mean.view(1, -1, 1, 1)
        if std.ndim == 1:
            std = std.view(1, -1, 1, 1)
        if not self.inplace:
            return torch_npu.image_normalize(tensor, mean, std, 0)
        return torch_npu.image_normalize_(tensor, mean, std, 0)


class RandomHorizontalFlip(torch.nn.Module):
    def forward(self, img):
        if torch.rand(1) < self.p:
            return torch_npu.reverse(img, axis=[3])
        return img


class RandomResizedCrop(torch.nn.Module):
    def forward(self, img):
        i, j, h, w = TRANS.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        if self.interpolation not in __interpolation__:
            img = img.cpu().squeeze(0)
            return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        width, height = F._get_image_size(img)
        if width <= 1 or height <= 1:
            return img
        boxes = np.minimum([[i / (height - 1), j / (width - 1), (i + h) / (height - 1), \
                            (j + w) / (width - 1)]], 1)
        boxes = torch.tensor(boxes, dtype=torch.float32).npu(non_blocking=True)
        box_index = [0]
        crop_size = self.size
        return torch_npu.crop_and_resize(img, boxes, box_index, crop_size, method=self.interpolation.value)