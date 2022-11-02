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


def add_transform_methods():
    torchvision.__name__ = 'torchvision_npu'
    torchvision.transforms.ToTensor.__call__ = ToTensor.__call__
    torchvision.transforms.RandomResizedCrop.forward = RandomResizedCrop.forward
    torchvision.transforms.RandomHorizontalFlip.forward = RandomHorizontalFlip.forward
    torchvision.transforms.Normalize.forward = Normalize.forward


class ToTensor:
    @classmethod
    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            if pic.device.type == 'npu':
                return pic.to(dtype=torch.get_default_dtype()).div(255)
        return F.to_tensor(pic)


class Normalize(torch.nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:
        if tensor.device.type == 'npu':
            mean = torch.tensor(self.mean).npu()
            std = torch.tensor(self.std).npu()
            if not self.inplace:
                return torch_npu.image_normalize(tensor, mean, std, 0)
            return torch_npu.image_normalize_(tensor, mean, std, 0)
        return F.normalize(tensor, self.mean, self.std, self.inplace)


class RandomHorizontalFlip(torch.nn.Module):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            if isinstance(img, torch.Tensor):
                if img.device.type == 'npu':
                    return torch_npu.reverse(img, axis=[3])
                else:
                    return F.hflip(img)
        return img


class RandomResizedCrop(torch.nn.Module):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = TRANS.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        if isinstance(img, torch.Tensor):
            if img.device.type == 'npu':
                interpolation = [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]
                if self.interpolation not in interpolation:
                    raise ValueError(f'Tochvision_Npu cannot support this interpolation method')
                width, height = F._get_image_size(img)
                if width == 1 or height == 1:
                    return img
                boxes = np.minimum([[i / (height - 1), j / (width - 1), (i + h) / (height - 1), \
                                   (j + w) / (width - 1)]], 1)
                boxes = torch.as_tensor(boxes, dtype=torch.float32).npu()
                box_index = torch.as_tensor([0], dtype=torch.int32).npu()
                crop_size = self.size
                return torch_npu.crop_and_resize(img, boxes, box_index, crop_size, method=self.interpolation.value)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)



