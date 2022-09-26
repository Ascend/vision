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
import unittest
import torch
import torchvision_npu
from torchvision.datasets import folder as fold
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image


class TestCropAndResize(unittest.TestCase):
    def test_resize(self):
        path = "./Data/dog/dog.0001.jpg"
        img = torchvision_npu.datasets.folder.npu_loader(path)
        output = transforms.RandomResizedCrop([224, 220])(img)
        self.assertEqual(output.device.type, 'npu')
        self.assertEqual(output.shape, torch.Size([1, 3, 224, 220]))

    def test_crop_and_resize_npuinput(self):
        path = "./Data/dog/dog.0001.jpg"
        img = torchvision_npu.datasets.folder.npu_loader(path)
        output = transforms.RandomResizedCrop(size=[224, 220], scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))(img)
        self.assertEqual(output.device.type, 'npu')
        self.assertEqual(output.shape, torch.Size([1, 3, 224, 220]))

    def test_crop_and_resize_cpuinput(self):
        path = "./Data/dog/dog.0001.jpg"
        img = fold.default_loader(path)
        output = transforms.RandomResizedCrop(224)(img)
        self.assertEqual(type(output), Image.Image)
        self.assertEqual(output.size, (224, 224))

    def test_crop_and_resize_interpolation_nearest(self):
        path = "./Data/dog/dog.0001.jpg"
        img = torchvision_npu.datasets.folder.npu_loader(path)
        interpolation_list = [InterpolationMode.NEAREST, InterpolationMode.BILINEAR]
        for i in interpolation_list:
            output = transforms.RandomResizedCrop(224, interpolation=i)(img)
            self.assertEqual(output.device.type, 'npu')
            self.assertEqual(output.shape, torch.Size([1, 3, 224, 224]))


if __name__ == '__main__':
    unittest.main()
